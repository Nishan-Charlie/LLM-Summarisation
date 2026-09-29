"""Append-only JSONL: how overnight runs survive being interrupted.

Rows are appended as each chunk finishes. A crash mid-write leaves at most one
partial last line. Reading ignores it, and the next append cuts it off first so
it never becomes a corrupt middle line.

Appends take an exclusive lock file, so two writers (say an API model and a
local model generating at once) cannot interleave, and one writer's tail repair
can never cut off another's line in flight (final review, finding I6).
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence

STALE_LOCK_SECONDS = 600


@contextmanager
def _locked(path: Path, timeout: float) -> Iterator[None]:
    lock = path.with_name(path.name + ".lock")
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > STALE_LOCK_SECONDS:
                    lock.unlink(missing_ok=True)  # left behind by a killed writer
                    continue
            except FileNotFoundError:
                continue
            if time.time() > deadline:
                raise TimeoutError(f"could not acquire {lock} within {timeout}s") from None
            time.sleep(0.05)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _repair_tail(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as fh:
        fh.seek(-1, 2)
        if fh.read(1) == b"\n":
            return
        fh.seek(0)
        data = fh.read()
        fh.seek(data.rfind(b"\n") + 1)
        fh.truncate()


def append_jsonl(path: str | Path, rows: Iterable[dict], lock_timeout: float = 120.0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with _locked(path, lock_timeout):
        _repair_tail(path)
        with open(path, "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if number == len(lines):
                break  # partial write from an interrupted run
            raise ValueError(f"{path}: corrupt JSON on line {number}") from exc
    return rows


def done_keys(path: str | Path, fields: Sequence[str]) -> set[tuple]:
    return {tuple(row[f] for f in fields) for row in read_jsonl(path)}
