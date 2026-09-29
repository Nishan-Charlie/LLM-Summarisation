"""Run manifest: ties a run's outputs to the exact inputs that produced them.

`kw_build_collections.py` rewrites collections and keywords in place. Without
a manifest, a resumed generation run after a rebuild would silently mix rows
made from two different keyword sets (final review, finding I5).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


class RunManifestMismatch(ValueError):
    """The inputs changed since this run started; appending would mix two runs."""


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def input_identity(collections_path: str | Path, keywords_path: str | Path) -> dict[str, str]:
    return {"collections_sha256": file_sha256(collections_path),
            "keywords_sha256": file_sha256(keywords_path)}


def environment_info() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("torch", "transformers", "bitsandbytes", "spacy", "sentence-transformers",
                "scikit-learn", "huggingface_hub"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "missing"
    return out


def check_or_write_manifest(path: str | Path, identity: Mapping[str, str],
                            info: Mapping[str, object] | None = None) -> None:
    """Write the manifest on first use; afterwards every identity key must match.

    `info` (package versions, decoding settings) is recorded on first write
    only, and never compared.
    """
    path = Path(path)
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        changed = [k for k, v in identity.items() if recorded.get("identity", {}).get(k) != v]
        if changed:
            raise RunManifestMismatch(
                f"{path}: inputs changed since this run started ({', '.join(changed)}); "
                "start a new run_id or restore the original files"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "identity": dict(identity),
        "info": dict(info or {}),
    }, indent=2), encoding="utf-8")
