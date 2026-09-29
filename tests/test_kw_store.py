"""Append-only JSONL storage: the resume mechanism for overnight runs."""

import os
import time

import pytest

from fairsum.kwsum.manifest import RunManifestMismatch, check_or_write_manifest, file_sha256
from fairsum.kwsum.store import append_jsonl, done_keys, read_jsonl


def test_append_then_read(tmp_path):
    p = tmp_path / "g.jsonl"
    append_jsonl(p, [{"a": 1}])
    append_jsonl(p, [{"a": 2}])
    assert read_jsonl(p) == [{"a": 1}, {"a": 2}]


def test_missing_file_reads_as_empty(tmp_path):
    assert read_jsonl(tmp_path / "none.jsonl") == []


def test_done_keys(tmp_path):
    p = tmp_path / "g.jsonl"
    append_jsonl(p, [{"cid": "c1", "cond": "direct", "model": "m", "x": 0}])
    assert done_keys(p, ("cid", "cond", "model")) == {("c1", "direct", "m")}


def test_a_half_written_last_line_is_ignored_on_read(tmp_path):
    p = tmp_path / "g.jsonl"
    append_jsonl(p, [{"a": 1}])
    with open(p, "a", encoding="utf-8") as fh:
        fh.write('{"a": 2')
    assert read_jsonl(p) == [{"a": 1}]


def test_appending_after_a_crash_drops_the_partial_line(tmp_path):
    p = tmp_path / "g.jsonl"
    append_jsonl(p, [{"a": 1}])
    with open(p, "a", encoding="utf-8") as fh:
        fh.write('{"a": 2')
    append_jsonl(p, [{"a": 3}])
    assert read_jsonl(p) == [{"a": 1}, {"a": 3}]


def test_append_waits_for_another_writers_lock_then_times_out(tmp_path):
    p = tmp_path / "g.jsonl"
    lock = tmp_path / "g.jsonl.lock"
    lock.write_text("held", encoding="utf-8")
    with pytest.raises(TimeoutError, match="g.jsonl.lock"):
        append_jsonl(p, [{"a": 1}], lock_timeout=0.3)
    assert read_jsonl(p) == []


def test_a_stale_lock_is_broken(tmp_path):
    p = tmp_path / "g.jsonl"
    lock = tmp_path / "g.jsonl.lock"
    lock.write_text("held", encoding="utf-8")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    append_jsonl(p, [{"a": 1}], lock_timeout=0.3)
    assert read_jsonl(p) == [{"a": 1}]


def test_the_lock_is_released_after_appending(tmp_path):
    p = tmp_path / "g.jsonl"
    append_jsonl(p, [{"a": 1}])
    assert not (tmp_path / "g.jsonl.lock").exists()


def test_a_corrupt_middle_line_raises_with_its_line_number(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text('{"a": 1}\nnot json\n{"a": 3}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_jsonl(p)


class TestRunManifest:
    """A run's outputs are tied to the exact collections and keywords that made them."""

    def test_first_call_writes_the_manifest(self, tmp_path):
        m = tmp_path / "run_manifest.json"
        check_or_write_manifest(m, {"keywords_sha256": "abc"}, info={"torch": "2.6"})
        assert '"keywords_sha256": "abc"' in m.read_text(encoding="utf-8")

    def test_matching_identity_passes(self, tmp_path):
        m = tmp_path / "run_manifest.json"
        check_or_write_manifest(m, {"keywords_sha256": "abc"})
        check_or_write_manifest(m, {"keywords_sha256": "abc"})

    def test_changed_inputs_refuse_to_resume(self, tmp_path):
        m = tmp_path / "run_manifest.json"
        check_or_write_manifest(m, {"keywords_sha256": "abc"})
        with pytest.raises(RunManifestMismatch, match="keywords_sha256"):
            check_or_write_manifest(m, {"keywords_sha256": "xyz"})

    def test_file_sha256_changes_with_content(self, tmp_path):
        f = tmp_path / "k.jsonl"
        f.write_text("a", encoding="utf-8")
        h1 = file_sha256(f)
        f.write_text("b", encoding="utf-8")
        assert file_sha256(f) != h1
