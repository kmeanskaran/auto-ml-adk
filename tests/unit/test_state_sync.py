"""The bucket sync that keeps runs/, registry/ and feature_store/ across redeploys.

A fake in-memory bucket stands in for Cloud Storage; two folders stand in for two
containers: the one that wrote, and the fresh one a redeploy starts.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.harness import state_sync


class FakeBlob:
    def __init__(self, bucket: FakeBucket, name: str) -> None:
        self.bucket, self.name = bucket, name

    def upload_from_filename(self, path: str) -> None:
        self.bucket.objects[self.name] = Path(path).read_bytes()

    def download_to_filename(self, path: str) -> None:
        Path(path).write_bytes(self.bucket.objects[self.name])

    def delete(self) -> None:
        from google.api_core.exceptions import NotFound

        if self.name not in self.bucket.objects:
            raise NotFound(self.name)
        del self.bucket.objects[self.name]


class FakeBucket:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self, name)

    def list_blobs(self, prefix: str = "") -> list[FakeBlob]:
        return [FakeBlob(self, n) for n in sorted(self.objects) if n.startswith(prefix)]


def container(root: Path, bucket: FakeBucket, monkeypatch) -> state_sync.StateSync:
    """A server whose three folders live under root, synced with bucket."""
    roots = {name: root / name for name in ("runs", "registry", "feature_store")}
    monkeypatch.setattr(state_sync, "ROOTS", roots)
    sync = object.__new__(state_sync.StateSync)  # skip the real Cloud Storage client
    sync.name, sync.synced = "fake", {}
    sync.bucket = bucket  # type: ignore[assignment]  # the fake stands in for Cloud Storage
    return sync


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def bucket() -> FakeBucket:
    return FakeBucket()


def test_push_uploads_new_files_and_skips_scratch(tmp_path, bucket, monkeypatch):
    sync = container(tmp_path / "a", bucket, monkeypatch)
    write(tmp_path / "a/runs/run-1/src/train.py", "print(1)")
    write(tmp_path / "a/registry/production.json", '{"version": "v1"}')
    write(tmp_path / "a/runs/run-1/__pycache__/train.cpython-312.pyc", "junk")
    write(tmp_path / "a/runs/console.json.tmp", "half written")

    assert sync.push() == (2, 0)
    assert sorted(bucket.objects) == [
        "registry/production.json",
        "runs/run-1/src/train.py",
    ]
    assert sync.push() == (0, 0)  # nothing changed, nothing sent


def test_push_sends_a_changed_file_again(tmp_path, bucket, monkeypatch):
    sync = container(tmp_path / "a", bucket, monkeypatch)
    model = tmp_path / "a/registry/production.json"
    write(model, '{"version": "v1"}')
    sync.push()

    write(model, '{"version": "v2", "previous": "v1"}')  # size changes
    stat = model.stat()
    os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    assert sync.push() == (1, 0)
    assert b"v2" in bucket.objects["registry/production.json"]


def test_a_fresh_container_gets_everything_back(tmp_path, bucket, monkeypatch):
    first = container(tmp_path / "a", bucket, monkeypatch)
    write(tmp_path / "a/runs/run-1/reports/evaluation.json", "{}")
    write(tmp_path / "a/feature_store/loan/v1/definition.json", "{}")
    first.push()

    redeployed = container(tmp_path / "b", bucket, monkeypatch)
    assert redeployed.pull() == 2
    assert (tmp_path / "b/runs/run-1/reports/evaluation.json").read_text() == "{}"
    assert (tmp_path / "b/feature_store/loan/v1/definition.json").exists()
    assert redeployed.push() == (0, 0)  # what it just pulled is not sent back


def test_a_deleted_file_leaves_the_bucket(tmp_path, bucket, monkeypatch):
    sync = container(tmp_path / "a", bucket, monkeypatch)
    run = tmp_path / "a/runs/run-1/notes/engineer.md"
    write(run, "notes")
    sync.push()

    run.unlink()  # "Clear runs"
    assert sync.push() == (0, 1)
    assert bucket.objects == {}


def test_off_without_a_bucket(monkeypatch):
    monkeypatch.delenv("ML_STATE_BUCKET", raising=False)
    assert state_sync.start() is None
