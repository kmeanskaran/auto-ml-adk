"""Keep runs/, registry/ and feature_store/ in a Cloud Storage bucket.

A deployed container's disk is lost on a redeploy, a restart or a scale-down, and the
team's work lives in these three folders: the runs (the agents' code, reviews and
trace), the model registry (what /predict serves) and the feature store. With
ML_STATE_BUCKET set, the server copies the folders down from the bucket before anything
reads them, pushes what changed every ML_STATE_SYNC_SECONDS, and once more at shutdown.
A file deleted locally (Clear runs) is deleted from the bucket too. Without the
variable (local runs) nothing happens: the folders are plain bind mounts.

The bucket mirrors one server's disk, so exactly one instance may write it: deploy
with --max-instances 1.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from pathlib import Path

from app.harness.feature_store import STORE
from app.harness.project import RUNS
from app.harness.registry import REGISTRY

log = logging.getLogger(__name__)

ROOTS = {"runs": RUNS, "registry": REGISTRY, "feature_store": STORE}
SKIP = ("__pycache__", ".tmp", ".staging-")  # scratch files never worth keeping


def _skip(key: str) -> bool:
    return any(part in key for part in SKIP)


class StateSync:
    def __init__(self, bucket: str) -> None:
        from google.cloud import storage

        self.name = bucket
        self.bucket = storage.Client().bucket(bucket)
        # key -> (mtime_ns, size) of each file as last pulled or pushed
        self.synced: dict[str, tuple[int, int]] = {}

    def _local(self) -> dict[str, tuple[Path, int, int]]:
        files = {}
        for prefix, root in ROOTS.items():
            if not root.exists():
                continue
            for path in root.rglob("*"):
                key = f"{prefix}/{path.relative_to(root).as_posix()}"
                if _skip(key):
                    continue
                with contextlib.suppress(FileNotFoundError):  # removed mid-scan
                    if path.is_file():
                        stat = path.stat()
                        files[key] = (path, stat.st_mtime_ns, stat.st_size)
        return files

    def pull(self) -> int:
        """Copy the bucket's folders onto this disk. Returns the number of files."""
        count = 0
        for prefix, root in ROOTS.items():
            for blob in self.bucket.list_blobs(prefix=f"{prefix}/"):
                relative = blob.name.removeprefix(f"{prefix}/")
                if not relative or relative.endswith("/") or _skip(blob.name):
                    continue
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(target))
                count += 1
        self.synced = {key: (m, s) for key, (_, m, s) in self._local().items()}
        return count

    def push(self) -> tuple[int, int]:
        """Upload new and changed files, delete removed ones. Returns (up, deleted)."""
        from google.api_core.exceptions import NotFound

        local = self._local()
        uploaded = 0
        for key, (path, mtime, size) in local.items():
            if self.synced.get(key) == (mtime, size):
                continue
            try:
                self.bucket.blob(key).upload_from_filename(str(path))
            except FileNotFoundError:  # removed since the scan
                continue
            # A file still being written changes its mtime again and goes up next time.
            self.synced[key] = (mtime, size)
            uploaded += 1
        gone = [key for key in self.synced if key not in local]
        for key in gone:
            with contextlib.suppress(NotFound):
                self.bucket.blob(key).delete()
            del self.synced[key]
        return uploaded, len(gone)


def start() -> StateSync | None:
    """Pull the bucket's state if ML_STATE_BUCKET is set. Call before the console
    restores its state from runs/."""
    bucket = os.environ.get("ML_STATE_BUCKET", "").removeprefix("gs://").strip("/")
    if not bucket:
        return None
    sync = StateSync(bucket)
    count = sync.pull()
    log.info("State: pulled %d files from gs://%s", count, bucket)
    return sync


async def keep_synced(sync: StateSync) -> None:
    """Push every ML_STATE_SYNC_SECONDS until cancelled, then once more."""
    interval = float(os.environ.get("ML_STATE_SYNC_SECONDS", "10"))
    try:
        while True:
            await asyncio.sleep(interval)
            await _push(sync)
    finally:
        await _push(sync)


async def _push(sync: StateSync) -> None:
    try:
        uploaded, deleted = await asyncio.to_thread(sync.push)
    except Exception:  # a failed sync must not stop the team; the next one retries
        log.exception("State: push to gs://%s failed", sync.name)
        return
    if uploaded or deleted:
        log.info(
            "State: pushed %d, deleted %d in gs://%s", uploaded, deleted, sync.name
        )
