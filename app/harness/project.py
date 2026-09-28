"""Folders the agents work in.

Each pipeline run gets its own folder under runs/; the analyst has one folder of its
own. Agents reach the read-only data lake through DATA_DIR.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ML_DATA_ROOT", APP_ROOT / "data"))
LAKE = DATA_ROOT / "lake"
TRAFFIC = DATA_ROOT / "traffic"
RUNS = Path(os.environ.get("ML_RUNS_ROOT", APP_ROOT / "runs"))
ANALYSIS = "analysis"
FOLDERS = (
    "src",
    "checks",
    "notes",
    "artifacts",
    "reports",
    "receipts",
    "reviews",
    "logs",
    "features",
)

_current = os.environ.get("ML_PROJECT", "")


def new_run() -> Path:
    """Start a fresh pipeline project folder and make it current."""
    global _current
    _current = time.strftime("run-%Y%m%d-%H%M%S")
    return current()


def use(name: str) -> Path:
    global _current
    _current = name
    return current()


def current_name() -> str:
    return _current


def current() -> Path:
    return folder(_current or "run-default")


def folder(name: str) -> Path:
    path = RUNS / name
    for sub in FOLDERS:
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def resolve(root: Path, relative: str) -> Path:
    """A path inside root. Raises ValueError for anything outside it."""
    base = root.resolve()
    path = (base / relative).resolve()
    if path != base and base not in path.parents:
        raise ValueError(f"{relative!r} is outside the project folder.")
    return path


def file_hash(root: Path, *relatives: str) -> str:
    digest = hashlib.sha256()
    for name in relatives:
        path = root / name
        digest.update(name.encode())
        digest.update(path.read_bytes() if path.is_file() else b"")
    return digest.hexdigest()[:12]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **payload}
    path.write_text(json.dumps(stamped, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}) + "\n"
        )
