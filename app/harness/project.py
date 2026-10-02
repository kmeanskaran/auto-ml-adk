"""Folders the agents work in.

Each pipeline run gets its own folder under runs/; the analyst has one folder of its
own. Agents reach the read-only data folder through DATA_DIR: config/config.yml names
the labelled dataset and the production records in it.

Which run a step belongs to is stored in the session state ("run"), not in the
process: every node and tool binds it from the state before touching a file, so
sessions running side by side, or resumed by another server, write to their own
folder. The process-wide pointer (use/current_name) only says which run the console
shows, and is the fallback outside a session.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ML_DATA_ROOT", APP_ROOT / "data"))
LAKE = DATA_ROOT  # config dataset and traffic paths are relative to it
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
    "charts",
)

RUN_KEY = "run"  # session state key holding the run's folder name
FEEDBACK_KEY = "feedback"  # session state key holding the human's feedback for the run
INDEX = "index.jsonl"  # one line per run event, under runs/

_current = os.environ.get("ML_PROJECT", "")
_bound: ContextVar[str] = ContextVar("ml_run", default="")


def new_run() -> Path:
    """Start a fresh pipeline project folder, bind it, and point the console at it.

    The caller stores current_name() in the session state under RUN_KEY."""
    global _current
    name = time.strftime("run-%Y%m%d-%H%M%S")
    if (RUNS / name).exists():  # two runs started in the same second
        name += f"-{uuid.uuid4().hex[:4]}"
    _current = name
    _bound.set(name)
    return current()


def bind(state: Any) -> Path:
    """Work in the run named by this session's state, if it names one."""
    name = state.get(RUN_KEY) if state is not None else None
    if name:
        _bound.set(str(name))
    return current()


def use(name: str) -> Path:
    global _current
    _current = name
    return current()


def traffic_file() -> Path | None:
    """The production records (no outcome) named by config traffic, if present."""
    from app import settings

    name = settings.load().traffic
    path = DATA_ROOT / name if name else None
    return path if path and path.is_file() else None


def forget() -> None:
    """Point the console at no run (after a cold start), without creating a folder."""
    global _current
    _current = ""


def current_name() -> str:
    return _bound.get() or _current


def current() -> Path:
    return folder(current_name() or "run-default")


def home(agent_name: str) -> Path:
    """The analyst works in its own folder; pipeline agents in the current run's."""
    return folder(ANALYSIS) if agent_name == "analyst" else current()


def folder(name: str) -> Path:
    path = RUNS / name
    for sub in FOLDERS:
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def empty(path: Path) -> None:
    """Delete everything inside a folder, keeping the folder (it may be a mount)."""
    if not path.is_dir():
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            for inner in [child, *child.rglob("*")]:  # frozen versions are read-only
                if inner.is_dir() and not inner.is_symlink():
                    inner.chmod(0o755)
            shutil.rmtree(child)
        else:
            child.unlink()


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
