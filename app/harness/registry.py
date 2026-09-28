"""Model registry: immutable versions, one production alias, and an audit log.

  registry/vN/version.json   lineage and metrics: run, feature view version, plan,
                             chosen model, threshold, every metric on valid and test
  registry/vN/src/           serving code (predict.py + the feature view's features.py)
                             and, for lineage, the data.py and train.py that built it
  registry/vN/artifacts/     model.joblib and model.json (threshold, feature list)
  registry/production.json   which version /predict serves
  registry/audit.jsonl       every register, promote and rollback

A version is either a candidate, production, or archived (was production once).
Registering and promoting are idempotent, so a workflow that resumes and runs a
step again cannot create a second version or promote twice.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

from app.harness.evaluation import MODELS_DIR, bundle
from app.harness.project import APP_ROOT, append_jsonl, read_json

REGISTRY = Path(os.environ.get("ML_REGISTRY_ROOT", APP_ROOT / "registry"))
_lock = threading.Lock()
_loaded: dict[str, Any] = {}


class NotFound(LookupError):
    """No such version, or nothing in production. Distinct from a KeyError in a record."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _number(path: Path) -> int:
    return int(path.name[1:]) if path.name[1:].isdigit() else 0


def live_name() -> str | None:
    alias = read_json(REGISTRY / "production.json")
    return alias.get("version") if alias else None


def versions() -> list[dict[str, Any]]:
    """All versions, newest first, each with its current status."""
    live = live_name()
    found = []
    for folder in sorted(REGISTRY.glob("v*"), key=_number, reverse=True):
        version = read_json(folder / "version.json")
        if not version:
            continue
        status = version.get("status", "archived")  # versions from before statuses
        if version["version"] == live:
            status = "production"
        elif status == "production":
            status = "archived"
        found.append({**version, "status": status})
    return found


def get(name: str) -> dict[str, Any] | None:
    return next((v for v in versions() if v["version"] == name), None)


def production() -> dict[str, Any] | None:
    live = live_name()
    return get(live) if live else None


def register(
    run: Path,
    model: str,
    evaluation: dict[str, Any],
    feature_ref: dict[str, str],
    feature_dir: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Freeze one trained candidate as a new version (or return the existing one)."""
    artifact_hash = _hash(run / MODELS_DIR / f"{model}.joblib")
    with _lock:
        for existing in versions():
            if (
                existing.get("run") == run.name
                and existing.get("artifact_hash") == artifact_hash
            ):
                return existing
        number = max((_number(p) for p in REGISTRY.glob("v*")), default=0) + 1
        name = f"v{number}"
        staging = REGISTRY / f".staging-{name}"
        shutil.rmtree(staging, ignore_errors=True)
        bundle(run, model, evaluation, feature_dir, staging, bundle_id=name)
        scores = evaluation["candidates"][model]
        version = {
            "version": name,
            "status": "candidate",
            "created_at": _now(),
            "run": run.name,
            "model": model,
            "threshold": scores["threshold"],
            "metric": evaluation["metric"],
            "valid": scores["valid"],
            "test": scores["test"],
            "baseline_test": evaluation["baseline"]["test"],
            "costs": evaluation["costs"],
            "feature_view": feature_ref,
            "plan": plan,
            "artifact_hash": artifact_hash,
        }
        (staging / "version.json").write_text(
            json.dumps(version, indent=2), encoding="utf-8"
        )
        REGISTRY.mkdir(parents=True, exist_ok=True)
        staging.rename(REGISTRY / name)
        append_jsonl(
            REGISTRY / "audit.jsonl",
            {"event": "register", "version": name, "run": run.name},
        )
    return version


def promote(name: str, reason: str = "promoted") -> dict[str, Any]:
    """Point production at a version. Promoting the live version again is a no-op."""
    with _lock:
        target = REGISTRY / name / "version.json"
        version = read_json(target)
        if not version or version.get("status") == "removed":
            raise NotFound(f"No version {name!r} in the registry.")
        previous = live_name()
        if previous == name:
            return get(name) or version
        _set_status(previous, "archived")
        _set_status(name, "production", promoted_at=_now())
        (REGISTRY / "production.json").write_text(
            json.dumps(
                {"version": name, "since": _now(), "previous": previous}, indent=2
            ),
            encoding="utf-8",
        )
        append_jsonl(
            REGISTRY / "audit.jsonl",
            {"event": reason, "version": name, "previous": previous},
        )
    return get(name) or version


def remove(name: str) -> dict[str, Any]:
    """Take a version out of the library. Its files stay for the audit trail, but it can
    no longer be served or made live. The production version cannot be removed."""
    with _lock:
        version = read_json(REGISTRY / name / "version.json")
        if not version:
            raise NotFound(f"No version {name!r} in the registry.")
        if name == live_name():
            raise ValueError(
                f"{name} is in production; make another version live first."
            )
        if version.get("status") != "removed":
            _set_status(name, "removed", removed_at=_now())
            _loaded.pop(name, None)
            append_jsonl(REGISTRY / "audit.jsonl", {"event": "remove", "version": name})
    return get(name) or version


def audit(limit: int = 20) -> list[dict[str, Any]]:
    path = REGISTRY / "audit.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line][
        -limit:
    ][::-1]


def predict(rows: list[dict[str, Any]], name: str | None = None) -> dict[str, Any]:
    """Score raw records with a version's own predict.py (production by default)."""
    name = name or live_name()
    if not name:
        raise NotFound("No model is in production yet.")
    if (read_json(REGISTRY / name / "version.json") or {}).get("status") == "removed":
        raise NotFound(f"Version {name!r} was removed from the registry.")
    if name not in _loaded:
        path = REGISTRY / name / "src" / "predict.py"
        if not path.is_file():
            raise NotFound(f"No version {name!r} in the registry.")
        spec = importlib.util.spec_from_file_location(f"registry_{name}_predict", path)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(
            0, str(path.parent)
        )  # versions from before bundles import siblings
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(path.parent))
        _loaded[name] = module
    return {"version": name, "predictions": _plain(_loaded[name].predict(rows))}


def _set_status(name: str | None, status: str, **extra: Any) -> None:
    if not name:
        return
    path = REGISTRY / name / "version.json"
    version = read_json(path)
    if version:
        path.write_text(
            json.dumps({**version, "status": status, **extra}, indent=2),
            encoding="utf-8",
        )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _plain(value: Any) -> Any:
    """numpy and pandas scalars as plain JSON values."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value.item() if hasattr(value, "item") else value
