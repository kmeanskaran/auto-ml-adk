"""Feature store: versioned feature views shared by training and serving.

A feature view version holds three things that must never drift apart:

  definition.json  what each feature means, where it comes from, its statistics
  features.py      the transform, build(raw) -> features, row-wise and stateless
  offline/         the materialized train / valid / test tables training reads

Training reads offline/ through FEATURE_DIR. Serving imports the same
features.py, so a model sees the same transform online as offline. Versions are
immutable and content-addressed: registering the same code and tables twice
returns the existing version (safe when a paused workflow resumes).
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

from app.harness import project
from app.harness.environment import ProjectEnvironment
from app.harness.project import APP_ROOT, read_json, write_json

STORE = Path(os.environ.get("ML_FEATURE_STORE_ROOT", APP_ROOT / "feature_store"))
SPLITS = ("train", "valid", "test")
STAGED = "features"  # run-folder subfolder holding materialized tables before approval
_lock = threading.Lock()

MATERIALIZE = r"""
import importlib.util, json, os, tempfile
import numpy as np
import pandas as pd

def signal(x, y):
    from sklearn.metrics import roc_auc_score
    if y.nunique() < 2 or x.nunique(dropna=True) < 2:
        return None
    if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
        score = x.astype(float).fillna(x.astype(float).median())
    else:
        levels = x.astype(str).where(x.notna(), "<missing>")
        keep = levels.value_counts().index[:50]
        levels = levels.where(levels.isin(keep), "<other>")
        score = levels.map(y.groupby(levels).mean())
    auc = float(roc_auc_score(y, score))
    return round(max(auc, 1 - auc), 3)

problems, stats, rows = [], {}, {}
report = json.load(open("reports/features.json"))
outcome, key = report["outcome_column"], report.get("entity_key") or None
ordered = (report.get("split") or {}).get("ordered_by") or None
declared = list(report["features"])
spec = importlib.util.spec_from_file_location("stage_features", "src/features.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
build = module.build
os.makedirs("features", exist_ok=True)

def transform(raw, where):
    out = build(raw.copy())
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"build() must return a DataFrame ({where})")
    if len(out) != len(raw):
        raise ValueError(f"build() returned {len(out)} rows for {len(raw)} ({where}); it must not drop or add rows")
    if outcome in out.columns:
        raise ValueError(f"build() output contains the outcome column {outcome!r}")
    extra, absent = sorted(set(out.columns) - set(declared)), sorted(set(declared) - set(out.columns))
    if extra or absent:
        raise ValueError(f"build() columns differ from reports/features.json ({where}): undeclared {extra}, missing {absent}")
    return out[declared].reset_index(drop=True)

try:
    for split in ("train", "valid", "test"):
        raw = pd.read_parquet(f"artifacts/{split}.parquet")
        if outcome not in raw.columns:
            raise ValueError(f"artifacts/{split}.parquet has no outcome column {outcome!r}")
        values = set(pd.unique(raw[outcome]).tolist())
        if raw[outcome].isna().any() or not values <= {0, 1}:
            raise ValueError(f"outcome column {outcome!r} in artifacts/{split}.parquet must hold only 0 and 1, found {sorted(map(str, values))[:5]}")
        if key and key not in raw.columns:
            raise ValueError(f"entity_key {key!r} is not a column of artifacts/{split}.parquet")
        if ordered and ordered not in raw.columns:
            raise ValueError(f"split.ordered_by {ordered!r} is not a column of artifacts/{split}.parquet")
        y = raw[outcome].reset_index(drop=True)
        table = transform(raw.drop(columns=[outcome]), split)
        if key:
            table.insert(0, key, raw[key].reset_index(drop=True))
        table[outcome] = y
        table.to_parquet(f"features/{split}.parquet", index=False)
        rows[split] = {"rows": len(table), "outcome_rate": round(float(y.mean()), 4)}
        if split == "train":
            n = max(1, min(200, len(raw) // 2))
            alone = transform(raw.drop(columns=[outcome]).head(n), f"first {n} rows alone")
            together = table[declared].head(n)
            if not alone.astype(str).equals(together.astype(str)):
                raise ValueError(f"build() is not row-wise: the first {n} rows get different values when transformed alone. Move anything fitted on data (encoders, fill values, bins) into the model pipeline.")
            for name in declared:
                col = table[name]
                stats[name] = {"dtype": str(col.dtype), "null_pct": round(100 * float(col.isna().mean()), 2),
                               "unique": int(col.nunique(dropna=True)), "signal_auc": signal(col, y)}
    sample = pd.read_csv(os.environ["SMOKE_SAMPLE"], nrows=50)
    sample = sample[[c for c in sample.columns if not str(c).startswith("Unnamed")]]
    # Served, build() runs alone: from the feature store and the serving bundle, with
    # none of this project's files beside it. Run it that way here.
    here, source = os.getcwd(), os.path.abspath("src/features.py")
    os.chdir(tempfile.mkdtemp())
    try:
        spec = importlib.util.spec_from_file_location("served_features", source)
        served = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(served)
        build = served.build
        transform(sample, "production-shaped records")
    except OSError as exc:
        raise RuntimeError(f"build() read a file ({exc}). It is served alone, without reports/ or any other project file: write what it needs (column lists, mappings) into src/features.py itself.") from exc
    finally:
        os.chdir(here)
except Exception as exc:
    problems.append(f"{type(exc).__name__}: {exc}")
print(json.dumps({"problems": problems, "stats": stats, "tables": rows}))
"""


async def materialize(run: Path) -> list[str]:
    """Run the stage's build() over its tables into run/features/. Returns problems."""
    sample = project.traffic_file()
    if sample is None:
        return ["no production sample available: set traffic in config/config.yml"]
    (run / ".home").mkdir(exist_ok=True)
    (run / ".home" / "materialize.py").write_text(MATERIALIZE, encoding="utf-8")
    result = await ProjectEnvironment(run, {"SMOKE_SAMPLE": str(sample)}).execute(
        f"{shlex.quote(sys.executable)} .home/materialize.py", timeout=900
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if not lines:
        return [
            f"materializing features crashed: {(result.stderr or 'no output')[-600:]}"
        ]
    outcome = json.loads(lines[-1])
    if not outcome["problems"]:
        write_json(
            run / "reports" / "feature_stats.json",
            {"features": outcome["stats"], "tables": outcome["tables"]},
        )
    return outcome["problems"]


def register(run: Path, view: str) -> dict[str, Any]:
    """Store the run's materialized features as a feature view version (idempotent)."""
    report = read_json(run / "reports" / "features.json") or {}
    stats = read_json(run / "reports" / "feature_stats.json") or {}
    files = [run / "src" / "features.py"] + [
        run / STAGED / f"{s}.parquet" for s in SPLITS
    ]
    fingerprint = _fingerprint(files)
    with _lock:
        for existing in versions(view):
            if existing.get("fingerprint") == fingerprint:
                return existing
        number = max((v["number"] for v in versions(view)), default=0) + 1
        staging = STORE / view / f".staging-v{number}"
        shutil.rmtree(staging, ignore_errors=True)
        (staging / "offline").mkdir(parents=True)
        shutil.copy2(run / "src" / "features.py", staging / "features.py")
        for split in SPLITS:
            shutil.copy2(
                run / STAGED / f"{split}.parquet",
                staging / "offline" / f"{split}.parquet",
            )
        definition = {
            "view": view,
            "version": f"v{number}",
            "number": number,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run": run.name,
            "fingerprint": fingerprint,
            "outcome_column": report.get("outcome_column"),
            "entity_key": report.get("entity_key"),
            "split": report.get("split"),
            "rows_removed": report.get("rows_removed"),
            "excluded_columns": report.get("excluded_columns"),
            "tables": stats.get("tables"),
            "features": {
                name: {**meta, **(stats.get("features") or {}).get(name, {})}
                for name, meta in (report.get("features") or {}).items()
            },
        }
        (staging / "definition.json").write_text(
            json.dumps(definition, indent=2), encoding="utf-8"
        )
        staging.rename(STORE / view / f"v{number}")
        freeze(STORE / view / f"v{number}")
    return definition


def freeze(folder: Path) -> None:
    """Make a registered version read-only: agents' scripts read FEATURE_DIR, and a
    script that tries to write there fails instead of changing a frozen version."""
    for path in sorted(folder.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    folder.chmod(0o555)


def versions(view: str) -> list[dict[str, Any]]:
    """Registered versions, oldest first. A definition.json without the harness's
    fields (overwritten by hand or by a script) is skipped, never trusted."""
    found = [_definition(p) for p in (STORE / view).glob("v*")]
    return sorted(
        (v for v in found if v and isinstance(v.get("number"), int)),
        key=lambda v: v["number"],
    )


def _definition(folder: Path) -> dict[str, Any] | None:
    try:
        data = read_json(folder / "definition.json")
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def path(view: str, version: str) -> Path:
    return STORE / view / version


def summary() -> list[dict[str, Any]]:
    """Every view with its versions, newest first, for the console."""
    views = (
        sorted(p.name for p in STORE.glob("*") if p.is_dir()) if STORE.is_dir() else []
    )
    return [
        {
            "view": view,
            "versions": [
                {
                    "version": v["version"],
                    "created_at": v["created_at"],
                    "run": v["run"],
                    "features": len(v.get("features") or {}),
                    "split": (v.get("split") or {}).get("method"),
                    "tables": v.get("tables") or {},
                    "excluded_columns": v.get("excluded_columns") or {},
                    "columns": [
                        {
                            "name": name,
                            "description": meta.get("description", ""),
                            "source": meta.get("source") or [],
                            "dtype": meta.get("dtype"),
                            "null_pct": meta.get("null_pct"),
                            "signal_auc": meta.get("signal_auc"),
                            "known_at_prediction": meta.get("known_at_prediction"),
                        }
                        for name, meta in (v.get("features") or {}).items()
                    ],
                }
                for v in reversed(versions(view))
            ],
        }
        for view in views
    ]


def _fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()[:16]
