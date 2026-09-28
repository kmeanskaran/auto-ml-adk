"""What each pipeline stage must hand over, checked in code.

Checks look at structure and completeness only: files and keys present, the
feature transform materializing cleanly and row-wise, every planned model
loading and scoring, the serving bundle answering production-shaped records.
Whether the work is right is the skeptic's job, and the human's.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from app import settings
from app.harness import catalog, evaluation, feature_store
from app.harness.project import read_json

STAGES = ("profile", "features", "model")

FEATURES_FORMAT = """reports/features.json:
{
  "outcome_column": "<name>",
  "entity_key": "<raw column that identifies a record>" | null,
  "split": {"method": "<str>", "reason": "<str>"},
  "rows_removed": {"<reason>": <count>},
  "excluded_columns": {"<raw column>": "<reason>"},
  "features": {"<EVERY column build() returns>": {
      "source": ["<raw column>", ...],
      "description": "<one line>",
      "known_at_prediction": "yes" | "unclear"}}
}
plus artifacts/train.parquet, artifacts/valid.parquet, artifacts/test.parquet written by
src/data.py (raw-shaped rows, with the outcome column), and src/features.py defining
build(raw: pandas.DataFrame) -> pandas.DataFrame: row-wise and stateless (nothing fitted
on data), never reading the outcome, returning exactly the declared features. It runs on
production records too, which have the raw columns but no outcome."""

MODEL_FORMAT = """For every model key in reports/plan.json, artifacts/models/<key>.joblib: a fitted
object with predict_proba(X), where X is a DataFrame of the feature view's feature
columns (FEATURE_DIR/definition.json "features", in that order). Anything fitted on data
(imputers, encoders, scalers) lives inside it, e.g. a sklearn Pipeline. Train on
FEATURE_DIR/offline/train.parquet; valid.parquet is for tuning. Do not compute final
metrics or thresholds: the harness evaluates every candidate on valid and test."""

FORMATS = {"profile": "", "features": FEATURES_FORMAT, "model": MODEL_FORMAT}


async def problems(root: Path, stage: str) -> list[str]:
    """Everything missing or malformed in a stage's hand-over. Empty means complete."""
    if stage == "profile":
        return (
            []
            if read_json(root / "reports" / "summary.json")
            else ["no summary submitted: call submit_summary"]
        )
    if stage == "features":
        return await _features(root)
    return await _model(root)


async def _features(root: Path) -> list[str]:
    report, issues = _load(root / "reports" / "features.json")
    if report is None:
        return issues
    issues = _require(
        report,
        {
            "outcome_column": str,
            "split": dict,
            "rows_removed": dict,
            "excluded_columns": dict,
            "features": dict,
        },
    )
    if issues:
        return issues
    issues += [
        f"split.{k} is required"
        for k in ("method", "reason")
        if not report["split"].get(k)
    ]
    if not report["features"]:
        issues.append("features is empty")
    for name, entry in report["features"].items():
        if (
            not isinstance(entry, dict)
            or not entry.get("description")
            or not isinstance(entry.get("source"), list)
        ):
            issues.append(f"features[{name!r}] needs source (list) and description")
        elif entry.get("known_at_prediction") not in ("yes", "unclear"):
            issues.append(
                f"features[{name!r}].known_at_prediction must be yes or unclear; "
                "drop features not known at prediction time"
            )
    for split in feature_store.SPLITS:
        if not (root / "artifacts" / f"{split}.parquet").is_file():
            issues.append(f"artifacts/{split}.parquet does not exist")
    if not (root / "src" / "features.py").is_file():
        issues.append("src/features.py does not exist")
    if not read_json(root / "reports" / "plan_proposal.json"):
        issues.append("no training plan proposed: call propose_plan")
    if issues:
        return issues
    return await feature_store.materialize(root)


async def _model(root: Path) -> list[str]:
    plan = read_json(root / "reports" / "plan.json")
    ref = read_json(root / "reports" / "feature_ref.json")
    if not plan or not ref:
        return [
            "the training plan or feature view is missing; this stage cannot be checked"
        ]
    config = settings.load()
    report, issues = await evaluation.evaluate(
        root, Path(ref["path"]), plan, (config.cost_missed, config.cost_false_alarm)
    )
    if issues or not report or not report["best"]:
        return issues or ["no candidate model could be evaluated"]
    staging = root / ".home" / "bundle"
    shutil.rmtree(staging, ignore_errors=True)
    evaluation.bundle(root, report["best"], report, Path(ref["path"]), staging, "smoke")
    smoke = await evaluation.smoke(staging)
    if not smoke.get("ok"):
        return [
            f"the serving bundle for {report['best']!r} failed on production-shaped "
            f"records: {smoke.get('error')}"
        ]
    return []


def plan_problems(models: list[str], metric: str) -> list[str]:
    known = catalog.models()
    issues = [
        f"unknown model {m!r}; choose from {sorted(known)}"
        for m in models
        if m not in known
    ]
    if not models:
        issues.append("choose at least one model")
    if metric not in catalog.METRICS:
        issues.append(
            f"unknown metric {metric!r}; choose from {sorted(catalog.METRICS)}"
        )
    return issues


def _load(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.is_file():
        return None, [f"{path.parent.name}/{path.name} does not exist"]
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [f"{path.name} is not valid JSON: {exc}"]
    return (
        (report, [])
        if isinstance(report, dict)
        else (None, [f"{path.name} must hold a JSON object"])
    )


def _require(report: dict[str, Any], fields: dict[str, Any]) -> list[str]:
    issues = []
    for key, kind in fields.items():
        if key not in report:
            issues.append(f"missing key {key!r}")
        elif not isinstance(report[key], kind) or isinstance(report[key], bool):
            issues.append(f"{key!r} has the wrong type")
    return issues
