"""Lifecycle stages the SME board walks through, one gate at a time."""

from __future__ import annotations

from typing import Any

from team.phase import latest_sme_note

STAGE_ORDER = ("stats", "prepare", "model", "train", "deliver")

STAGES: dict[str, dict[str, Any]] = {
    "stats": {
        "title": "Statistics",
        "summary": "Schema, mean, median, mode, std, outliers, and the target.",
        "phases": (
            "frame_technical_spec",
            "inspect_schema",
            "profile_quality",
            "analyze_target",
            "write_business_brief",
            "compile_research_brief",
        ),
    },
    "prepare": {
        "title": "Data prep",
        "summary": "Clean the table, then load the scores that fit this question.",
        "phases": ("clean_dataset",),
    },
    "model": {
        "title": "Features & models",
        "summary": "Feature spec and two model families. Pick one to train.",
        "phases": ("create_features", "select_models"),
    },
    "train": {
        "title": "Training",
        "summary": "Fit candidates, keep the best F1, compare with the target.",
        "phases": ("run_training", "evaluate_model", "record_improvement_plan"),
    },
    "deliver": {
        "title": "Delivery",
        "summary": "Metadata, serving contract, and the solution note.",
        "phases": (
            "write_model_metadata",
            "publish_serving_contract",
            "write_solution_design",
        ),
    },
}


def next_stage_id(stage_id: str | None) -> str | None:
  if not stage_id:
    return STAGE_ORDER[0]
  try:
    index = STAGE_ORDER.index(stage_id)
  except ValueError:
    return STAGE_ORDER[0]
  if index + 1 >= len(STAGE_ORDER):
    return None
  return STAGE_ORDER[index + 1]


def phase_result(record: dict[str, Any], name: str) -> dict[str, Any]:
  return (record.get("phases") or {}).get(name, {}).get("result") or {}


def stage_done(record: dict[str, Any], stage_id: str) -> bool:
  phases = STAGES[stage_id]["phases"]
  stored = record.get("phases") or {}
  if stage_id == "train":
    return bool(stored.get("evaluate_model", {}).get("status") == "ok")
  required = [name for name in phases if name != "record_improvement_plan"]
  return all(stored.get(name, {}).get("status") == "ok" for name in required)


def compact_report(stage_id: str, record: dict[str, Any]) -> dict[str, Any]:
  if stage_id == "stats":
    return _stats_report(record)
  if stage_id == "prepare":
    return _prepare_report(record)
  if stage_id == "model":
    return _model_report(record)
  if stage_id == "train":
    return _train_report(record)
  if stage_id == "deliver":
    return _deliver_report(record)
  return {"numbers": [], "findings": [], "notes": []}


def board(record: dict[str, Any]) -> dict[str, Any]:
  current = record.get("current_stage") or STAGE_ORDER[0]
  checkpoints = list(record.get("checkpoints") or STAGE_ORDER)
  blocks = []
  for stage_id in STAGE_ORDER:
    status = _block_status(record, stage_id, current)
    item = {
        "id": stage_id,
        "title": STAGES[stage_id]["title"],
        "summary": STAGES[stage_id]["summary"],
        "checkpoint": stage_id in checkpoints,
        "status": status,
        "report": compact_report(stage_id, record) if status in {"review", "done", "awaiting_approval"} else None,
    }
    blocks.append(item)
  return {
      "id": record.get("id"),
      "status": record.get("status"),
      "mode": record.get("mode") or "full",
      "current_stage": current,
      "checkpoints": checkpoints,
      "objective": record.get("objective") or {},
      "error": record.get("error"),
      "pending_approval": record.get("pending_approval"),
      "target_met": record.get("target_met"),
      "best": record.get("best"),
      "notes": record.get("notes") or [],
      "sme_notes": record.get("sme_notes") or [],
      "stages": blocks,
      "can_next": record.get("status") == "awaiting_review" and not (
          current == "model" and not record.get("chosen_models")
      ),
      "can_retry": record.get("status") in {"failed", "interrupted"},
      "can_rework": record.get("status") in {"awaiting_review", "awaiting_approval"},
      "can_approve": record.get("status") == "awaiting_approval",
      "can_choose_model": (
          record.get("status") == "awaiting_review" and current == "model"
      ),
      "choices": _model_choices(record),
      "chosen_models": list(record.get("chosen_models") or []),
  }


def _block_status(record: dict[str, Any], stage_id: str, current: str) -> str:
  status = record.get("status")
  if status == "failed" and current == stage_id:
    return "failed"
  if status == "interrupted" and current == stage_id:
    return "interrupted"
  if status == "running" and current == stage_id:
    return "running"
  if status == "awaiting_approval" and current == stage_id:
    return "awaiting_approval"
  if status == "awaiting_review" and current == stage_id:
    return "review"
  if stage_done(record, stage_id):
    return "done"
  current_index = STAGE_ORDER.index(current) if current in STAGE_ORDER else 0
  this_index = STAGE_ORDER.index(stage_id)
  if this_index < current_index:
    return "done"
  return "pending"


def _stats_report(record: dict[str, Any]) -> dict[str, Any]:
  schema = phase_result(record, "inspect_schema")
  quality = phase_result(record, "profile_quality")
  target = phase_result(record, "analyze_target")
  tech = phase_result(record, "frame_technical_spec")
  columns = schema.get("columns") or []
  classes = target.get("classes") or {}
  numbers = [
      {"label": "Rows", "value": schema.get("rows")},
      {"label": "Columns", "value": len(columns)},
      {"label": "Numeric", "value": quality.get("numeric_predictors")},
      {"label": "Categorical", "value": quality.get("categorical_predictors", 0)},
      {"label": "Outliers", "value": quality.get("outlier_cells", 0)},
      {"label": "Missing cells", "value": quality.get("missing_cells", 0)},
  ]
  findings = _note_findings(record, "stats", quality.get("categorical_summary"))
  if tech.get("target_column"):
    findings.append(f"Target `{tech.get('target_column')}` ({tech.get('task_type') or 'classification'}).")
  if classes:
    parts = ", ".join(f"{name}: {count}" for name, count in list(classes.items())[:6])
    findings.append(f"Class counts — {parts}.")
  spread = _spread_finding(quality)
  if spread:
    findings.append(spread)
  outliers = quality.get("outlier_columns") or []
  if outliers:
    top = ", ".join(f"{item['name']} ({item['outliers']})" for item in outliers[:4])
    findings.append(f"IQR outliers: {quality.get('outlier_cells', 0)} cells. Highest in {top}.")
  elif quality.get("numeric_predictors"):
    findings.append("IQR outliers: 0.")
  missing = quality.get("missing_by_column") or {}
  if missing:
    top = ", ".join(f"{name} ({count})" for name, count in list(missing.items())[:4])
    findings.append(f"Gaps in {top}.")
  duplicates = quality.get("duplicate_id_rows") or 0
  if duplicates:
    findings.append(f"Duplicate id rows: {duplicates}.")
  constants = quality.get("constant_columns") or []
  if constants:
    findings.append(f"Constant columns: {', '.join(constants[:6])}.")
  if not findings:
    findings.append("Table profiled. No quality flags.")
  notes = []
  if tech.get("technical_problem"):
    notes.append(tech["technical_problem"])
  return {"numbers": numbers, "findings": findings, "notes": notes}


_METRIC_LABELS = {
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
    "accuracy": "Accuracy",
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
}


def _prepare_report(record: dict[str, Any]) -> dict[str, Any]:
  cleaning = phase_result(record, "clean_dataset")
  fills = cleaning.get("fills") or {}
  loaded = [str(name) for name in (cleaning.get("metrics") or [])]
  numbers = [
      {"label": "Rows in", "value": cleaning.get("rows_before")},
      {"label": "Rows out", "value": cleaning.get("rows_after")},
      {"label": "Duplicates dropped", "value": cleaning.get("duplicate_rows_removed", 0)},
      {"label": "Missing after", "value": cleaning.get("missing_after", 0)},
  ]
  if cleaning.get("task_type"):
    numbers.append({"label": "Task", "value": str(cleaning["task_type"]).replace("_", " ")})
  if loaded:
    numbers.append(
        {
            "label": "Scores",
            "value": ", ".join(_METRIC_LABELS.get(name, name) for name in loaded),
        }
    )
  quality = phase_result(record, "profile_quality")
  findings = _note_findings(record, "prepare", quality.get("categorical_summary"))
  if cleaning.get("metric_reason"):
    findings.insert(0, str(cleaning["metric_reason"]))
  dropped = cleaning.get("rows_missing_target_removed") or 0
  if dropped:
    findings.append(f"Removed {dropped} rows with a missing target.")
  if fills:
    findings.append(
        "Filled " + ", ".join(f"{name}←{value}" for name, value in list(fills.items())[:5]) + "."
    )
  if not findings:
    findings.append("Table was already clean enough to train.")
  return {"numbers": numbers, "findings": findings, "notes": []}


def _model_choices(record: dict[str, Any]) -> list[dict[str, Any]]:
  selection = phase_result(record, "select_models")
  return list(selection.get("options") or [])


def _model_report(record: dict[str, Any]) -> dict[str, Any]:
  features = phase_result(record, "create_features")
  selection = phase_result(record, "select_models")
  engineered = features.get("engineered") or []
  chosen = list(record.get("chosen_models") or [])
  options = selection.get("options") or []
  numbers = [
      {"label": "Numeric", "value": len(features.get("numeric") or [])},
      {"label": "Categorical", "value": len(features.get("categorical") or [])},
      {"label": "Engineered", "value": len(engineered)},
      {"label": "Picked", "value": chosen[0] if chosen else "—"},
  ]
  summary = selection.get("categorical_summary") or features.get("categorical_summary")
  findings = _note_findings(record, "model", summary)
  if engineered:
    findings.append("Added " + ", ".join(f"`{name}`" for name in engineered) + ".")
  for item in options[:2]:
    findings.append(f"{item.get('title')}: {item.get('reason')}")
  if chosen:
    findings.append(f"SME picked `{chosen[0]}`.")
  notes = [selection["decision"]] if selection.get("decision") else []
  return {
      "numbers": numbers,
      "findings": findings or ["Feature spec written."],
      "notes": notes,
      "choices": options,
  }


def _train_report(record: dict[str, Any]) -> dict[str, Any]:
  evaluation = phase_result(record, "evaluate_model")
  training = phase_result(record, "run_training")
  best = record.get("best") or {}
  metrics = evaluation.get("metrics") or best.get("metrics") or training.get("metrics") or {}
  if evaluation.get("f1") is not None:
    metrics = {
        "f1": evaluation.get("f1"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "accuracy": metrics.get("accuracy"),
    }
  objective = record.get("objective") or {}
  wanted = [str(name) for name in (objective.get("metrics") or ["f1", "precision", "recall"])]
  numbers = [
      {"label": _METRIC_LABELS[name], "value": _metric(metrics.get(name))}
      for name in wanted
      if name in _METRIC_LABELS
  ]
  numbers.append(
      {
          "label": "Target",
          "value": evaluation.get("target") or objective.get("target"),
      }
  )
  features = phase_result(record, "create_features")
  model_name = evaluation.get("model") or best.get("model")
  findings = []
  if model_name:
    findings.append(f"Trained `{model_name}`.")
  findings.extend(_note_findings(record, "train", features.get("categorical_summary")))
  if evaluation.get("target_met") or record.get("target_met"):
    findings.append("Target met. Loop can stop.")
  else:
    findings.append("Below target. Next strategy is queued if you continue.")
  winner = training.get("winner")
  if winner:
    findings.append(f"Kept `{winner}` from the candidate panel.")
  return {"numbers": numbers, "findings": findings, "notes": []}


def _deliver_report(record: dict[str, Any]) -> dict[str, Any]:
  metadata = phase_result(record, "write_model_metadata")
  serving = phase_result(record, "publish_serving_contract")
  numbers = [
      {"label": "Algorithm", "value": metadata.get("algorithm")},
      {"label": "Threshold", "value": metadata.get("decision_threshold")},
      {"label": "Features", "value": len(metadata.get("feature_columns") or [])},
  ]
  features = phase_result(record, "create_features")
  findings = _note_findings(record, "deliver", features.get("categorical_summary"))
  if serving.get("path"):
    findings.append(f"Serve {serving.get('method')} `{serving.get('path')}`.")
  if record.get("design_path"):
    findings.append("DESIGN.md is ready.")
  return {"numbers": numbers, "findings": findings or ["Delivery artifacts written."], "notes": []}


def _note_findings(
    record: dict[str, Any],
    stage_id: str,
    summary: str | None,
) -> list[str]:
  """Put the SME rework prompt in front of the stage reading."""
  note = latest_sme_note(record, stage_id)
  if note and "categor" in note.lower():
    detail = summary or "Categorical features were checked."
    return [f"SME note ({note}): {detail}"]
  lines = []
  if note:
    lines.append(f"Followed the SME note: {note}")
  if summary:
    lines.append(summary)
  return lines


def _spread_finding(quality: dict[str, Any]) -> str | None:
  rows = quality.get("spread") or []
  if not rows:
    return None
  bits = []
  for item in rows:
    mode = item.get("mode")
    mode_text = "n/a" if mode is None else mode
    bits.append(
        f"`{item['name']}` mean {item.get('mean')}, median {item.get('median')}, "
        f"mode {mode_text}, std {item.get('std')}"
    )
  return "Distributions — " + "; ".join(bits) + "."


def _metric(value: Any) -> Any:
  if value is None:
    return None
  try:
    return round(float(value), 3)
  except (TypeError, ValueError):
    return value
