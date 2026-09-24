"""Infer a target column and write the technical specification."""

from __future__ import annotations

from typing import Any

import pandas as pd

from team.data_engineer.metrics import choose_metrics


def infer_target(frame: pd.DataFrame) -> str:
  lookup = {str(column).lower(): column for column in frame.columns}
  for name in ("target", "label", "diagnosis", "churn", "class"):
    if name in lookup:
      return str(lookup[name])
  chosen = None
  chosen_unique = 10**9
  for column in frame.columns:
    if column == "id" or str(column).endswith("_id"):
      continue
    unique = int(frame[column].nunique(dropna=True))
    if 1 < unique <= 12 and unique < chosen_unique:
      chosen = column
      chosen_unique = unique
  if chosen is None:
    return str(frame.columns[-1])
  return str(chosen)


def technical_spec(
    frame: pd.DataFrame, record: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
  objective = dict(record.get("objective") or {})
  target = str(objective.get("target_column") or "")
  if target not in frame.columns:
    target = infer_target(frame)
    objective["target_column"] = target
  choice = choose_metrics(frame, target, str(objective.get("text") or ""))
  if choice["task_type"] == "regression":
    problem = f"Predict {target} from the other columns."
  else:
    problem = f"Train a classifier that predicts {target} from the other columns."
  gate = str(objective.get("metric") or "")
  if gate in choice["metrics"] and objective.get("target") is not None:
    problem += f" Optimize {gate} to at least {objective.get('target')}."
  spec = {
      "business_problem": objective.get("text") or "",
      "technical_problem": problem,
      "task_type": choice["task_type"],
      "target_column": target,
      "metric": gate if gate in choice["metrics"] else "",
      "target_value": objective.get("target"),
      "column_count": int(frame.shape[1]),
  }
  stages = dict(record.get("stages") or {})
  stages["product_manager"] = "done"
  return spec, {"objective": objective, "stages": stages}
