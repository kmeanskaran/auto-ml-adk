"""ADK tools owned by the data engineer."""

from __future__ import annotations

from typing import Any

import pandas as pd
from google.adk.tools.tool_context import ToolContext

from team.artifacts import ArtifactSaver
from team.data_engineer.clean import clean_frame
from team.data_engineer.metrics import choose_metrics
from team.data_engineer.sandbox import execute_snippet
from team.product_manager.spec import infer_target
from team.phase import dataset_fingerprint
from team.phase import experiment_dir
from team.phase import load_table
from team.phase import run_phase


def clean_dataset(tool_context: ToolContext) -> dict[str, Any]:
  """Drop duplicate ids, drop unlabeled rows, and fill remaining gaps."""

  def compute(
      frame: pd.DataFrame, record: dict[str, Any]
  ) -> tuple[dict[str, Any], dict[str, Any]]:
    objective = dict(record.get("objective") or {})
    target = str(objective.get("target_column") or "")
    if target not in frame.columns:
      target = infer_target(frame)
    cleaned, report = clean_frame(frame, target)
    path = ArtifactSaver(experiment_dir(tool_context)).save_clean_table(cleaned)
    report["path"] = str(path)
    choice = choose_metrics(cleaned, target, str(objective.get("text") or ""))
    report["task_type"] = choice["task_type"]
    report["metric"] = choice["metric"]
    report["metrics"] = choice["metrics"]
    report["metric_reason"] = choice["reason"]
    objective["target_column"] = target
    objective["task_type"] = choice["task_type"]
    objective["metrics"] = list(choice["metrics"])
    gate = str(objective.get("metric") or "")
    if gate not in choice["metrics"]:
      objective["metric"] = choice["metric"]
    return report, {"objective": objective}

  return run_phase(
      tool_context,
      "clean_dataset",
      fingerprint=dataset_fingerprint(tool_context),
      compute=compute,
  )


def run_analysis_snippet(code: str, tool_context: ToolContext) -> dict[str, Any]:
  """Run a short pandas snippet. Policy blocks this unless the human enables it.

  The snippet must assign result. It cannot import, open files, or use dunder names.
  """
  frame = load_table(tool_context, "clean")
  return execute_snippet(frame, code)
