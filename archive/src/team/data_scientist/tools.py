"""ADK tools owned by the data scientist."""

from __future__ import annotations

from typing import Any

import pandas as pd
from google.adk.tools.tool_context import ToolContext

from team.artifacts import ArtifactSaver
from team.data_scientist.features import feature_spec
from team.data_scientist.selection import recommend_two
from team.phase import experiment_dir
from team.phase import run_phase


def create_features(tool_context: ToolContext) -> dict[str, Any]:
  """Write the feature spec for the current strategy level."""

  def compute(frame: pd.DataFrame, record: dict[str, Any]) -> dict[str, Any]:
    level = int(record.get("strategy_level") or 0)
    target = record["objective"]["target_column"]
    spec = feature_spec(frame, level, target)
    path = ArtifactSaver(experiment_dir(tool_context)).save_feature_pipeline(spec)
    spec["path"] = str(path)
    return spec

  return run_phase(
      tool_context,
      "create_features",
      fingerprint=lambda record: f"strategy:{int(record.get('strategy_level') or 0)}",
      compute=compute,
      table="clean",
  )


def select_models(tool_context: ToolContext) -> dict[str, Any]:
  """Recommend two model families for this table. The SME picks one to train."""

  def compute(frame: pd.DataFrame, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    target = record["objective"]["target_column"]
    decision = recommend_two(frame, target)
    stages = dict(record.get("stages") or {})
    stages["data_scientist"] = "done"
    tool_context.state["model_selection"] = decision["decision"]
    updates: dict[str, Any] = {"stages": stages}
    if record.get("mode") != "staged":
      updates["chosen_models"] = [item["id"] for item in decision["options"][:1]]
    return decision, updates

  return run_phase(
      tool_context,
      "select_models",
      fingerprint="recommend-two-v1",
      compute=compute,
      table="clean",
  )
