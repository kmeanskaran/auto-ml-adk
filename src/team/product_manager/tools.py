"""ADK tools owned by the product manager."""

from __future__ import annotations

from typing import Any

import pandas as pd
from google.adk.tools.tool_context import ToolContext

from team.phase import dataset_fingerprint
from team.phase import run_phase
from team.product_manager.spec import technical_spec


def frame_technical_spec(tool_context: ToolContext) -> dict[str, Any]:
  """Turn the business problem into a classification task and a target column."""

  def compute(frame: pd.DataFrame, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    spec, updates = technical_spec(frame, record)
    tool_context.state["tech_spec"] = spec["technical_problem"]
    return spec, updates

  return run_phase(
      tool_context,
      "frame_technical_spec",
      fingerprint=dataset_fingerprint(tool_context) + ":spec",
      compute=compute,
  )
