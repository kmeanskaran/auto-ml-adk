"""Production harness around every tool call.

The harness does not know what feature engineering means. It checkpoints,
blocks, traces, and counts calls for whatever tool the agent requested.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from typing import Optional

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool

from team.config import load_config
from team.store import ExperimentStore
from team.store import jsonable


class HarnessPlugin(BasePlugin):
  """Checkpoint, policy, and trace plugin installed on the ADK app."""

  def __init__(self, store: ExperimentStore):
    super().__init__(name="ml_harness")
    self.store = store
    self._started: dict[str, float] = {}

  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: Any,
  ) -> Optional[dict[str, Any]]:
    experiment_id = tool_context.state.get("experiment_id")
    if not experiment_id:
      return None
    record = self.store.load(str(experiment_id))
    blocked = set(record.get("blocked_tools") or [])
    self._checkpoint(record, tool.name)
    call_id = str(getattr(tool_context, "function_call_id", None) or tool.name)
    self._started[call_id] = time.perf_counter()
    self._trace(
        record,
        {
            "event": "tool_requested",
            "tool": tool.name,
            "agent": getattr(tool_context, "agent_name", None),
            "identity": load_config().runtime.identity,
            "args": _safe_args(tool_args),
        },
    )
    if tool.name in blocked:
      self._trace(
          record,
          {
              "event": "tool_blocked",
              "tool": tool.name,
              "identity": load_config().runtime.identity,
          },
      )
      return {
          "error": f"Tool {tool.name} is blocked by policy.",
          "tool": tool.name,
      }
    calls = int(record.get("tool_calls") or 0) + 1

    def mutate(current: dict[str, Any]) -> None:
      current["tool_calls"] = calls

    self.store.update(str(experiment_id), mutate)
    return None

  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: Any,
      result: dict[str, Any],
  ) -> Optional[dict[str, Any]]:
    experiment_id = tool_context.state.get("experiment_id")
    if not experiment_id:
      return None
    call_id = str(getattr(tool_context, "function_call_id", None) or tool.name)
    started = self._started.pop(call_id, None)
    duration_ms = None
    if started is not None:
      duration_ms = round((time.perf_counter() - started) * 1000, 2)
    record = self.store.load(str(experiment_id))
    status = "error" if isinstance(result, dict) and result.get("error") else "ok"
    self._trace(
        record,
        {
            "event": "tool_finished",
            "tool": tool.name,
            "status": status,
            "duration_ms": duration_ms,
            "identity": load_config().runtime.identity,
            "cached": bool(isinstance(result, dict) and result.get("cached")),
        },
    )
    return None

  def _checkpoint(self, record: dict[str, Any], tool_name: str) -> None:
    directory = self.store.experiment_dir(record["id"]) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "tool": tool_name,
        "status": record.get("status"),
        "strategy_level": record.get("strategy_level"),
        "phases": sorted((record.get("phases") or {}).keys()),
        "trajectory": record.get("trajectory") or [],
        "training_approved": record.get("training_approved"),
    }
    path = directory / f"{time.time_ns()}-{tool_name}.json"
    path.write_text(json.dumps(jsonable(snapshot), indent=2), encoding="utf-8")

  def _trace(self, record: dict[str, Any], event: dict[str, Any]) -> None:
    path = self.store.experiment_dir(record["id"]) / "trace.jsonl"
    event = dict(event)
    event["experiment_id"] = record["id"]
    with path.open("a", encoding="utf-8") as handle:
      handle.write(json.dumps(jsonable(event)) + "\n")


def _safe_args(tool_args: dict[str, Any]) -> dict[str, Any]:
  safe = {}
  for key, value in tool_args.items():
    if key == "tool_context":
      continue
    text = str(value)
    safe[key] = text if len(text) <= 300 else text[:300] + "..."
  return safe
