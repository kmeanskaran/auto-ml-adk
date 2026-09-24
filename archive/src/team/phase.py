"""Shared ADK tool plumbing: cache a phase, load a table, talk to the store."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Callable

from google.adk.tools.tool_context import ToolContext

from team.store import ExperimentStore
from team.tables import read_table


def run_phase(
    tool_context: ToolContext,
    name: str,
    fingerprint: Any,
    compute: Callable[..., Any],
    table: str = "raw",
) -> dict[str, Any]:
  store, experiment_id = store_and_id(tool_context)
  preview = store.load(experiment_id)
  token = fingerprint(preview) if callable(fingerprint) else fingerprint
  _consume_fault_or_raise(store, experiment_id, name)
  cached = _cached(store.load(experiment_id), name, token)
  if cached is not None:
    _append_trajectory(store, experiment_id, name)
    cached = dict(cached)
    cached["cached"] = True
    mirror_state(tool_context, name, cached)
    return cached

  record = store.load(experiment_id)
  produced = compute(load_table(tool_context, table), record)
  updates: dict[str, Any] = {}
  if isinstance(produced, tuple):
    result, updates = produced
  else:
    result = produced

  def mutate(current: dict[str, Any]) -> None:
    for key, value in updates.items():
      current[key] = value
    phases = dict(current.get("phases") or {})
    phases[name] = {"status": "ok", "fingerprint": token, "result": result}
    current["phases"] = phases
    current["trajectory"] = list(current.get("trajectory") or []) + [name]

  store.update(experiment_id, mutate)
  mirror_state(tool_context, name, result)
  return result


def dataset_fingerprint(tool_context: ToolContext) -> str:
  record = load_record(tool_context)
  raw = Path(record["raw_path"])
  stat = raw.stat()
  return f"{raw}:{stat.st_size}:{stat.st_mtime_ns}"


def load_table(tool_context: ToolContext, kind: str) -> Any:
  record = load_record(tool_context)
  if kind == "clean":
    path = (
        (record.get("phases") or {}).get("clean_dataset", {}).get("result") or {}
    ).get("path")
    if path and Path(path).exists():
      return read_table(path)
  return read_table(record["raw_path"])


def experiment_dir(tool_context: ToolContext) -> Path:
  return store(tool_context).experiment_dir(experiment_id(tool_context))


def load_record(tool_context: ToolContext) -> dict[str, Any]:
  handle, experiment = store_and_id(tool_context)
  return handle.load(experiment)


def store_and_id(tool_context: ToolContext) -> tuple[ExperimentStore, str]:
  return store(tool_context), experiment_id(tool_context)


def store(tool_context: ToolContext) -> ExperimentStore:
  root = tool_context.state.get("store_root")
  if not root:
    raise RuntimeError("Session state is missing store_root.")
  return ExperimentStore(Path(str(root)))


def experiment_id(tool_context: ToolContext) -> str:
  value = tool_context.state.get("experiment_id")
  if not value:
    raise RuntimeError("Session state is missing experiment_id.")
  return str(value)


def as_bool(value: Any, default: bool) -> bool:
  if value is None:
    return default
  if isinstance(value, bool):
    return value
  return str(value).lower() in {"1", "true", "yes"}


def latest_sme_note(record: dict[str, Any], stage: str | None = None) -> str:
  """Latest rework prompt, optionally limited to one lifecycle stage."""
  notes = list(record.get("sme_notes") or [])
  if stage:
    notes = [
        item
        for item in notes
        if isinstance(item, dict) and item.get("stage") == stage
    ]
  if not notes:
    return ""
  last = notes[-1]
  if isinstance(last, dict):
    return str(last.get("prompt") or "").strip()
  return str(last).strip()


def mirror_state(tool_context: ToolContext, name: str, result: dict[str, Any]) -> None:
  if name == "profile_quality":
    tool_context.state["quality_summary"] = (
        f"missing_cells={result.get('missing_cells')} "
        f"duplicate_ids={result.get('duplicate_id_rows')} "
        f"categorical_predictors={result.get('categorical_predictors')} "
        f"outlier_cells={result.get('outlier_cells')}"
    )
  elif name == "analyze_target":
    tool_context.state["target_summary"] = str(result.get("classes"))
  elif name == "evaluate_model":
    tool_context.state["target_met"] = "true" if result.get("target_met") else "false"
    if "f1" in result:
      tool_context.state["best_f1"] = f"{float(result['f1']):.4f}"


def _consume_fault_or_raise(
    handle: ExperimentStore, experiment: str, name: str
) -> None:
  should_fail = False

  def mutate(record: dict[str, Any]) -> None:
    nonlocal should_fail
    fault = dict(record.get("fault") or {})
    if fault.get("tool") != name or int(fault.get("remaining") or 0) <= 0:
      return
    fault["remaining"] = int(fault["remaining"]) - 1
    record["fault"] = fault
    record["status"] = "interrupted"
    record["error"] = f"Execution interrupted during {name}"
    record["interrupted_tool"] = name
    should_fail = True

  handle.update(experiment, mutate)
  if should_fail:
    raise RuntimeError(f"Execution interrupted during {name}")


def _cached(
    record: dict[str, Any], name: str, fingerprint: str
) -> dict[str, Any] | None:
  phase = (record.get("phases") or {}).get(name) or {}
  if phase.get("status") == "ok" and phase.get("fingerprint") == fingerprint:
    result = phase.get("result")
    return result if isinstance(result, dict) else None
  return None


def _append_trajectory(handle: ExperimentStore, experiment: str, name: str) -> None:
  def mutate(record: dict[str, Any]) -> None:
    record["trajectory"] = list(record.get("trajectory") or []) + [name]

  handle.update(experiment, mutate)
