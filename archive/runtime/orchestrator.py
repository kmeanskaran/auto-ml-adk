"""Drive the ADK workflow until it finishes, pauses for a human, or crashes."""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.runners import Runner
from google.genai import types

from team.config import detect_model_name
from team.config import load_config
from team.data_engineer.catalog import write_sample
from team.data_engineer.sample import write_churn_csv
from team.store import ExperimentStore
from runtime.agents import build_orchestrator
from runtime.agents import build_stage_agent
from runtime.agents import resolve_model
from runtime.harness import HarnessPlugin
from runtime.stages import STAGE_ORDER
from runtime.stages import STAGES
from runtime.stages import board as render_board
from runtime.stages import next_stage_id


class ExperimentError(RuntimeError):
  """A caller-facing error with an HTTP status the API can reuse."""

  def __init__(self, message: str, status_code: int = 400):
    super().__init__(message)
    self.status_code = status_code


_DEFAULTS = load_config()


@dataclass
class ExperimentRequest:
  """What the human hands the runtime to start an experiment."""

  objective: str = _DEFAULTS.experiment.objective
  target_column: str = _DEFAULTS.experiment.target_column
  metric: str = _DEFAULTS.experiment.metric
  metrics: list[str] = field(default_factory=list)
  target_value: float = _DEFAULTS.experiment.target_value
  dataset_path: str | None = None
  approve_training: bool = _DEFAULTS.runtime.approve_training
  fault_tool: str | None = None
  blocked_tools: list[str] | None = None
  user_id: str = _DEFAULTS.runtime.user
  frame: pd.DataFrame | None = None
  notes: list[str] = field(default_factory=list)
  staged: bool = False
  checkpoints: list[str] | None = None
  sample_id: str | None = None


class MlRuntime:
  """One ADK app, one store, and the methods the API calls."""

  def __init__(
      self,
      root: Path,
      model_name: str | None = None,
      session_service: Any | None = None,
  ):
    self.root = Path(root)
    self.store = ExperimentStore(self.root)
    self.model_name = model_name or detect_model_name()
    self.model = resolve_model(self.model_name)
    self.app_name = load_config().session.app_name
    self.session_service = session_service or _build_session_service(self.root)
    self._stage_runners: dict[str, Runner] = {}
    application = App(
        name=self.app_name,
        root_agent=build_orchestrator(self.model),
        plugins=[HarnessPlugin(self.store)],
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    self.runner = Runner(
        app=application,
        session_service=self.session_service,
        auto_create_session=False,
    )

  async def start(self, request: ExperimentRequest) -> dict[str, Any]:
    experiment_id = uuid.uuid4().hex[:12]
    directory = self.store.experiment_dir(experiment_id)
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "raw.csv"
    sample_id = request.sample_id
    if request.frame is not None:
      request.frame.to_csv(raw_path, index=False)
    elif request.dataset_path:
      shutil.copyfile(request.dataset_path, raw_path)
    elif sample_id:
      try:
        write_sample(sample_id, raw_path)
      except KeyError as exc:
        raise ExperimentError(str(exc), status_code=400) from exc
    else:
      write_churn_csv(raw_path)
    cfg = load_config()
    blocked = (
        list(request.blocked_tools)
        if request.blocked_tools is not None
        else list(cfg.runtime.blocked_tools)
    )
    record = {
        "id": experiment_id,
        "status": "running",
        "identity": cfg.runtime.identity,
        "model_name": self.model_name,
        "user_id": request.user_id,
        "session_id": experiment_id,
        "objective": {
            "text": request.objective,
            "target_column": request.target_column,
            "metric": request.metric,
            "metrics": list(request.metrics or [request.metric]),
            "target": request.target_value,
        },
        "raw_path": str(raw_path),
        "strategy_level": 0,
        "approve_training": request.approve_training,
        "training_approved": not request.approve_training,
        "fault": (
            {"tool": request.fault_tool, "remaining": 1}
            if request.fault_tool
            else None
        ),
        "blocked_tools": blocked,
        "phases": {},
        "history": [],
        "plans": [],
        "notes": list(request.notes),
        "trajectory": [],
        "tool_calls": 0,
        "target_met": False,
        "best": None,
        "pending_approval": None,
        "error": None,
        "mode": "staged" if request.staged else "full",
        "current_stage": STAGE_ORDER[0],
        "checkpoints": _checkpoints(request.checkpoints) if request.staged else [],
        "sme_notes": [],
        "sample_id": sample_id,
        "chosen_models": [],
    }
    self.store.create(record)
    await self.session_service.create_session(
        app_name=self.app_name,
        user_id=request.user_id,
        session_id=experiment_id,
        state=_session_state(self.root, record),
    )
    brief_parts = [request.objective.strip()]
    if request.target_column:
      brief_parts.append(f"Target column {request.target_column}.")
    if request.metric:
      brief_parts.append(f"Reach {request.metric} {request.target_value}.")
    brief = " ".join(part for part in brief_parts if part)
    if request.staged:
      return await self._run_from_stage(experiment_id, STAGE_ORDER[0], brief)
    return await self._run_text(experiment_id, brief)

  async def approve(
      self,
      experiment_id: str,
      confirmed: bool,
      allow_retries: bool = True,
  ) -> dict[str, Any]:
    record = self.store.load(experiment_id)
    pending = record.get("pending_approval")
    if record.get("status") != "awaiting_approval" or not pending:
      raise ExperimentError(
          "This experiment is not waiting for approval.",
          status_code=409,
      )
    if not confirmed:
      self.store.update(experiment_id, lambda current: _reject(current))
      if record.get("mode") == "staged":
        return self.board(experiment_id)
      return self.store.load(experiment_id)

    self.store.update(experiment_id, lambda current: _grant(current))
    runner = self._staged_runner(record)
    try:
      confirmation = await self._consume(
          user_id=record["user_id"],
          session_id=record["session_id"],
          invocation_id=pending.get("invocation_id"),
          new_message=_confirmation_content(pending, True, allow_retries),
          state_delta={"training_approved": True},
          runner=runner,
      )
    except Exception:
      if record.get("mode") == "staged":
        return await self._run_from_stage(
            experiment_id,
            str(record.get("current_stage") or "train"),
            "Training is approved. Continue from the last checkpoint.",
            state_delta={"training_approved": True},
        )
      return await self._run_text(
          experiment_id,
          "Training is approved. Continue from the last checkpoint.",
      )
    if record.get("mode") == "staged":
      if confirmation:
        self.store.update(
            experiment_id,
            lambda current: _mark_waiting(current, confirmation),
        )
        return self.board(experiment_id)
      return await self._finish_stage(
          experiment_id, str(record.get("current_stage") or "train")
      )
    return self._finish(experiment_id, confirmation)

  async def resume(self, experiment_id: str) -> dict[str, Any]:
    record = self.store.load(experiment_id)
    if record.get("status") not in {"interrupted", "failed"}:
      raise ExperimentError(
          f"Cannot resume an experiment in status {record.get('status')}.",
          status_code=409,
      )
    self.store.update(experiment_id, lambda current: _mark_running(current))
    if record.get("mode") == "staged":
      return await self._run_from_stage(
          experiment_id,
          str(record.get("current_stage") or STAGE_ORDER[0]),
          "Resume from the last checkpoint. Completed phases are already stored.",
      )
    return await self._run_text(
        experiment_id,
        "Resume from the last checkpoint. Completed phases are already stored.",
    )

  async def iterate(
      self,
      experiment_id: str,
      note: str,
      target_value: float | None = None,
  ) -> dict[str, Any]:
    record = self.store.load(experiment_id)
    if record.get("status") not in {"completed", "rejected"}:
      raise ExperimentError(
          "Iterate after the current run has finished.",
          status_code=409,
      )

    def mutate(current: dict[str, Any]) -> None:
      notes = list(current.get("notes") or [])
      notes.append(note)
      current["notes"] = notes
      if target_value is not None:
        current["objective"]["target"] = float(target_value)
      current["strategy_level"] = int(current.get("strategy_level") or 0) + 1
      phases = dict(current.get("phases") or {})
      for key in (
          "create_features",
          "run_training",
          "evaluate_model",
          "write_solution_design",
      ):
        phases.pop(key, None)
      current["phases"] = phases
      current["target_met"] = False
      if current.get("approve_training"):
        current["training_approved"] = False
      current["status"] = "running"
      current["error"] = None
      current["pending_approval"] = None

    updated = self.store.update(experiment_id, mutate)
    return await self._run_text(
        experiment_id,
        f"Iterate on the solution. Human note: {note}",
        state_delta={
            "training_approved": bool(updated.get("training_approved")),
            "strategy_level": str(updated.get("strategy_level")),
            "last_plan": note,
        },
    )

  def board(self, experiment_id: str) -> dict[str, Any]:
    return render_board(self.store.load(experiment_id))

  async def next_stage(self, experiment_id: str) -> dict[str, Any]:
    record = self.store.load(experiment_id)
    if record.get("mode") != "staged":
      raise ExperimentError("Next is only for staged board runs.", status_code=409)
    if record.get("status") != "awaiting_review":
      raise ExperimentError(
          "Advance after the current stage is ready for review.",
          status_code=409,
      )
    if record.get("current_stage") == "model" and not record.get("chosen_models"):
      raise ExperimentError(
          "Pick one of the two recommended models before continuing.",
          status_code=409,
      )
    nxt = next_stage_id(record.get("current_stage"))
    if nxt is None:
      self.store.update(experiment_id, _mark_completed)
      return self.board(experiment_id)
    return await self._run_from_stage(
        experiment_id,
        nxt,
        "Continue to the next lifecycle stage. Completed stages are stored.",
        state_delta={"last_plan": ""},
    )

  async def rework_stage(self, experiment_id: str, prompt: str) -> dict[str, Any]:
    record = self.store.load(experiment_id)
    if record.get("mode") != "staged":
      raise ExperimentError("Rework is only for staged board runs.", status_code=409)
    if record.get("status") not in {"awaiting_review", "awaiting_approval"}:
      raise ExperimentError(
          "Rework the stage that is waiting for the SME.",
          status_code=409,
      )
    if record.get("status") == "awaiting_approval":
      raise ExperimentError(
          "Approve or reject training before reworking this stage.",
          status_code=409,
      )
    stage_id = str(record.get("current_stage") or STAGE_ORDER[0])
    note = prompt.strip()
    if not note:
      raise ExperimentError("A rework prompt is required.", status_code=400)

    def mutate(current: dict[str, Any]) -> None:
      notes = list(current.get("sme_notes") or [])
      notes.append({"stage": stage_id, "prompt": note})
      current["sme_notes"] = notes
      current["notes"] = list(current.get("notes") or []) + [note]
      phases = dict(current.get("phases") or {})
      for key in STAGES[stage_id]["phases"]:
        phases.pop(key, None)
      current["phases"] = phases
      if stage_id == "model":
        current["chosen_models"] = []
      current["status"] = "running"
      current["error"] = None
      current["pending_approval"] = None

    self.store.update(experiment_id, mutate)
    return await self._run_from_stage(
        experiment_id,
        stage_id,
        f"Rework this stage with the SME note: {note}",
        state_delta={"last_plan": note},
    )

  def choose_model(self, experiment_id: str, model: str) -> dict[str, Any]:
    record = self.store.load(experiment_id)
    if record.get("mode") != "staged":
      raise ExperimentError("Model choice is only for staged board runs.", status_code=409)
    if record.get("status") != "awaiting_review" or record.get("current_stage") != "model":
      raise ExperimentError(
          "Choose a model while the features & models stage is in review.",
          status_code=409,
      )
    name = str(model or "").strip()
    options = (
        (record.get("phases") or {}).get("select_models", {}).get("result") or {}
    ).get("options") or []
    allowed = {item.get("id") for item in options if item.get("id")}
    if name not in allowed:
      raise ExperimentError(
          f"Pick one of: {', '.join(sorted(str(item) for item in allowed))}.",
          status_code=400,
      )

    def mutate(current: dict[str, Any]) -> None:
      current["chosen_models"] = [name]

    self.store.update(experiment_id, mutate)
    return self.board(experiment_id)

  def get(self, experiment_id: str) -> dict[str, Any]:
    return self.store.load(experiment_id)

  def list_experiments(self) -> list[dict[str, Any]]:
    summaries = []
    for experiment_id in self.store.list_ids():
      record = self.store.load(experiment_id)
      best = record.get("best") or {}
      summaries.append(
          {
              "id": experiment_id,
              "status": record.get("status"),
              "mode": record.get("mode") or "full",
              "current_stage": record.get("current_stage"),
              "objective": (record.get("objective") or {}).get("text"),
              "target_met": record.get("target_met"),
              "model": best.get("model"),
              "f1": (best.get("metrics") or {}).get("f1"),
          }
      )
    return summaries

  def clear_experiments(self) -> dict[str, Any]:
    """Remove every saved run, including ones awaiting review."""
    removed = self.store.clear()
    uploads = self.root / "uploads"
    if uploads.exists():
      shutil.rmtree(uploads)
    memory = self.root / "memory.jsonl"
    if memory.exists():
      memory.unlink()
    return {"removed": removed}

  def predict(self, experiment_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    record = self.store.load(experiment_id)
    best = record.get("best") or {}
    path = best.get("path")
    if not path or not Path(path).exists():
      raise ExperimentError(
          "This experiment has no model artifact yet.",
          status_code=409,
      )
    model = joblib.load(path)
    return {
        "model": best.get("model"),
        "metrics": best.get("metrics"),
        "predictions": model.predict_records(rows),
    }

  def design_text(self, experiment_id: str) -> str:
    record = self.store.load(experiment_id)
    path = record.get("design_path")
    if not path or not Path(path).exists():
      raise ExperimentError(
          "The solution design has not been written yet.",
          status_code=409,
      )
    return Path(path).read_text(encoding="utf-8")

  async def _run_text(
      self,
      experiment_id: str,
      text: str,
      state_delta: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    record = self.store.load(experiment_id)
    message = types.Content(role="user", parts=[types.Part(text=text)])
    try:
      confirmation = await self._consume(
          user_id=record["user_id"],
          session_id=record["session_id"],
          new_message=message,
          state_delta=state_delta,
      )
    except Exception as exc:
      return self._capture_error(experiment_id, exc)
    return self._finish(experiment_id, confirmation)

  async def _run_from_stage(
      self,
      experiment_id: str,
      stage_id: str,
      text: str,
      state_delta: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    def mark(current: dict[str, Any]) -> None:
      current["current_stage"] = stage_id
      current["status"] = "running"
      current["error"] = None

    self.store.update(experiment_id, mark)
    record = self.store.load(experiment_id)
    message = types.Content(role="user", parts=[types.Part(text=text)])
    try:
      confirmation = await self._consume(
          user_id=record["user_id"],
          session_id=record["session_id"],
          new_message=message,
          state_delta=state_delta,
          runner=self._runner_for(stage_id),
      )
    except Exception as exc:
      self._capture_error(experiment_id, exc)
      return self.board(experiment_id)
    if confirmation:
      self.store.update(
          experiment_id,
          lambda current: _mark_waiting(current, confirmation),
      )
      return self.board(experiment_id)
    return await self._finish_stage(experiment_id, stage_id)

  async def _finish_stage(self, experiment_id: str, stage_id: str) -> dict[str, Any]:
    record = self.store.load(experiment_id)
    if record.get("status") in {"interrupted", "failed", "rejected"}:
      return self.board(experiment_id)
    checkpoints = set(record.get("checkpoints") or [])
    if stage_id == "model" and stage_id not in checkpoints:
      self.store.update(experiment_id, _auto_pick_model)
    nxt = next_stage_id(stage_id)
    if stage_id in checkpoints:
      self.store.update(
          experiment_id,
          lambda current: _mark_review(current, stage_id),
      )
      return self.board(experiment_id)
    if nxt is None:
      self.store.update(experiment_id, _mark_completed)
      return self.board(experiment_id)
    return await self._run_from_stage(
        experiment_id,
        nxt,
        "Checkpoint skipped. Continue to the next lifecycle stage.",
    )

  def _runner_for(self, stage_id: str) -> Runner:
    cached = self._stage_runners.get(stage_id)
    if cached is not None:
      return cached
    application = App(
        name=self.app_name,
        root_agent=build_stage_agent(self.model, stage_id),
        plugins=[HarnessPlugin(self.store)],
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    runner = Runner(
        app=application,
        session_service=self.session_service,
        auto_create_session=False,
    )
    self._stage_runners[stage_id] = runner
    return runner

  def _staged_runner(self, record: dict[str, Any]) -> Runner | None:
    if record.get("mode") != "staged":
      return None
    return self._runner_for(str(record.get("current_stage") or STAGE_ORDER[0]))

  async def _consume(
      self,
      *,
      user_id: str,
      session_id: str,
      new_message: types.Content | None = None,
      invocation_id: str | None = None,
      state_delta: dict[str, Any] | None = None,
      runner: Runner | None = None,
  ) -> dict[str, Any] | None:
    confirmation = None
    engine = runner or self.runner
    async for event in engine.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message,
        invocation_id=invocation_id,
        state_delta=state_delta,
    ):
      found = _confirmation_from_event(event)
      if found:
        confirmation = found
    return confirmation

  def _finish(
      self,
      experiment_id: str,
      confirmation: dict[str, Any] | None,
  ) -> dict[str, Any]:
    if confirmation:
      self.store.update(
          experiment_id,
          lambda current: _mark_waiting(current, confirmation),
      )
      record = self.store.load(experiment_id)
      if record.get("mode") == "staged":
        return self.board(experiment_id)
      return record
    record = self.store.load(experiment_id)
    if record.get("status") in {"interrupted", "completed", "rejected"}:
      return record
    if (record.get("phases") or {}).get("write_solution_design") or record.get("best"):
      self.store.update(experiment_id, _mark_completed)
    else:
      self.store.update(
          experiment_id,
          lambda current: _set_error(
              current, "The workflow stopped before a model was chosen."
          ),
      )
    return self.store.load(experiment_id)

  def _capture_error(self, experiment_id: str, exc: Exception) -> dict[str, Any]:
    text = f"{type(exc).__name__}: {exc}"

    def mutate(current: dict[str, Any]) -> None:
      if current.get("status") != "interrupted":
        current["status"] = (
            "interrupted" if "interrupted" in text.lower() else "failed"
        )
      current["error"] = text

    return self.store.update(experiment_id, mutate)


def _build_session_service(root: Path) -> Any:
  backend = load_config().session.backend
  override = os.environ.get("ML_SESSION_BACKEND")
  if override:
    backend = override.strip()
  if backend == "in_memory":
    from google.adk.sessions.in_memory_session_service import InMemorySessionService

    return InMemorySessionService()
  if backend == "sqlite":
    from google.adk.sessions.sqlite_session_service import SqliteSessionService

    path = root / (load_config().session.db_path or "sessions.db")
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSessionService(str(path))
  raise ValueError(
      f"session.backend {backend!r} is not supported. Use in_memory or sqlite."
  )


def _checkpoints(raw: list[str] | None) -> list[str]:
  if not raw:
    return list(STAGE_ORDER)
  allowed = set(STAGE_ORDER)
  picked = [item for item in raw if item in allowed]
  return picked or list(STAGE_ORDER)


def _auto_pick_model(current: dict[str, Any]) -> None:
  if current.get("chosen_models"):
    return
  options = (
      (current.get("phases") or {}).get("select_models", {}).get("result") or {}
  ).get("options") or []
  if options and options[0].get("id"):
    current["chosen_models"] = [options[0]["id"]]


def _mark_review(current: dict[str, Any], stage_id: str) -> None:
  current["status"] = "awaiting_review"
  current["current_stage"] = stage_id
  current["pending_approval"] = None
  current["error"] = None


def _session_state(root: Path, record: dict[str, Any]) -> dict[str, Any]:
  return {
      "experiment_id": record["id"],
      "store_root": str(root),
      "approve_training": record["approve_training"],
      "training_approved": record["training_approved"],
      "strategy_level": "0",
      "target_met": "",
      "quality_summary": "",
      "target_summary": "",
      "best_f1": "",
      "best_model": "",
      "design_path": "",
      "last_plan": "",
  }


def _confirmation_content(
    pending: dict[str, Any],
    confirmed: bool,
    allow_retries: bool,
) -> types.Content:
  return types.Content(
      role="user",
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  id=pending["id"],
                  name="adk_request_confirmation",
                  response={
                      "confirmed": confirmed,
                      "hint": pending.get("hint") or "",
                      "payload": {"allow_retries": allow_retries},
                  },
              )
          )
      ],
  )


def _confirmation_from_event(event: Any) -> dict[str, Any] | None:
  if not hasattr(event, "get_function_calls"):
    return None
  for call in event.get_function_calls():
    if call.name != "adk_request_confirmation":
      continue
    args = dict(call.args or {})
    original = args.get("originalFunctionCall") or {}
    hint_source = args.get("toolConfirmation") or {}
    return {
        "id": call.id,
        "invocation_id": event.invocation_id,
        "hint": hint_source.get("hint") or "",
        "tool": original.get("name"),
    }
  return None


def _mark_waiting(current: dict[str, Any], confirmation: dict[str, Any]) -> None:
  current["status"] = "awaiting_approval"
  current["pending_approval"] = confirmation


def _grant(current: dict[str, Any]) -> None:
  current["training_approved"] = True
  current["status"] = "running"
  current["pending_approval"] = None
  current["error"] = None


def _reject(current: dict[str, Any]) -> None:
  current["status"] = "rejected"
  current["pending_approval"] = None
  current["error"] = "Training was rejected."


def _mark_running(current: dict[str, Any]) -> None:
  current["status"] = "running"
  current["error"] = None


def _mark_completed(current: dict[str, Any]) -> None:
  current["status"] = "completed"


def _set_error(current: dict[str, Any], message: str) -> None:
  current["status"] = "failed"
  current["error"] = message
