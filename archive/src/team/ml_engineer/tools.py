"""ADK tools owned by the ML engineer."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from google.adk.tools.tool_context import ToolContext

from team.artifacts import ArtifactSaver
from team.ml_engineer.design import render_design
from team.ml_engineer.role import PLANS
from team.ml_engineer.train import train_candidates
from team.phase import as_bool
from team.phase import experiment_dir
from team.phase import load_record
from team.phase import run_phase
from team.phase import store_and_id
from team.store import ExperimentStore


def training_requires_approval(tool_context: ToolContext) -> bool:
  """Ask a human the first time training is about to spend a fit."""
  if not as_bool(tool_context.state.get("approve_training"), default=True):
    return False
  record = load_record(tool_context)
  return not bool(record.get("training_approved"))


def run_training(tool_context: ToolContext) -> dict[str, Any]:
  """Train the current strategy's candidates in parallel and keep the best F1."""

  def compute(
      frame: pd.DataFrame, record: dict[str, Any]
  ) -> tuple[dict[str, Any], dict[str, Any]]:
    level = int(record.get("strategy_level") or 0)
    objective = record["objective"]
    if objective.get("task_type") == "regression":
      raise RuntimeError(
          "This table is a regression problem. The scores are MAE, RMSE, and R²."
      )
    target = objective["target_column"]
    chosen = list(record.get("chosen_models") or [])
    if not chosen:
      raise RuntimeError("Pick one of the recommended models before training.")
    model, report = train_candidates(frame, target, level, models=chosen)
    run_id = int(record.get("training_run_seq") or 0) + 1
    saver = ArtifactSaver(experiment_dir(tool_context))
    artifact = saver.save_model(model, f"model_iter{run_id}.joblib")
    report["run_id"] = run_id
    report["path"] = str(artifact)
    report["iteration"] = run_id
    metrics = report["metrics"]
    history = list(record.get("history") or [])
    history.append(
        {
            "iteration": run_id,
            "strategy_level": level,
            "model": report["winner"],
            "metrics": metrics,
            "candidates": report["candidates"],
            "path": str(artifact),
        }
    )
    updates: dict[str, Any] = {
        "training_run_seq": run_id,
        "training_run_id": run_id,
        "training_approved": True,
        "history": history,
    }
    best = record.get("best") or {}
    best_f1 = float((best.get("metrics") or {}).get("f1") or -1)
    if float(metrics["f1"]) >= best_f1:
      chosen = saver.save_model(model)
      updates["best"] = {
          "model": report["winner"],
          "strategy_level": level,
          "metrics": metrics,
          "path": str(chosen),
          "run_id": run_id,
      }
    tool_context.state["training_approved"] = True
    tool_context.state["best_f1"] = f"{float(metrics['f1']):.4f}"
    tool_context.state["best_model"] = report["winner"]
    return report, updates

  return run_phase(
      tool_context,
      "run_training",
      fingerprint=lambda record: (
          f"strategy:{int(record.get('strategy_level') or 0)}:"
          f"models:{','.join(record.get('chosen_models') or [])}"
      ),
      compute=compute,
      table="clean",
  )


def evaluate_model(tool_context: ToolContext) -> dict[str, Any]:
  """Compare the latest trial with the objective and say if the loop can stop."""

  def compute(
      frame: pd.DataFrame, record: dict[str, Any]
  ) -> tuple[dict[str, Any], dict[str, Any]]:
    del frame
    history = record.get("history") or []
    if not history:
      raise RuntimeError("evaluate_model requires a finished training trial.")
    latest = history[-1]
    metrics = latest["metrics"]
    objective = record["objective"]
    metric_name = str(objective.get("metric") or "f1")
    if metric_name not in metrics:
      metric_name = "f1"
    score = float(metrics[metric_name])
    target = float(objective["target"])
    target_met = score >= target
    eval_count = int(record.get("eval_count") or 0) + 1
    result = {
        "target_met": target_met,
        "metric": metric_name,
        "f1": float(metrics["f1"]),
        "score": score,
        "target": target,
        "model": latest["model"],
        "strategy_level": latest["strategy_level"],
        "eval_count": eval_count,
        "run_id": record.get("training_run_id"),
    }
    tool_context.state["target_met"] = "true" if target_met else "false"
    tool_context.state["best_f1"] = f"{float(metrics['f1']):.4f}"
    return result, {"eval_count": eval_count, "target_met": target_met}

  return run_phase(
      tool_context,
      "evaluate_model",
      fingerprint=lambda record: (
          f"run:{record.get('training_run_id')}:target:{record['objective']['target']}"
      ),
      compute=compute,
      table="clean",
  )


def record_improvement_plan(tool_context: ToolContext) -> dict[str, Any]:
  """Raise the strategy level after a miss so the next loop trains a stronger set."""
  handle, experiment_id = store_and_id(tool_context)
  outcome: dict[str, Any] = {}

  def mutate(record: dict[str, Any]) -> None:
    marker = record.get("eval_count")
    if record.get("planned_for_eval") == marker and record.get("last_plan"):
      outcome["result"] = {
          "status": "ok",
          "cached": True,
          "strategy_level": record.get("strategy_level"),
          "plan": record.get("last_plan"),
          "target_met": False,
      }
    else:
      options = (
          (record.get("phases") or {}).get("select_models", {}).get("result") or {}
      ).get("options") or []
      tried = {item.get("model") for item in (record.get("history") or [])}
      remaining = [
          item["id"]
          for item in options
          if item.get("id") and item["id"] not in tried
      ]
      level = int(record.get("strategy_level") or 0)
      if remaining:
        record["chosen_models"] = [remaining[0]]
        plan = (
            f"Train `{remaining[0]}`, the other recommended family, "
            "and compare F1 with the last trial."
        )
      else:
        level += 1
        record["strategy_level"] = level
        plan = PLANS.get(level, f"Raise the strategy to level {level} and train again.")
      record["planned_for_eval"] = marker
      record["last_plan"] = plan
      plans = list(record.get("plans") or [])
      plans.append({"strategy_level": level, "plan": plan, "models": record.get("chosen_models")})
      record["plans"] = plans
      outcome["result"] = {
          "status": "ok",
          "strategy_level": level,
          "plan": plan,
          "models": record.get("chosen_models"),
          "target_met": False,
      }
    record["trajectory"] = list(record.get("trajectory") or []) + [
        "record_improvement_plan"
    ]

  handle.update(experiment_id, mutate)
  result = outcome["result"]
  tool_context.state["strategy_level"] = str(result["strategy_level"])
  tool_context.state["last_plan"] = result["plan"]
  return result


def write_model_metadata(tool_context: ToolContext) -> dict[str, Any]:
  """Record the chosen algorithm, features, threshold, and evaluation metrics."""

  def compute(frame: pd.DataFrame, record: dict[str, Any]) -> dict[str, Any]:
    del frame
    best = record.get("best") or {}
    features = (record.get("phases") or {}).get("create_features", {}).get("result") or {}
    if not best:
      raise RuntimeError("write_model_metadata requires a chosen model.")
    return {
        "algorithm": best.get("model"),
        "strategy_level": best.get("strategy_level"),
        "artifact": best.get("path"),
        "metrics": best.get("metrics") or {},
        "positive_label": ((record.get("phases") or {}).get("run_training", {}).get("result") or {}).get(
            "positive_label"
        ),
        "decision_threshold": (
            (record.get("phases") or {}).get("run_training", {}).get("result") or {}
        ).get("decision_threshold"),
        "feature_columns": (features.get("numeric") or []) + (features.get("categorical") or []),
        "run_id": best.get("run_id"),
    }

  return run_phase(
      tool_context,
      "write_model_metadata",
      fingerprint=lambda record: f"best:{(record.get('best') or {}).get('run_id')}",
      compute=compute,
      table="clean",
  )


def publish_serving_contract(tool_context: ToolContext) -> dict[str, Any]:
  """Point FastAPI at the saved artifact. The process is already the serving backend."""

  def compute(frame: pd.DataFrame, record: dict[str, Any]) -> dict[str, Any]:
    del frame
    best = record.get("best") or {}
    if not best.get("path"):
      raise RuntimeError("publish_serving_contract requires a saved model.")
    experiment_id = record["id"]
    return {
        "service": "fastapi",
        "method": "POST",
        "path": f"/experiments/{experiment_id}/predict",
        "health": "/health",
        "artifact": best.get("path"),
        "request": {"rows": ["one object per prediction, using the raw feature columns"]},
        "response": {"predictions": ["label and positive-class probability"]},
    }

  return run_phase(
      tool_context,
      "publish_serving_contract",
      fingerprint=lambda record: f"serve:{(record.get('best') or {}).get('path')}",
      compute=compute,
      table="clean",
  )


def recall_similar_experiments(tool_context: ToolContext) -> dict[str, Any]:
  """Read long-term memory of earlier experiments for this runtime."""
  handle, experiment_id = store_and_id(tool_context)
  path = handle.root / "memory.jsonl"
  memories: list[dict[str, Any]] = []
  if path.exists():
    for line in path.read_text(encoding="utf-8").splitlines():
      if not line.strip():
        continue
      item = json.loads(line)
      if item.get("id") != experiment_id:
        memories.append(item)
  memories = memories[-3:]

  def mutate(record: dict[str, Any]) -> None:
    record["trajectory"] = list(record.get("trajectory") or []) + [
        "recall_similar_experiments"
    ]

  handle.update(experiment_id, mutate)
  return {"status": "ok", "memories": memories}


def write_solution_design(tool_context: ToolContext) -> dict[str, Any]:
  """Write DESIGN.md from the experiment record and remember the outcome."""
  handle, experiment_id = store_and_id(tool_context)
  record = handle.load(experiment_id)
  document = render_design(record)
  path = ArtifactSaver(experiment_dir(tool_context)).save_design(document)
  _remember(handle, record, str(path))

  def mutate(current: dict[str, Any]) -> None:
    current["design_path"] = str(path)
    current["status"] = "completed"
    phases = dict(current.get("phases") or {})
    phases["write_solution_design"] = {
        "status": "ok",
        "fingerprint": f"history:{len(current.get('history') or [])}",
        "result": {"path": str(path), "chars": len(document)},
    }
    current["phases"] = phases
    current["trajectory"] = list(current.get("trajectory") or []) + [
        "write_solution_design"
    ]

  handle.update(experiment_id, mutate)
  tool_context.state["design_path"] = str(path)
  return {"status": "ok", "path": str(path), "chars": len(document)}


def _remember(handle: ExperimentStore, record: dict[str, Any], design_path: str) -> None:
  best = record.get("best") or {}
  entry = {
      "id": record["id"],
      "objective": (record.get("objective") or {}).get("text"),
      "target_column": (record.get("objective") or {}).get("target_column"),
      "model": best.get("model"),
      "f1": (best.get("metrics") or {}).get("f1"),
      "design_path": design_path,
  }
  path = handle.root / "memory.jsonl"
  with path.open("a", encoding="utf-8") as handle_file:
    handle_file.write(json.dumps(entry) + "\n")
