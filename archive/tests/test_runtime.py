"""The ADK workflow, approval gate, and crash resume share one store."""

from __future__ import annotations

import asyncio
from pathlib import Path

from runtime.agents import build_orchestrator
from runtime.agents import build_stage_agent
from runtime.agents import resolve_model
from runtime.harness import HarnessPlugin
from runtime.orchestrator import ExperimentRequest
from runtime.orchestrator import MlRuntime
from team.data_engineer.sample import inject_missing
from team.data_engineer.sample import make_churn_frame
from team.store import ExperimentStore


def _runtime(tmp_path: Path) -> MlRuntime:
  return MlRuntime(root=tmp_path, model_name="scripted")


def test_delivery_stage_is_its_own_tree() -> None:
  model = resolve_model("scripted")
  stage = build_stage_agent(model, "deliver")
  assert stage.name == "deliver_stage"
  assert [agent.name for agent in stage.sub_agents] == ["delivery_panel", "report_agent"]
  assert build_orchestrator(resolve_model("scripted")).name == "ml_team"


def test_improvement_loop_picks_a_real_model(tmp_path: Path) -> None:
  runtime = _runtime(tmp_path)
  record = asyncio.run(
      runtime.start(
          ExperimentRequest(
              target_value=0.75,
              approve_training=False,
          )
      )
  )

  assert record["status"] == "completed", record.get("error")
  assert record["target_met"] is True
  best = record["best"]
  assert best["model"] != "majority_baseline"
  assert best["metrics"]["f1"] >= 0.75
  assert Path(record["design_path"]).exists()
  assert Path(best["path"]).exists()
  trajectory = record["trajectory"]
  for tool in (
      "inspect_schema",
      "profile_quality",
      "analyze_target",
      "clean_dataset",
      "create_features",
      "select_models",
      "run_training",
      "evaluate_model",
      "write_solution_design",
  ):
    assert tool in trajectory, trajectory
  options = record["phases"]["select_models"]["result"]["options"]
  assert len(options) == 2
  assert record.get("chosen_models")
  prediction = runtime.predict(
      record["id"],
      [
          {
              "tenure_months": 4,
              "monthly_charges": 95.0,
              "total_charges": 380.0,
              "contract": "month-to-month",
              "internet": "fiber",
              "support_calls": 4,
              "senior": 1,
              "paperless": 1,
          }
      ],
  )
  assert prediction["predictions"][0]["probability"] >= 0
  design = Path(record["design_path"]).read_text(encoding="utf-8")
  assert "FeaturePipeline" in design
  assert "POST /experiments/{id}/predict" in design
  assert record["phases"]["frame_technical_spec"]["result"]["task_type"] == "classification"
  assert record["stages"]["product_manager"] == "done"
  assert record["stages"]["researcher"] == "done"
  report = Path(record["phases"]["compile_research_brief"]["result"]["path"])
  assert report.exists()
  report_text = report.read_text(encoding="utf-8")
  assert "Statistical report" in report_text
  assert "Median" in report_text
  assert "Outliers" in report_text
  assert record["phases"]["select_models"]["result"]["options"]
  assert record["phases"]["write_model_metadata"]["result"]["algorithm"] == best["model"]
  assert record["phases"]["publish_serving_contract"]["result"]["path"].endswith("/predict")
  checkpoints = list((tmp_path / "experiments" / record["id"] / "checkpoints").glob("*.json"))
  assert checkpoints
  assert (tmp_path / "memory.jsonl").exists()


def test_training_waits_for_a_human(tmp_path: Path) -> None:
  runtime = _runtime(tmp_path)
  waiting = asyncio.run(
      runtime.start(ExperimentRequest(target_value=0.0, approve_training=True))
  )

  assert waiting["status"] == "awaiting_approval", waiting.get("error")
  assert waiting.get("best") is None
  assert waiting["pending_approval"]["tool"] == "run_training"
  assert not list((tmp_path / "experiments" / waiting["id"]).glob("model*.joblib"))

  finished = asyncio.run(runtime.approve(waiting["id"], confirmed=True))
  assert finished["status"] == "completed", finished.get("error")
  assert finished["best"]["model"] != "majority_baseline"
  assert finished["target_met"] is True


def test_rejected_training_does_not_fit(tmp_path: Path) -> None:
  runtime = _runtime(tmp_path)
  waiting = asyncio.run(
      runtime.start(ExperimentRequest(target_value=0.0, approve_training=True))
  )
  rejected = asyncio.run(runtime.approve(waiting["id"], confirmed=False))
  assert rejected["status"] == "rejected"
  assert rejected.get("best") is None


def test_interrupted_feature_step_resumes(tmp_path: Path) -> None:
  runtime = _runtime(tmp_path)
  interrupted = asyncio.run(
      runtime.start(
          ExperimentRequest(
              target_value=0.0,
              approve_training=False,
              fault_tool="create_features",
          )
      )
  )
  assert interrupted["status"] == "interrupted", interrupted
  assert interrupted["interrupted_tool"] == "create_features"
  assert "clean_dataset" in interrupted["trajectory"]
  assert "run_training" not in interrupted["trajectory"]

  resumed = asyncio.run(runtime.resume(interrupted["id"]))
  assert resumed["status"] == "completed", resumed.get("error")
  assert "run_training" in resumed["trajectory"]
  assert resumed["fault"]["remaining"] == 0


def test_missing_values_are_cleaned(tmp_path: Path) -> None:
  runtime = _runtime(tmp_path)
  dirty = inject_missing(make_churn_frame(rows=120, seed=2))
  record = asyncio.run(
      runtime.start(
          ExperimentRequest(
              frame=dirty,
              target_value=0.0,
              approve_training=False,
          )
      )
  )
  assert record["status"] == "completed", record.get("error")
  quality = record["phases"]["profile_quality"]["result"]
  cleaning = record["phases"]["clean_dataset"]["result"]
  assert quality["missing_cells"] > 0
  assert cleaning["missing_after"] == 0


def test_human_iteration_retrains(tmp_path: Path) -> None:
  runtime = _runtime(tmp_path)
  first = asyncio.run(
      runtime.start(ExperimentRequest(target_value=0.0, approve_training=False))
  )
  assert first["status"] == "completed", first.get("error")
  assert first["best"]["model"] != "majority_baseline"

  second = asyncio.run(
      runtime.iterate(
          first["id"],
          note="Try a stronger model",
          target_value=0.75,
      )
  )
  assert second["status"] == "completed", second.get("error")
  assert second["target_met"] is True
  assert second["best"]["model"] != "majority_baseline"
  assert "Try a stronger model" in second["notes"]
  design = Path(second["design_path"]).read_text(encoding="utf-8")
  assert "Try a stronger model" in design


def test_policy_blocks_the_code_tool(tmp_path: Path) -> None:
  store = ExperimentStore(tmp_path)
  store.create(
      {
          "id": "exp",
          "status": "running",
          "blocked_tools": ["run_analysis_snippet"],
          "trajectory": [],
          "phases": {},
      }
  )
  plugin = HarnessPlugin(store)

  class _Tool:
    name = "run_analysis_snippet"

  class _Context:
    state = {"experiment_id": "exp"}
    function_call_id = "call-1"
    agent_name = "data_agent"

  blocked = asyncio.run(
      plugin.before_tool_callback(tool=_Tool(), tool_args={"code": "result = 1"}, tool_context=_Context())
  )
  assert blocked is not None
  assert "blocked" in blocked["error"]
