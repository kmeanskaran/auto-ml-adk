"""The SME board runs one lifecycle stage, then waits for next or rework."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from api.app import create_app
from runtime.orchestrator import ExperimentRequest
from runtime.orchestrator import MlRuntime


def test_staged_stats_then_next(tmp_path: Path) -> None:
  runtime = MlRuntime(root=tmp_path, model_name="scripted")
  board = asyncio.run(
      runtime.start(
          ExperimentRequest(
              target_value=0.0,
              approve_training=False,
              staged=True,
              checkpoints=["stats", "prepare"],
          )
      )
  )
  assert board["status"] == "awaiting_review"
  assert board["current_stage"] == "stats"
  stats = next(item for item in board["stages"] if item["id"] == "stats")
  assert stats["status"] == "review"
  assert stats["report"]["numbers"]
  assert stats["report"]["findings"]
  assert all(item["status"] == "pending" for item in board["stages"] if item["id"] != "stats")

  prepared = asyncio.run(runtime.next_stage(board["id"]))
  assert prepared["status"] == "awaiting_review"
  assert prepared["current_stage"] == "prepare"
  prepare = next(item for item in prepared["stages"] if item["id"] == "prepare")
  assert prepare["report"]["numbers"]


def test_rework_stats_keeps_the_sme_note(tmp_path: Path) -> None:
  runtime = MlRuntime(root=tmp_path, model_name="scripted")
  board = asyncio.run(
      runtime.start(
          ExperimentRequest(
              target_value=0.0,
              approve_training=False,
              staged=True,
              checkpoints=["stats"],
          )
      )
  )
  again = asyncio.run(
      runtime.rework_stage(board["id"], "Focus on class imbalance in the brief.")
  )
  assert again["status"] == "awaiting_review"
  assert again["current_stage"] == "stats"
  assert "class imbalance" in again["sme_notes"][-1]["prompt"]


def test_sample_and_model_choice(tmp_path: Path) -> None:
  runtime = MlRuntime(root=tmp_path, model_name="scripted")
  board = asyncio.run(
      runtime.start(
          ExperimentRequest(
              sample_id="iris",
              objective="Predict iris species. Optimize for F1.",
              target_column="species",
              target_value=0.5,
              approve_training=False,
              staged=True,
              checkpoints=["stats", "prepare", "model"],
          )
      )
  )
  assert board["status"] == "awaiting_review"
  prepared = asyncio.run(runtime.next_stage(board["id"]))
  modeled = asyncio.run(runtime.next_stage(prepared["id"]))
  assert modeled["current_stage"] == "model"
  assert modeled["can_next"] is False
  assert len(modeled["choices"]) == 2
  picked = modeled["choices"][0]["id"]
  chosen = runtime.choose_model(modeled["id"], picked)
  assert chosen["chosen_models"] == [picked]
  assert chosen["can_next"] is True


def test_datasets_endpoint_lists_samples(tmp_path: Path) -> None:
  client = TestClient(create_app(root=tmp_path, model="scripted"))
  response = client.get("/datasets")
  assert response.status_code == 200
  names = {item["id"] for item in response.json()}
  assert {"breast_cancer", "iris", "churn"} <= names


def test_http_board_upload_and_next(tmp_path: Path) -> None:
  client = TestClient(create_app(root=tmp_path, model="scripted"))
  created = client.post(
      "/experiments",
      json={
          "objective": "Predict customer churn. Optimize for F1.",
          "target_column": "churn",
          "target_value": 0.0,
          "approve_training": False,
          "staged": True,
          "checkpoints": ["stats", "prepare"],
      },
  )
  assert created.status_code == 200, created.text
  body = created.json()
  assert body["status"] == "awaiting_review"
  experiment_id = body["id"]
  board = client.get(f"/experiments/{experiment_id}/board")
  assert board.status_code == 200
  assert board.json()["can_next"] is True
  nxt = client.post(f"/experiments/{experiment_id}/next")
  assert nxt.status_code == 200, nxt.text
  assert nxt.json()["current_stage"] == "prepare"
  assert isinstance(nxt.json()["stages"], list)


def test_clear_removes_runs_awaiting_review(tmp_path: Path) -> None:
  client = TestClient(create_app(root=tmp_path, model="scripted"))
  created = client.post(
      "/experiments",
      json={
          "objective": "Predict customer churn. Optimize for F1.",
          "target_column": "churn",
          "target_value": 0.0,
          "approve_training": False,
          "staged": True,
          "checkpoints": ["stats"],
      },
  )
  assert created.status_code == 200, created.text
  body = created.json()
  assert body["status"] == "awaiting_review"
  listed = client.get("/experiments")
  assert listed.status_code == 200
  assert len(listed.json()) == 1

  cleared = client.delete("/experiments")
  assert cleared.status_code == 200, cleared.text
  assert cleared.json()["removed"] == 1
  assert client.get("/experiments").json() == []
  assert not (tmp_path / "experiments" / body["id"]).exists()
