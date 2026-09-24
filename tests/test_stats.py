"""Breast cancer has one categorical column, and it is the target."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd

from runtime.orchestrator import ExperimentRequest
from runtime.orchestrator import MlRuntime
from runtime.stages import board
from team.config import PROJECT_ROOT
from team.data_engineer.sample import make_churn_frame
from team.data_scientist.selection import recommend_two
from team.researcher.brief import research_brief
from team.researcher.brief import statistical_markdown
from team.researcher.profile import profile_quality


def _breast() -> pd.DataFrame:
  return pd.read_csv(PROJECT_ROOT / "data" / "datasets" / "breast_cancer.csv")


def test_breast_cancer_categorical_zero_is_the_target() -> None:
  frame = _breast()
  quality = profile_quality(frame, "diagnosis")

  assert quality["categorical_predictors"] == 0
  assert quality["numeric_predictors"] == 30
  summary = quality["categorical_summary"]
  assert "diagnosis" in summary
  assert "target" in summary
  assert "one-hot" in summary
  assert quality["outlier_cells"] > 0
  assert quality["spread"]
  for item in quality["spread"]:
    assert {"mean", "median", "mode", "std"} <= set(item)

  report = research_brief(
      frame,
      {
          "objective": {
              "text": "Predict tumor diagnosis.",
              "target_column": "diagnosis",
          },
          "phases": {},
          "sme_notes": [
              {
                  "stage": "stats",
                  "prompt": "look into categorical values which are zero",
              }
          ],
      },
  )
  text = statistical_markdown(report)
  assert "Median" in text
  assert "Mode" in text
  assert "Outliers" in text
  assert "diagnosis" in text
  assert "look into categorical values which are zero" in text


def test_churn_keeps_real_categorical_features() -> None:
  frame = make_churn_frame(rows=40, seed=1)
  quality = profile_quality(frame, "churn")
  assert quality["categorical_predictors"] >= 2
  assert "are 0" not in quality["categorical_summary"]
  picked = recommend_two(frame, "churn")
  assert picked["table"]["categorical"] >= 2


def test_breast_cancer_model_choice_skips_one_hot() -> None:
  frame = _breast()
  picked = recommend_two(frame, "diagnosis")
  assert picked["table"]["categorical"] == 0
  assert "one-hot" in picked["categorical_summary"]
  logistic = next(item for item in picked["options"] if item["id"] == "logistic_regression")
  assert "one-hot" in logistic["reason"]


def test_stats_board_answers_a_zero_categorical_note() -> None:
  frame = _breast()
  quality = profile_quality(frame, "diagnosis")
  record = {
      "id": "breast",
      "status": "awaiting_review",
      "current_stage": "stats",
      "checkpoints": ["stats"],
      "objective": {"target_column": "diagnosis"},
      "sme_notes": [
          {
              "stage": "stats",
              "prompt": "look into categorical values which are zero",
          }
      ],
      "phases": {
          "inspect_schema": {
              "status": "ok",
              "result": {"rows": len(frame), "columns": [{}] * frame.shape[1]},
          },
          "profile_quality": {"status": "ok", "result": quality},
          "analyze_target": {
              "status": "ok",
              "result": {"classes": {"benign": 357, "malignant": 212}},
          },
          "frame_technical_spec": {
              "status": "ok",
              "result": {
                  "target_column": "diagnosis",
                  "task_type": "classification",
                  "technical_problem": "Classify diagnosis.",
              },
          },
      },
  }
  rendered = board(record)
  stats = next(item for item in rendered["stages"] if item["id"] == "stats")
  labels = {item["label"]: item["value"] for item in stats["report"]["numbers"]}
  assert labels["Categorical"] == 0
  assert labels["Numeric"] == 30
  assert labels["Outliers"] > 0
  text = " ".join(stats["report"]["findings"])
  assert "look into categorical values which are zero" in text
  assert "diagnosis" in text
  assert "mean" in text
  assert "median" in text
  assert "std" in text


def test_rework_prompt_is_on_the_live_stats_board(tmp_path: Path) -> None:
  runtime = MlRuntime(root=tmp_path, model_name="scripted")
  first = asyncio.run(
      runtime.start(
          ExperimentRequest(
              sample_id="breast_cancer",
              objective="Predict tumor diagnosis (malignant vs benign). Optimize for F1.",
              target_column="diagnosis",
              target_value=0.0,
              approve_training=False,
              staged=True,
              checkpoints=["stats"],
          )
      )
  )
  assert first["status"] == "awaiting_review", first.get("error")
  again = asyncio.run(
      runtime.rework_stage(
          first["id"],
          "look into categorical values which are zero",
      )
  )
  assert again["status"] == "awaiting_review", again.get("error")
  stats = next(item for item in again["stages"] if item["id"] == "stats")
  labels = {item["label"]: item["value"] for item in stats["report"]["numbers"]}
  assert labels["Categorical"] == 0
  text = " ".join(stats["report"]["findings"])
  assert "look into categorical values which are zero" in text
  assert "`diagnosis`" in text
  assert "mean" in text
  assert "Outliers" in text or "outliers" in text
