"""Data engineering picks scores from the column and the question."""

from __future__ import annotations

import pandas as pd

from runtime.stages import board
from team.data_engineer.metrics import choose_metrics
from team.data_engineer.sample import make_churn_frame


def test_category_target_loads_classification_scores() -> None:
  frame = make_churn_frame(rows=80, seed=1)
  choice = choose_metrics(frame, "churn", "Which customers will cancel?")

  assert choice["task_type"] == "classification"
  assert choice["metrics"] == ["f1", "precision", "recall", "accuracy"]
  assert "MAE" not in choice["reason"]


def test_continuous_target_does_not_load_classification_scores() -> None:
  frame = pd.DataFrame(
      {
          "area": [40, 55, 80, 120, 30, 90],
          "rooms": [2, 3, 4, 5, 1, 4],
          "price": [210.5, 340.0, 510.25, 800.0, 150.5, 640.75],
      }
  )
  choice = choose_metrics(frame, "price", "Predict the sale price.")

  assert choice["task_type"] == "regression"
  assert choice["metrics"] == ["mae", "rmse", "r2"]
  assert "f1" not in choice["metrics"]
  assert "accuracy" not in choice["metrics"]


def test_prepare_stage_shows_only_the_loaded_scores() -> None:
  record = {
      "id": "price",
      "status": "awaiting_review",
      "current_stage": "prepare",
      "checkpoints": ["prepare"],
      "objective": {"text": "Predict the sale price.", "metrics": ["mae", "rmse", "r2"]},
      "phases": {
          "clean_dataset": {
              "status": "ok",
              "result": {
                  "rows_before": 6,
                  "rows_after": 6,
                  "duplicate_rows_removed": 0,
                  "missing_after": 0,
                  "task_type": "regression",
                  "metrics": ["mae", "rmse", "r2"],
                  "metric_reason": (
                      "The target is a continuous number, so the scores are MAE, RMSE, and R²."
                  ),
              },
          }
      },
  }
  rendered = board(record)
  prepare = next(item for item in rendered["stages"] if item["id"] == "prepare")
  labels = {item["label"]: item["value"] for item in prepare["report"]["numbers"]}
  text = " ".join(prepare["report"]["findings"])

  assert labels["Task"] == "regression"
  assert labels["Scores"] == "MAE, RMSE, R²"
  assert "F1" not in labels["Scores"]
  assert "Accuracy" not in text
