"""The ML tools have to beat a majority baseline on the sample table."""

from __future__ import annotations

import pandas as pd

from team.data_engineer.sample import inject_missing
from team.data_engineer.sample import make_churn_frame
from team.data_engineer.sandbox import execute_snippet
from team.ml_engineer.train import train_candidates


def test_chosen_models_beat_the_majority() -> None:
  frame = make_churn_frame(rows=400, seed=7)
  _, baseline = train_candidates(
      frame, "churn", strategy_level=0, models=["majority_baseline"]
  )
  _, chosen = train_candidates(
      frame, "churn", strategy_level=1, models=["logistic_regression", "random_forest"]
  )

  assert baseline["winner"] == "majority_baseline"
  assert baseline["metrics"]["f1"] < 0.2
  assert chosen["metrics"]["f1"] >= 0.75
  assert chosen["winner"] != "majority_baseline"
  assert len(chosen["candidates"]) == 2


def test_saved_model_scores_raw_rows() -> None:
  frame = make_churn_frame(rows=200, seed=3)
  model, report = train_candidates(
      frame, "churn", strategy_level=1, models=["random_forest"]
  )
  row = frame.drop(columns=["churn"]).iloc[0].to_dict()
  predictions = model.predict_records([row])

  assert report["metrics"]["f1"] >= 0.7
  assert predictions[0]["label"] in (0, 1)
  assert 0.0 <= predictions[0]["probability"] <= 1.0


def test_missing_cells_are_visible() -> None:
  dirty = inject_missing(make_churn_frame(rows=80, seed=1))
  assert int(dirty.isna().sum().sum()) > 0
  assert isinstance(dirty, pd.DataFrame)


def test_snippet_sandbox_allows_a_reduction_and_blocks_imports() -> None:
  frame = make_churn_frame(rows=30, seed=1)
  allowed = execute_snippet(frame, "result = float(df['monthly_charges'].mean())")
  assert allowed["status"] == "ok"
  assert allowed["result"] > 0

  try:
    execute_snippet(frame, "import os\nresult = 1")
  except ValueError as exc:
    assert "import" in str(exc)
  else:
    raise AssertionError("import should be rejected")
