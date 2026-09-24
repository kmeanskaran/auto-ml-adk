"""The artifact saver persists the column configuration beside an experiment."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from team.artifacts import ArtifactSaver
from team.data_engineer.sample import make_churn_frame
from team.data_scientist.features import feature_spec


def test_feature_pipeline_saves_column_roles(tmp_path: Path) -> None:
  frame = make_churn_frame(rows=40, seed=1)
  spec = feature_spec(frame, strategy_level=1, target="churn")
  saver = ArtifactSaver(tmp_path)
  path = saver.save_feature_pipeline(spec)

  loaded = saver.load_feature_pipeline()
  by_name = {column["name"]: column for column in loaded["columns"]}

  assert path == tmp_path / "feature_pipeline.json"
  assert loaded["target"] == "churn"
  assert loaded["strategy_level"] == 1
  assert by_name["customer_id"]["role"] == "id"
  assert by_name["customer_id"]["transform"] == []
  assert by_name["churn"]["role"] == "target"
  assert by_name["contract"]["role"] == "categorical"
  assert by_name["contract"]["transform"] == ["most_frequent_impute", "one_hot"]
  assert by_name["monthly_charges"]["role"] == "numeric"
  assert by_name["monthly_charges"]["transform"] == ["median_impute", "standard_scale"]
  assert by_name["charges_per_tenure"]["role"] == "engineered"
  assert by_name["charges_per_tenure"]["inputs"] == ["monthly_charges", "tenure_months"]
  assert by_name["is_month_to_month"]["formula"] == "contract == month-to-month"
  assert "support_x_monthly" not in by_name
  assert not list(tmp_path.glob("*.tmp"))


def test_later_strategy_adds_the_support_interaction(tmp_path: Path) -> None:
  frame = make_churn_frame(rows=20, seed=2)
  spec = feature_spec(frame, strategy_level=2, target="churn")
  loaded = ArtifactSaver(tmp_path).save_feature_pipeline(spec)
  saved = ArtifactSaver(tmp_path).load_feature_pipeline()
  support = next(column for column in saved["columns"] if column["name"] == "support_x_monthly")

  assert loaded.name == "feature_pipeline.json"
  assert support["role"] == "engineered"
  assert support["level"] == 2
  assert support["inputs"] == ["support_calls", "monthly_charges"]


def test_saver_rejects_a_path_and_reloads_a_model(tmp_path: Path) -> None:
  saver = ArtifactSaver(tmp_path / "run")
  with pytest.raises(ValueError):
    saver.save_text("../escape.md", "no")

  payload = {"label": "kept"}
  saver.save_model(payload, "model.joblib")

  assert joblib.load(tmp_path / "run" / "model.joblib") == payload
  assert not list((tmp_path / "run").glob("*.tmp"))
