"""Strategy-level feature engineering and the spec the trainer consumes."""

from __future__ import annotations

from typing import Any

import pandas as pd

from team.tables import describe_categoricals
from team.tables import id_columns
from team.tables import predictor_kinds

# These match the ColumnTransformer in team.ml_engineer.train.
_TRANSFORMS = {
    "numeric": ["median_impute", "standard_scale"],
    "categorical": ["most_frequent_impute", "one_hot"],
    "engineered": ["median_impute", "standard_scale"],
    "id": [],
    "target": [],
}

_ENGINEERED = {
    "charges_per_tenure": {
        "level": 1,
        "inputs": ["monthly_charges", "tenure_months"],
        "formula": "monthly_charges / clip(tenure_months, lower=1)",
    },
    "is_month_to_month": {
        "level": 1,
        "inputs": ["contract"],
        "formula": "contract == month-to-month",
    },
    "support_x_monthly": {
        "level": 2,
        "inputs": ["support_calls", "monthly_charges"],
        "formula": "support_calls * monthly_charges",
    },
}


def engineer(frame: pd.DataFrame, strategy_level: int) -> pd.DataFrame:
  """Add a few interactions once the baseline has missed the target."""
  engineered = frame.copy()
  if strategy_level >= 1 and {"tenure_months", "monthly_charges"} <= set(engineered.columns):
    tenure = pd.to_numeric(engineered["tenure_months"], errors="coerce").clip(lower=1)
    monthly = pd.to_numeric(engineered["monthly_charges"], errors="coerce")
    engineered["charges_per_tenure"] = monthly / tenure
  if strategy_level >= 1 and "contract" in engineered.columns:
    engineered["is_month_to_month"] = (
        engineered["contract"].astype(str) == "month-to-month"
    ).astype(int)
  if strategy_level >= 2 and {"support_calls", "monthly_charges"} <= set(engineered.columns):
    support = pd.to_numeric(engineered["support_calls"], errors="coerce")
    monthly = pd.to_numeric(engineered["monthly_charges"], errors="coerce")
    engineered["support_x_monthly"] = support * monthly
  return engineered


def feature_columns(frame: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
  return predictor_kinds(frame, target)


def feature_spec(frame: pd.DataFrame, strategy_level: int, target: str) -> dict[str, Any]:
  """Column configuration for this strategy level, including engineered fields."""
  engineered = engineer(frame, strategy_level)
  extra = [column for column in engineered.columns if column not in frame.columns]
  numeric, categorical = feature_columns(engineered, target)
  return {
      "strategy_level": strategy_level,
      "target": target,
      "base_columns": [column for column in frame.columns if column != target],
      "engineered": extra,
      "numeric": numeric,
      "categorical": categorical,
      "columns": _column_config(engineered, target, extra, numeric, categorical),
      "categorical_summary": describe_categoricals(engineered, target),
  }


def _column_config(
    frame: pd.DataFrame,
    target: str,
    engineered: list[str],
    numeric: list[str],
    categorical: list[str],
) -> list[dict[str, Any]]:
  numeric_names = set(numeric)
  categorical_names = set(categorical)
  engineered_names = set(engineered)
  ids = set(id_columns(frame))
  columns: list[dict[str, Any]] = []
  for name in frame.columns:
    if name == target:
      role = "target"
    elif name in ids:
      role = "id"
    elif name in engineered_names:
      role = "engineered"
    elif name in categorical_names:
      role = "categorical"
    elif name in numeric_names:
      role = "numeric"
    else:
      role = "unused"
    record: dict[str, Any] = {
        "name": name,
        "role": role,
        "dtype": str(frame[name].dtype),
        "transform": list(_TRANSFORMS.get(role, [])),
    }
    if name in _ENGINEERED:
      record.update(_ENGINEERED[name])
    columns.append(record)
  return columns
