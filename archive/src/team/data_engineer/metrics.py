"""Choose the scores for a run from the target column and the question.

Classification and regression do not share a score list. Accuracy, F1,
precision, and recall are only loaded when the target is a category.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

CLASSIFICATION_METRICS = ("f1", "precision", "recall", "accuracy")
REGRESSION_METRICS = ("mae", "rmse", "r2")

_REGRESSION_WORDS = (
    "price",
    "cost",
    "revenue",
    "amount",
    "forecast",
    "how much",
    "how many",
    "regress",
    "salary",
    "income",
    "temperature",
    "duration",
    "quantity",
)
_CLASSIFICATION_WORDS = (
    "classif",
    "churn",
    "cancel",
    "diagnosis",
    "category",
    "which ",
    "spam",
    "fraud",
    "label",
    "benign",
    "malignant",
)


def choose_metrics(
    frame: pd.DataFrame,
    target: str,
    requirement: str = "",
) -> dict[str, Any]:
  """Return the task and the metric list data engineering should load."""
  if target not in frame.columns:
    raise ValueError(f"Target column {target!r} is not in the dataset.")
  task = _task_type(frame[target], requirement or "")
  if task == "regression":
    return {
        "task_type": "regression",
        "metric": "rmse",
        "metrics": list(REGRESSION_METRICS),
        "reason": (
            "The target is a continuous number, so the scores are MAE, RMSE, and R²."
        ),
    }
  return {
      "task_type": "classification",
      "metric": "f1",
      "metrics": list(CLASSIFICATION_METRICS),
      "reason": (
          "The target is a category, so the scores are F1, precision, recall, and accuracy."
      ),
  }


def _task_type(series: pd.Series, requirement: str) -> str:
  text = requirement.lower()
  class_hit = any(word in text for word in _CLASSIFICATION_WORDS)
  reg_hit = any(word in text for word in _REGRESSION_WORDS)
  quantity = _looks_like_quantity(series)
  label = _looks_like_label(series)
  if class_hit and not reg_hit:
    return "classification"
  if reg_hit and not class_hit:
    if quantity or not label:
      return "regression"
    return "classification"
  if quantity:
    return "regression"
  return "classification"


def _looks_like_label(series: pd.Series) -> bool:
  values = series.dropna()
  if values.empty:
    return True
  unique = int(values.nunique())
  if not pd.api.types.is_numeric_dtype(values):
    return unique <= 20
  return unique <= 12 and _mostly_whole(values)


def _looks_like_quantity(series: pd.Series) -> bool:
  values = series.dropna()
  if values.empty or not pd.api.types.is_numeric_dtype(values):
    return False
  unique = int(values.nunique())
  if unique <= 2:
    return False
  if unique <= 12 and _mostly_whole(values):
    return False
  return True


def _mostly_whole(values: pd.Series) -> bool:
  numeric = pd.to_numeric(values, errors="coerce").dropna()
  if numeric.empty:
    return False
  return bool(((numeric - numeric.round()).abs() < 1e-9).mean() > 0.95)
