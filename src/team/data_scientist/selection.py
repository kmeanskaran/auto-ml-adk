"""Recommend two model families for this table. The SME picks one."""

from __future__ import annotations

from typing import Any

import pandas as pd

from team.tables import describe_categoricals
from team.tables import predictor_kinds

MODEL_TITLES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
}

MODEL_BLURBS = {
    "logistic_regression": "A linear model. Strong when the signal is mostly additive.",
    "random_forest": "A tree ensemble. Handles mixed types and interactions.",
    "gradient_boosting": "Boosted trees. Useful when there is enough data to fit a sharper boundary.",
}


def recommend_two(frame: pd.DataFrame, target: str) -> dict[str, Any]:
  """Score the small catalog on this table and keep the top two."""
  rows = int(len(frame))
  numeric_columns, categorical_columns = predictor_kinds(frame, target)
  numeric = len(numeric_columns)
  categorical = len(categorical_columns)
  classes = 0
  if target in frame.columns:
    classes = int(frame[target].nunique(dropna=True))
  summary = describe_categoricals(frame, target)

  scores: dict[str, tuple[float, str]] = {
      "logistic_regression": (
          3.0 + (1.5 if categorical == 0 else 0.0) + (0.5 if rows < 800 else 0.0),
          "Categorical features are 0, so skip one-hot encoding. "
          "A linear model is the workaround on an all-numeric table."
          if categorical == 0
          else "A simple linear baseline next to a tree model.",
      ),
      "random_forest": (
          3.2 + (1.0 if categorical else 0.4) + (0.4 if classes > 2 else 0.0),
          "Trees handle mixed columns and non-linear splits without much tuning.",
      ),
      "gradient_boosting": (
          (2.4 if rows >= 250 else 1.1) + (0.3 if classes == 2 else 0.0),
          "Worth a shot when there are enough rows to support a sharper fit."
          if rows >= 250
          else "Easy to overfit on a small table, so it ranks lower here.",
      ),
  }
  ranked = sorted(scores.items(), key=lambda item: item[1][0], reverse=True)
  options = []
  for name, (_score, reason) in ranked[:2]:
    options.append(
        {
            "id": name,
            "title": MODEL_TITLES[name],
            "reason": reason,
            "blurb": MODEL_BLURBS[name],
        }
    )
  decision = (
      f"Two families fit this table ({rows} rows, {numeric} numeric, "
      f"{categorical} categorical, {classes} classes). {summary} Pick one to train."
  )
  return {
      "task_type": "classification",
      "decision": decision,
      "categorical_summary": summary,
      "options": options,
      "table": {
          "rows": rows,
          "numeric": numeric,
          "categorical": categorical,
          "classes": classes,
      },
  }
