"""Shared table IO. Roles decide what to do with the frame."""

from __future__ import annotations

import pandas as pd

ID_SUFFIX = "_id"


def read_table(path: str) -> pd.DataFrame:
  file_path = str(path)
  if file_path.endswith(".parquet"):
    return pd.read_parquet(file_path)
  return pd.read_csv(file_path)


def id_columns(frame: pd.DataFrame) -> list[str]:
  return [
      column
      for column in frame.columns
      if column == "id" or column.endswith(ID_SUFFIX)
  ]


def predictor_kinds(
    frame: pd.DataFrame,
    target: str | None = None,
) -> tuple[list[str], list[str]]:
  """Split model inputs into numeric and categorical columns.

  The target and id columns are not predictors. A categorical target therefore
  does not increase the categorical feature count.
  """
  skipped = set(id_columns(frame))
  if target:
    skipped.add(target)
  numeric: list[str] = []
  categorical: list[str] = []
  for column in frame.columns:
    if column in skipped:
      continue
    if pd.api.types.is_numeric_dtype(frame[column]):
      numeric.append(column)
    else:
      categorical.append(column)
  return numeric, categorical


def describe_categoricals(frame: pd.DataFrame, target: str | None = None) -> str:
  """Say what a categorical count means, including a count of zero."""
  numeric, categorical = predictor_kinds(frame, target)
  if categorical:
    shown = categorical[:8]
    named = ", ".join(f"`{name}`" for name in shown)
    extra = "" if len(categorical) <= 8 else f" (+{len(categorical) - 8} more)"
    return (
        f"Categorical features: {len(categorical)} ({named}{extra}). "
        f"Numeric predictors: {len(numeric)}."
    )
  cats = [
      column
      for column in frame.columns
      if column not in set(id_columns(frame))
      and not pd.api.types.is_numeric_dtype(frame[column])
  ]
  if len(cats) == 1 and cats[0] == target:
    return (
        f"Categorical features are 0. `{target}` is the only categorical column, "
        "and it is the target, so it is not a feature. "
        f"The other {len(numeric)} columns are numeric predictors. "
        "Workaround: do not one-hot encode; keep the predictors numeric "
        "and prefer logistic regression."
    )
  if cats:
    named = ", ".join(f"`{name}`" for name in cats)
    return (
        f"Categorical features are 0. Categorical column(s) {named} are not predictors. "
        f"The other {len(numeric)} columns are numeric. "
        "Workaround: do not one-hot encode; keep the predictors numeric "
        "and prefer logistic regression."
    )
  return (
      f"Categorical features are 0. All {len(numeric)} predictors are numeric. "
      "Workaround: skip one-hot encoding and prefer logistic regression."
  )
