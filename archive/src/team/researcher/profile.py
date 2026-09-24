"""Read-only views of the raw table: schema, quality, and target balance."""

from __future__ import annotations

from typing import Any

import pandas as pd

from team.tables import describe_categoricals
from team.tables import id_columns
from team.tables import predictor_kinds


def inspect_schema(frame: pd.DataFrame) -> dict[str, Any]:
  columns = []
  for name in frame.columns:
    series = frame[name]
    columns.append(
        {
            "name": name,
            "dtype": str(series.dtype),
            "unique": int(series.nunique(dropna=True)),
            "missing": int(series.isna().sum()),
        }
    )
  return {"rows": int(len(frame)), "columns": columns}


def profile_quality(frame: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
  missing = {
      column: int(frame[column].isna().sum())
      for column in frame.columns
      if int(frame[column].isna().sum()) > 0
  }
  constant = [
      column
      for column in frame.columns
      if frame[column].nunique(dropna=True) <= 1
  ]
  duplicate_ids = 0
  ids = id_columns(frame)
  if ids:
    duplicate_ids = int(frame.duplicated(subset=ids).sum())
  distributions = column_profile(frame, target)
  distributions.pop("numeric_summary", None)
  return {
      "rows": int(len(frame)),
      "missing_cells": int(frame.isna().sum().sum()),
      "missing_by_column": missing,
      "constant_columns": constant,
      "duplicate_id_rows": duplicate_ids,
      **distributions,
  }


def column_profile(frame: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
  """Mean, median, mode, std, and IQR outliers, plus the categorical reading."""
  ids = set(id_columns(frame))
  numeric_predictors, _categorical = predictor_kinds(frame, target)
  numeric_summary: dict[str, Any] = {}
  categorical_columns: list[dict[str, Any]] = []
  for column in frame.columns:
    if column in ids:
      continue
    series = frame[column]
    if pd.api.types.is_numeric_dtype(series):
      numeric_summary[column] = _numeric_stats(series)
    else:
      categorical_columns.append(_categorical_stats(series, column, target))
  predictor_summary = {
      name: numeric_summary[name]
      for name in numeric_predictors
      if name in numeric_summary
  }
  outlier_columns = sorted(
      (
          {"name": name, "outliers": int(stats["outliers"])}
          for name, stats in predictor_summary.items()
          if int(stats["outliers"]) > 0
      ),
      key=lambda item: item["outliers"],
      reverse=True,
  )
  ranked = sorted(
      predictor_summary.items(),
      key=lambda item: item[1]["std"] if item[1]["std"] is not None else -1,
      reverse=True,
  )
  spread = [
      {
          "name": name,
          "mean": stats["mean"],
          "median": stats["median"],
          "mode": stats["mode"],
          "std": stats["std"],
          "outliers": stats["outliers"],
      }
      for name, stats in ranked[:3]
  ]
  return {
      "numeric_predictors": len(numeric_predictors),
      "categorical_predictors": len(_categorical),
      "categorical_columns": categorical_columns,
      "categorical_summary": describe_categoricals(frame, target),
      "outlier_cells": int(sum(item["outliers"] for item in outlier_columns)),
      "outlier_columns": outlier_columns[:5],
      "spread": spread,
      "numeric_summary": numeric_summary,
  }


def _numeric_stats(series: pd.Series) -> dict[str, Any]:
  clean = pd.to_numeric(series, errors="coerce").dropna()
  missing = int(series.isna().sum())
  empty = {
      "missing": missing,
      "mean": None,
      "median": None,
      "mode": None,
      "std": None,
      "min": None,
      "max": None,
      "outliers": 0,
  }
  if clean.empty:
    return empty
  q1 = float(clean.quantile(0.25))
  q3 = float(clean.quantile(0.75))
  iqr = q3 - q1
  outliers = 0
  if iqr > 0:
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outliers = int(((clean < low) | (clean > high)).sum())
  counts = clean.value_counts()
  mode = _round(counts.index[0]) if int(counts.iloc[0]) > 1 else None
  return {
      "missing": missing,
      "mean": _round(clean.mean()),
      "median": _round(clean.median()),
      "mode": mode,
      "std": None if len(clean) < 2 else _round(clean.std()),
      "min": _round(clean.min()),
      "max": _round(clean.max()),
      "outliers": outliers,
  }


def _categorical_stats(series: pd.Series, column: str, target: str | None) -> dict[str, Any]:
  cleaned = series.dropna()
  mode = None
  if not cleaned.empty:
    mode = str(cleaned.astype(str).mode().iloc[0])
  return {
      "name": column,
      "role": "target" if column == target else "feature",
      "unique": int(series.nunique(dropna=True)),
      "mode": mode,
      "missing": int(series.isna().sum()),
  }


def _round(value: Any) -> float | None:
  if value is None or pd.isna(value):
    return None
  return round(float(value), 4)


def analyze_target(frame: pd.DataFrame, target: str) -> dict[str, Any]:
  if target not in frame.columns:
    raise ValueError(f"Target column {target!r} is not in the dataset.")
  series = frame[target]
  if pd.api.types.is_numeric_dtype(series) and int(series.nunique(dropna=True)) > 12:
    raise ValueError(
        "This runtime trains classifiers. "
        f"{target!r} looks continuous; choose a categorical target."
    )
  counts = series.astype(str).value_counts(dropna=True).to_dict()
  return {
      "target": target,
      "rows": int(series.notna().sum()),
      "classes": {str(key): int(value) for key, value in counts.items()},
      "missing_target": int(series.isna().sum()),
  }
