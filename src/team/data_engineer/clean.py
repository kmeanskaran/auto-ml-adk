"""Drop duplicate ids, drop unlabeled rows, and fill remaining gaps."""

from __future__ import annotations

from typing import Any

import pandas as pd

from team.tables import id_columns


def clean_frame(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict[str, Any]]:
  """Drop duplicate ids and fill gaps. The saved model still imputes on its own."""
  if target not in frame.columns:
    raise ValueError(f"Target column {target!r} is not in the dataset.")
  report: dict[str, Any] = {
      "rows_before": int(len(frame)),
      "missing_before": int(frame.isna().sum().sum()),
  }
  cleaned = frame.copy()
  ids = id_columns(cleaned)
  if ids:
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=ids)
    report["duplicate_rows_removed"] = int(before - len(cleaned))
  else:
    report["duplicate_rows_removed"] = 0

  before_target = len(cleaned)
  cleaned = cleaned.dropna(subset=[target])
  report["rows_missing_target_removed"] = int(before_target - len(cleaned))

  fills: dict[str, Any] = {}
  for column in cleaned.columns:
    if column == target or int(cleaned[column].isna().sum()) == 0:
      continue
    if pd.api.types.is_numeric_dtype(cleaned[column]):
      fill = cleaned[column].median()
      fill = 0 if pd.isna(fill) else float(fill)
    else:
      mode = cleaned[column].mode(dropna=True)
      fill = "unknown" if mode.empty else mode.iloc[0]
    fills[column] = fill
    cleaned[column] = cleaned[column].fillna(fill)
  report["fills"] = fills
  report["rows_after"] = int(len(cleaned))
  report["missing_after"] = int(cleaned.isna().sum().sum())
  return cleaned, report
