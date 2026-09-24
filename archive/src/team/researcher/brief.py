"""Business reading and the statistical report assembled from research tools."""

from __future__ import annotations

from typing import Any

import pandas as pd

from team.phase import latest_sme_note
from team.researcher.profile import column_profile


def business_brief(frame: pd.DataFrame, record: dict[str, Any]) -> dict[str, Any]:
  objective = record.get("objective") or {}
  target = objective.get("target_column")
  features = [column for column in frame.columns if column != target]
  return {
      "business_problem": objective.get("text") or "",
      "target_column": target,
      "feature_columns": features,
      "understanding": (
          f"The business asks: {objective.get('text') or 'predict the target'}. "
          f"The measurable outcome is `{target}`. "
          f"The other {len(features)} columns are the inputs the model can use."
      ),
  }


def research_brief(frame: pd.DataFrame, record: dict[str, Any]) -> dict[str, Any]:
  phases = record.get("phases") or {}
  schema = (phases.get("inspect_schema") or {}).get("result") or {}
  quality = (phases.get("profile_quality") or {}).get("result") or {}
  target = (phases.get("analyze_target") or {}).get("result") or {}
  business = (phases.get("write_business_brief") or {}).get("result") or {}
  tech = (phases.get("frame_technical_spec") or {}).get("result") or {}
  target_name = (record.get("objective") or {}).get("target_column")
  profile = column_profile(frame, target_name)
  return {
      "business_problem": (record.get("objective") or {}).get("text"),
      "technical_problem": tech.get("technical_problem"),
      "understanding": business.get("understanding"),
      "rows": int(len(frame)),
      "column_count": int(frame.shape[1]),
      "missing_cells": quality.get("missing_cells", int(frame.isna().sum().sum())),
      "class_balance": target.get("classes") or schema.get("columns"),
      "categorical_summary": profile["categorical_summary"],
      "categorical_columns": profile["categorical_columns"],
      "outlier_cells": profile["outlier_cells"],
      "sme_note": latest_sme_note(record, "stats"),
      "numeric_summary": profile["numeric_summary"],
  }


def statistical_markdown(report: dict[str, Any]) -> str:
  lines = [
      "# Statistical report",
      "",
      report.get("understanding") or "",
      "",
      f"Rows: {report.get('rows')}",
      f"Columns: {report.get('column_count')}",
      f"Missing cells: {report.get('missing_cells')}",
      f"IQR outlier cells: {report.get('outlier_cells')}",
      f"Class balance: {report.get('class_balance')}",
      "",
      report.get("categorical_summary") or "",
  ]
  note = report.get("sme_note")
  if note:
    lines.extend(["", f"SME note: {note}", "The categorical reading above answers that note."])
  lines.extend(
      [
          "",
          "## Categorical columns",
          "",
      ]
  )
  cats = report.get("categorical_columns") or []
  if not cats:
    lines.append("None.")
  for column in cats:
    lines.append(
        f"- `{column.get('name')}` role={column.get('role')}, "
        f"unique={column.get('unique')}, mode={column.get('mode')}, "
        f"missing={column.get('missing')}"
    )
  lines.extend(
      [
          "",
          "## Numeric summaries",
          "",
          "| Column | Missing | Mean | Median | Mode | Std | Min | Max | Outliers |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
      ]
  )
  for name, stats in (report.get("numeric_summary") or {}).items():
    lines.append(
        f"| {name} | {stats.get('missing')} | {stats.get('mean')} | "
        f"{stats.get('median')} | {stats.get('mode')} | {stats.get('std')} | "
        f"{stats.get('min')} | {stats.get('max')} | {stats.get('outliers')} |"
    )
  return "\n".join(lines) + "\n"
