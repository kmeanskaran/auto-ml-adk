"""ADK tools owned by the researcher."""

from __future__ import annotations

from typing import Any

import pandas as pd
from google.adk.tools.tool_context import ToolContext

from team.artifacts import ArtifactSaver
from team.phase import dataset_fingerprint
from team.phase import experiment_dir
from team.phase import run_phase
from team.researcher.brief import business_brief
from team.researcher.brief import research_brief
from team.researcher.brief import statistical_markdown
from team.researcher.profile import analyze_target as target_profile
from team.researcher.profile import inspect_schema as schema_profile
from team.researcher.profile import profile_quality as quality_profile


def inspect_schema(tool_context: ToolContext) -> dict[str, Any]:
  """Inspect column names, types, and how many rows the dataset has."""
  return run_phase(
      tool_context,
      "inspect_schema",
      fingerprint=dataset_fingerprint(tool_context),
      compute=lambda frame, record: schema_profile(frame),
  )


def profile_quality(tool_context: ToolContext) -> dict[str, Any]:
  """Profile missing values, distributions, outliers, and categorical columns."""
  return run_phase(
      tool_context,
      "profile_quality",
      fingerprint=dataset_fingerprint(tool_context),
      compute=lambda frame, record: quality_profile(
          frame, (record.get("objective") or {}).get("target_column")
      ),
  )


def analyze_target(tool_context: ToolContext) -> dict[str, Any]:
  """Count classes for the objective's target column."""
  return run_phase(
      tool_context,
      "analyze_target",
      fingerprint=dataset_fingerprint(tool_context),
      compute=lambda frame, record: target_profile(
          frame, record["objective"]["target_column"]
      ),
  )


def write_business_brief(tool_context: ToolContext) -> dict[str, Any]:
  """Connect the business wording to the columns that are actually in the file."""
  return run_phase(
      tool_context,
      "write_business_brief",
      fingerprint=dataset_fingerprint(tool_context) + ":business",
      compute=lambda frame, record: business_brief(frame, record),
  )


def compile_research_brief(tool_context: ToolContext) -> dict[str, Any]:
  """Write the statistical report from schema, quality, target, and the business brief."""

  def compute(frame: pd.DataFrame, record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    report = research_brief(frame, record)
    path = ArtifactSaver(experiment_dir(tool_context)).save_statistical_report(
        statistical_markdown(report)
    )
    report["path"] = str(path)
    stages = dict(record.get("stages") or {})
    stages["researcher"] = "done"
    tool_context.state["research_summary"] = (
        f"rows={report['rows']} missing={report['missing_cells']}"
    )
    stored = dict(report)
    stored.pop("numeric_summary", None)
    stored["numeric_columns"] = list((report.get("numeric_summary") or {}).keys())
    return stored, {"stages": stages}

  return run_phase(
      tool_context,
      "compile_research_brief",
      fingerprint=dataset_fingerprint(tool_context) + ":research",
      compute=compute,
  )
