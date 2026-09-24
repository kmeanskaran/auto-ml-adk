"""Turn a finished experiment record into a system design the human can edit."""

from __future__ import annotations

from typing import Any

COMPONENTS = [
    ("TechnicalSpec", "product_manager", "frame_technical_spec", "Turn the business problem into a modeling task."),
    ("BusinessBrief", "business_analyst", "write_business_brief", "Tie columns to the business question."),
    ("StatisticalReport", "research_lead", "compile_research_brief", "Schema, quality, balance, and numeric summaries."),
    ("DataContract", "schema_agent", "inspect_schema", "Column names, types, and row count."),
    ("QualityProfile", "quality_agent", "profile_quality", "Missing cells, constants, duplicate ids."),
    ("TargetAnalysis", "target_agent", "analyze_target", "Class balance for the objective."),
    ("CleaningTransform", "data_engineer", "clean_dataset", "Drop duplicate ids and fill gaps."),
    ("FeaturePipeline", "feature_agent", "create_features", "Strategy-level feature spec."),
    ("ModelSelection", "model_selector", "select_models", "Choose the candidate families."),
    ("CandidateTrainer", "training_agent", "run_training", "Parallel candidate fit, keep best F1."),
    ("EvaluationGate", "evaluation_agent", "evaluate_model", "Compare the metric with the target."),
    ("ModelMetadata", "metadata_agent", "write_model_metadata", "Artifact, features, threshold, and metrics."),
    ("ServingContract", "serving_agent", "publish_serving_contract", "FastAPI route for the saved model."),
    ("PredictionService", "fastapi", "POST /experiments/{id}/predict", "Load model.joblib and score rows."),
]


def render_design(record: dict[str, Any]) -> str:
  objective = record.get("objective") or {}
  best = record.get("best") or {}
  phases = record.get("phases") or {}
  history = record.get("history") or []
  plans = record.get("plans") or []
  lines = [
      f"# Solution design: {record.get('id')}",
      "",
      "## Problem",
      "",
      objective.get("text") or "Train a classifier.",
      "",
      f"- Target column: `{objective.get('target_column')}`",
      f"- Metric: `{objective.get('metric')}` >= {objective.get('target')}",
      f"- Identity: `{record.get('identity')}`",
      f"- Status: `{record.get('status')}`",
      f"- Target met: `{record.get('target_met')}`",
      "",
      "## Technical specification",
      "",
  ]
  tech = (phases.get("frame_technical_spec") or {}).get("result") or {}
  if tech:
    lines.append(tech.get("technical_problem") or "")
    lines.append("")
    lines.append(f"- Task: `{tech.get('task_type')}`")
    lines.append(f"- Target: `{tech.get('target_column')}`")
  else:
    lines.append("The product manager has not framed the task.")
  business = (phases.get("write_business_brief") or {}).get("result") or {}
  if business.get("understanding"):
    lines.extend(["", business["understanding"]])
  research = (phases.get("compile_research_brief") or {}).get("result") or {}
  lines.extend(["", "## Statistical report", ""])
  if research:
    lines.append(f"Rows: {research.get('rows')}. Columns: {research.get('column_count')}.")
    lines.append(f"Class balance: `{research.get('class_balance')}`.")
    lines.append(f"Missing cells: {research.get('missing_cells')}.")
    lines.append(f"IQR outlier cells: {research.get('outlier_cells')}.")
    if research.get("categorical_summary"):
      lines.append(research["categorical_summary"])
    if research.get("sme_note"):
      lines.append(f"SME note: {research['sme_note']}")
    if research.get("path"):
      lines.append(f"Full report: `{research.get('path')}`.")
  else:
    lines.append("No statistical report yet.")
  selection = (phases.get("select_models") or {}).get("result") or {}
  lines.extend(["", "## Model selection", ""])
  if selection:
    lines.append(selection.get("decision") or "")
    for item in selection.get("options") or []:
      lines.append(f"- `{item.get('id')}` — {item.get('reason')}")
    chosen = record.get("chosen_models") or []
    if chosen:
      lines.append(f"SME picked: `{chosen[0]}`.")
  else:
    lines.append("No model family has been chosen.")

  lines.extend(
      [
          "",
          "## ML components",
          "",
          "| Component | Owner | Tool | Responsibility |",
          "| --- | --- | --- | --- |",
      ]
  )
  for name, owner, tool, duty in COMPONENTS:
    lines.append(f"| {name} | {owner} | `{tool}` | {duty} |")

  lines.extend(["", "## Data contract", ""])
  schema = (phases.get("inspect_schema") or {}).get("result") or {}
  if schema:
    lines.append(f"{schema.get('rows')} rows.")
    lines.append("")
    for column in schema.get("columns") or []:
      lines.append(
          f"- `{column['name']}` {column['dtype']}, "
          f"unique={column['unique']}, missing={column['missing']}"
      )
  else:
    lines.append("Schema has not been inspected.")

  quality = (phases.get("profile_quality") or {}).get("result") or {}
  cleaning = (phases.get("clean_dataset") or {}).get("result") or {}
  lines.extend(["", "## Quality and cleaning", ""])
  if quality:
    lines.append(
        f"Missing cells before cleaning: {quality.get('missing_cells')}. "
        f"Duplicate id rows: {quality.get('duplicate_id_rows')}. "
        f"IQR outlier cells: {quality.get('outlier_cells', 0)}."
    )
    if quality.get("categorical_summary"):
      lines.append(quality["categorical_summary"])
    for item in quality.get("spread") or []:
      lines.append(
          f"- `{item.get('name')}` mean {item.get('mean')}, "
          f"median {item.get('median')}, mode {item.get('mode')}, std {item.get('std')}"
      )
  if cleaning:
    lines.append(
        f"Rows {cleaning.get('rows_before')} -> {cleaning.get('rows_after')}. "
        f"Missing cells after cleaning: {cleaning.get('missing_after')}."
    )
    fills = cleaning.get("fills") or {}
    if fills:
      lines.append("")
      lines.append("Fills:")
      for column, value in fills.items():
        lines.append(f"- `{column}` <- {value}")

  target = (phases.get("analyze_target") or {}).get("result") or {}
  lines.extend(["", "## Target", ""])
  if target:
    lines.append(f"Classes: `{target.get('classes')}`.")

  feature = (phases.get("create_features") or {}).get("result") or {}
  lines.extend(["", "## Feature pipeline", ""])
  if feature:
    lines.append(f"Strategy level: {feature.get('strategy_level')}.")
    lines.append(f"Numeric: `{feature.get('numeric')}`.")
    lines.append(f"Categorical: `{feature.get('categorical')}`.")
    extra = feature.get("engineered") or []
    if extra:
      lines.append(f"Engineered: `{extra}`.")

  lines.extend(["", "## Candidates", ""])
  if history:
    lines.append("| Iteration | Strategy | Model | F1 | Precision | Recall |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for trial in history:
      metrics = trial.get("metrics") or {}
      lines.append(
          "| {iteration} | {strategy} | {model} | {f1:.3f} | {precision:.3f} | {recall:.3f} |".format(
              iteration=trial.get("iteration"),
              strategy=trial.get("strategy_level"),
              model=trial.get("model"),
              f1=float(metrics.get("f1") or 0),
              precision=float(metrics.get("precision") or 0),
              recall=float(metrics.get("recall") or 0),
          )
      )
  else:
    lines.append("No trials yet.")

  metadata = (phases.get("write_model_metadata") or {}).get("result") or {}
  lines.extend(["", "## Model metadata", ""])
  if metadata:
    lines.append(f"Algorithm: `{metadata.get('algorithm')}`.")
    lines.append(f"Positive label: `{metadata.get('positive_label')}`.")
    lines.append(f"Decision threshold: `{metadata.get('decision_threshold')}`.")
    lines.append(f"Features: `{metadata.get('feature_columns')}`.")
    metrics = metadata.get("metrics") or {}
    if metrics:
      lines.append(
          f"F1 {float(metrics.get('f1') or 0):.3f}, "
          f"precision {float(metrics.get('precision') or 0):.3f}, "
          f"recall {float(metrics.get('recall') or 0):.3f}."
      )
  else:
    lines.append("Metadata is written after a model is chosen.")

  lines.extend(["", "## Chosen model", ""])
  if best:
    metrics = best.get("metrics") or {}
    lines.append(
        f"`{best.get('model')}` at strategy {best.get('strategy_level')} "
        f"with F1 {float(metrics.get('f1') or 0):.3f}."
    )
    lines.append(f"Artifact: `{best.get('path')}`.")
  else:
    lines.append("No model has been selected.")

  lines.extend(["", "## Improvement notes", ""])
  if plans:
    for plan in plans:
      lines.append(f"- Level {plan.get('strategy_level')}: {plan.get('plan')}")
  else:
    lines.append("The first strategy met the target, so the loop stopped.")
  for note in record.get("notes") or []:
    lines.append(f"- Human note: {note}")

  lines.extend(
      [
          "",
          "## Serving",
          "",
          "The ML engineer publishes the artifact. FastAPI serves it.",
          "",
          "```",
          "POST /experiments/{id}/predict",
          '{ "rows": [ { "<feature>": <value> } ] }',
          "```",
          "",
          "Each row is the raw cleaned schema (ids optional). The artifact applies",
          "the same engineering and preprocessing it was fit with. The response",
          "carries a label and the positive-class probability.",
          "",
          "Iterate with `POST /experiments/{id}/iterate` after reading this file.",
          "Training stays behind the harness approval gate. Checkpoints live in",
          "`checkpoints/` and the append-only tool trace is `trace.jsonl`.",
          "",
          "## Code execution",
          "",
          "`run_analysis_snippet` is a governed tool: an allowlist Python sandbox",
          "over the dataframe. It is blocked unless the experiment policy removes",
          "it from `blocked_tools`, and it still requires human confirmation.",
          "",
          "## Trajectory",
          "",
          " -> ".join(record.get("trajectory") or []) or "(empty)",
          "",
      ]
  )
  return "\n".join(lines) + "\n"
