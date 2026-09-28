"""The only actions agents can take: read, write, run, check, and report.

Agents decide what to do; these tools enforce who may touch what. The pipeline
order and the human reviews live in app/pipeline.py, not here.
"""

from __future__ import annotations

import json
import math
import shlex
import sys
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from google.adk.tools.tool_context import ToolContext

from app.harness import contracts
from app.harness.environment import ProjectEnvironment, screen
from app.harness.project import (
    ANALYSIS,
    append_jsonl,
    current,
    folder,
    read_json,
    resolve,
    write_json,
)

RUN_TIMEOUT_SECONDS = 900
MAX_READ_CHARS = 12_000
OUTPUT_HEAD, OUTPUT_TAIL = 1_500, 4_500
MAX_FINDINGS, MAX_FINDING_CHARS = 3, 160
MAX_SUMMARY_FINDINGS, MAX_RECOMMENDATIONS = 5, 5
MAX_CATEGORIES, MAX_SERIES = 40, 3
CHART_TYPES = ("bar", "hbar", "line")

# What each agent may write, relative to its folder. The model engineer cannot touch
# data.py or features.py: those are approved and frozen in the feature store.
OWNS = {
    "analyst_profile": ("checks/analyst_*.py", "notes/analyst.md"),
    "engineer_features": (
        "src/data.py",
        "src/features.py",
        "checks/*.py",
        "notes/engineer.md",
    ),
    "skeptic_features": ("checks/skeptic_*.py", "notes/skeptic.md"),
    "engineer_model": ("src/train.py", "checks/*.py", "notes/engineer.md"),
    "skeptic_model": ("checks/skeptic_*.py", "notes/skeptic.md"),
    "analyst": ("*.py", "charts/*.json", "notes.md"),
}


def root(tool_context: ToolContext) -> Path:
    """The analyst works in its own folder; pipeline agents in the current run's."""
    return folder(ANALYSIS) if tool_context.agent_name == "analyst" else current()


def stage_of(tool_context: ToolContext) -> str:
    return tool_context.agent_name.rsplit("_", 1)[-1]


def list_files(folder_name: str, tool_context: ToolContext) -> dict[str, Any]:
    """Lists files in a folder of the project, with sizes.

    Args:
      folder_name: Folder relative to the project root, e.g. "src" or "reports". "." for the root.

    Returns:
      dict with each file's path and size in bytes.
    """
    base = root(tool_context)
    try:
        target = resolve(base, folder_name)
    except ValueError as exc:
        return _error(str(exc))
    files = [
        {"path": str(p.relative_to(base)), "bytes": p.stat().st_size}
        for p in sorted(target.rglob("*"))
        if p.is_file() and ".home" not in p.parts
    ]
    return {"status": "ok", "files": files[:200]}


def read_file(path: str, tool_context: ToolContext) -> dict[str, Any]:
    """Reads a text file from the project.

    Args:
      path: File path relative to the project root, e.g. "reports/data.json".

    Returns:
      dict with the file text, truncated when long.
    """
    try:
        target = resolve(root(tool_context), path)
    except ValueError as exc:
        return _error(str(exc))
    if not target.is_file():
        return _error(f"No file {path!r}.")
    text = target.read_text(encoding="utf-8", errors="replace")
    return {
        "status": "ok",
        "text": text[:MAX_READ_CHARS],
        "truncated": len(text) > MAX_READ_CHARS,
    }


def write_file(path: str, content: str, tool_context: ToolContext) -> dict[str, Any]:
    """Creates or replaces a file you own.

    Args:
      path: File path relative to the project root, e.g. "src/data.py".
      content: The full content of the file.

    Returns:
      dict with the path written.
    """
    base = root(tool_context)
    try:
        target = resolve(base, path)
    except ValueError as exc:
        return _error(str(exc))
    relative = str(target.relative_to(base.resolve()))
    owned = OWNS.get(tool_context.agent_name, ())
    if not any(fnmatch(relative, pattern) for pattern in owned):
        return _error(f"You may write only {', '.join(owned)}.")
    if target.suffix == ".py" and (blocked := screen(content)):
        return _error(f"Not allowed in project code: {', '.join(blocked)}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"status": "ok", "path": relative}


async def run_python(script: str, tool_context: ToolContext) -> dict[str, Any]:
    """Runs a Python script from the project. The raw data is in the DATA_DIR folder.

    Scripts should print a short JSON summary as their last line.

    Args:
      script: Script path relative to the project root, e.g. "src/data.py".

    Returns:
      dict with exit code, seconds, and the start and end of stdout and stderr.
    """
    base = root(tool_context)
    try:
        target = resolve(base, script)
    except ValueError as exc:
        return _error(str(exc))
    if target.suffix != ".py" or not target.is_file():
        return _error(f"No Python script {script!r}.")
    if blocked := screen(target.read_text(encoding="utf-8", errors="replace")):
        return _error(f"Blocked: {', '.join(blocked)}.")
    relative = str(target.relative_to(base.resolve()))
    started = time.time()
    result = await ProjectEnvironment(base, _env(tool_context)).execute(
        f"{shlex.quote(sys.executable)} {shlex.quote(relative)}",
        timeout=RUN_TIMEOUT_SECONDS,
    )
    seconds = round(time.time() - started, 1)
    append_jsonl(
        base / "logs" / "runs.jsonl",
        {
            "agent": tool_context.agent_name,
            "script": relative,
            "exit_code": result.exit_code,
            "seconds": seconds,
        },
    )
    return {
        "status": "ok" if result.exit_code == 0 else "failed",
        "exit_code": result.exit_code,
        "seconds": seconds,
        "stdout": _clip(result.stdout),
        "stderr": _clip(result.stderr),
    }


async def check_stage(tool_context: ToolContext) -> dict[str, Any]:
    """Checks that your stage's hand-over is complete. Run it before you submit your receipt.

    Returns:
      dict with complete (true/false), the problems found, and the required format.
    """
    stage = stage_of(tool_context)
    issues = await contracts.problems(current(), stage)
    return {
        "status": "ok",
        "complete": not issues,
        "problems": issues,
        "required_format": contracts.FORMATS[stage] if issues else "",
    }


def submit_receipt(findings: list[str], tool_context: ToolContext) -> dict[str, Any]:
    """Hands your stage over with at most three one-line findings the human will read.

    Args:
      findings: Up to three short lines, each a decision or result with its number,
        e.g. "Split by arrival date: train to Jun 2016, test Oct-Dec 2016".

    Returns:
      dict with the stored receipt.
    """
    stage = stage_of(tool_context)
    if not findings or len(findings) > MAX_FINDINGS:
        return _error(f"Give between 1 and {MAX_FINDINGS} findings.")
    receipt = {"stage": stage, "findings": _lines(findings)}
    write_json(current() / "receipts" / f"{stage}.json", receipt)
    return {"status": "ok", "receipt": receipt}


def submit_summary(
    headline: str, findings: list[str], tool_context: ToolContext
) -> dict[str, Any]:
    """Hands the human your data summary: one headline and up to five findings.

    Args:
      headline: One sentence on what this dataset is, with its size and outcome rate.
      findings: Up to five one-line findings, each with the number you measured, the
        ones that matter most for building a model first.

    Returns:
      dict with the stored summary.
    """
    del tool_context
    if not headline.strip():
        return _error("Give a one-sentence headline.")
    if not findings or len(findings) > MAX_SUMMARY_FINDINGS:
        return _error(f"Give between 1 and {MAX_SUMMARY_FINDINGS} findings.")
    summary = {"headline": headline.strip()[:240], "findings": _lines(findings)}
    write_json(current() / "reports" / "summary.json", summary)
    return {"status": "ok", "summary": summary}


def propose_plan(
    models: list[str], metric: str, reason: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Proposes the training plan the human confirms before any model is trained.

    Args:
      models: Model keys to train and compare, from: logistic_regression, random_forest,
        extra_trees, gradient_boosting, hist_gradient_boosting, xgboost.
      metric: The metric that picks the best candidate and compares it with production,
        from: roc_auc, pr_auc, f1, precision, recall, accuracy, log_loss, brier,
        cost_per_1000.
      reason: One line on why these models and this metric fit this data.

    Returns:
      dict with the stored proposal.
    """
    del tool_context
    models = list(dict.fromkeys(m.strip() for m in models))
    if issues := contracts.plan_problems(models, metric.strip()):
        return _error("; ".join(issues))
    proposal = {
        "models": models,
        "metric": metric.strip(),
        "reason": reason.strip()[:240],
    }
    write_json(current() / "reports" / "plan_proposal.json", proposal)
    return {"status": "ok", "proposal": proposal}


def submit_review(
    verdict: str,
    findings: list[str],
    recommendations: list[str],
    tool_context: ToolContext,
    recommended_model: str = "",
) -> dict[str, Any]:
    """Hands your review to the human: findings, then recommendations they can apply.

    Args:
      verdict: "pass" if the stage is sound, "concerns" if something should change.
      findings: Up to three one-line findings, each with the number you measured.
      recommendations: Up to five one-line instructions for the engineer, most important
        first, e.g. "Remove X: it is set after the outcome (AUC 0.99 alone)". Empty if
        nothing should change.
      recommended_model: Model stage only: the candidate key you would promote, or "".

    Returns:
      dict with the stored review.
    """
    stage = stage_of(tool_context)
    if verdict not in ("pass", "concerns"):
        return _error('verdict must be "pass" or "concerns".')
    if not findings or len(findings) > MAX_FINDINGS:
        return _error(f"Give between 1 and {MAX_FINDINGS} findings.")
    if len(recommendations) > MAX_RECOMMENDATIONS:
        return _error(f"Give at most {MAX_RECOMMENDATIONS} recommendations.")
    if recommended_model:
        candidates = (read_json(current() / "reports" / "evaluation.json") or {}).get(
            "candidates", {}
        )
        if recommended_model not in candidates:
            return _error(
                f"recommended_model must be one of {sorted(candidates)} or empty."
            )
    review = {
        "stage": stage,
        "verdict": verdict,
        "findings": _lines(findings),
        "recommendations": _lines(r for r in recommendations if r.strip()),
        "recommended_model": recommended_model,
    }
    write_json(current() / "reviews" / f"{stage}.json", review)
    append_jsonl(current() / "reviews" / "history.jsonl", review)
    return {"status": "ok", "review": review}


def show_chart(spec_path: str, tool_context: ToolContext) -> dict[str, Any]:
    """Shows the human a chart whose data your script wrote to a JSON file.

    The file holds {"type": "bar" | "hbar" | "line", "title": str, "x_label": str,
    "y_label": str, "categories": [label, ...], "series": [{"name": str,
    "values": [number, ...]}]}: at most 40 categories and 3 series, one value per
    category. Use "hbar" for many or long category names, "line" for values over time.

    Args:
      spec_path: The JSON file your script wrote, e.g. "charts/cancel_by_month.json".

    Returns:
      dict with the chart as it will be shown, or what to fix.
    """
    try:
        target = resolve(root(tool_context), spec_path)
        spec = json.loads(target.read_text(encoding="utf-8"))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return _error(f"Cannot read {spec_path!r}: {exc}")
    if issues := chart_problems(spec):
        return _error("; ".join(issues))
    chart = {
        "type": spec["type"],
        "title": str(spec.get("title", ""))[:120],
        "x_label": str(spec.get("x_label", ""))[:60],
        "y_label": str(spec.get("y_label", ""))[:60],
        "categories": [str(c)[:40] for c in spec["categories"]],
        "series": [
            {"name": str(s.get("name", ""))[:40], "values": s["values"]}
            for s in spec["series"]
        ],
    }
    return {"status": "ok", "chart": chart}


def chart_problems(spec: Any) -> list[str]:
    if not isinstance(spec, dict):
        return ["the chart file must hold a JSON object"]
    issues = []
    if spec.get("type") not in CHART_TYPES:
        issues.append(f"type must be one of {CHART_TYPES}")
    categories, series = spec.get("categories"), spec.get("series")
    if not isinstance(categories, list) or not 1 <= len(categories) <= MAX_CATEGORIES:
        issues.append(f"categories must be a list of 1 to {MAX_CATEGORIES} labels")
    if not isinstance(series, list) or not 1 <= len(series) <= MAX_SERIES:
        issues.append(f"series must be a list of 1 to {MAX_SERIES} series")
    if issues:
        return issues
    for s in series:
        values = s.get("values") if isinstance(s, dict) else None
        if not isinstance(values, list) or len(values) != len(categories):
            issues.append("each series needs one value per category")
        elif not all(
            v is None
            or (
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
            )
            for v in values
        ):
            issues.append("values must be finite numbers or null")
    return issues


def tool_error_as_result(
    tool: Any, args: dict[str, Any], tool_context: ToolContext, error: Exception
) -> dict[str, Any]:
    """A failed or unknown tool call becomes a result the model can correct, not a crash."""
    del args, tool_context
    return _error(
        f"Tool {getattr(tool, 'name', '?')!r} failed: {str(error).split('Possible causes')[0].strip()}"
    )


def _env(tool_context: ToolContext) -> dict[str, str]:
    """Paths an agent's scripts may read besides DATA_DIR."""
    if tool_context.agent_name == "analyst":
        return {"RUN_DIR": str(current())}
    ref = read_json(current() / "reports" / "feature_ref.json")
    return {"FEATURE_DIR": ref["path"]} if ref else {}


def _lines(lines: Any) -> list[str]:
    return [line.strip()[:MAX_FINDING_CHARS] for line in lines]


def _clip(text: str) -> str:
    if len(text) <= OUTPUT_HEAD + OUTPUT_TAIL:
        return text
    return f"{text[:OUTPUT_HEAD]}\n…[{len(text) - OUTPUT_HEAD - OUTPUT_TAIL} chars cut]…\n{text[-OUTPUT_TAIL:]}"


def _error(message: str) -> dict[str, Any]:
    return {"status": "error", "message": message}
