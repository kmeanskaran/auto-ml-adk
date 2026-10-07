"""The only actions agents can take: read, write, run, check, and report.

Agents decide what to do; these tools enforce who may touch what. The pipeline
order and the human reviews live in app/pipeline.py, not here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
import shutil
import sys
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from app import settings
from app.harness import contracts, feature_store, project
from app.harness.environment import ProjectEnvironment, screen
from app.harness.project import (
    append_jsonl,
    current,
    home,
    read_json,
    resolve,
    write_json,
)

MAX_READ_CHARS = 12_000
OUTPUT_HEAD, OUTPUT_TAIL = 1_500, 4_500
MAX_FINDINGS, MAX_FINDING_CHARS, MAX_RECOMMENDATION_CHARS = 3, 140, 110
MAX_SUMMARY_FINDINGS, MAX_RECOMMENDATIONS, MAX_SUMMARY_CHARTS = 5, 3, 3
MAX_CATEGORIES, MAX_SERIES = 40, 3
CHART_TYPES = ("bar", "hbar", "line", "stacked")
MAX_MATCHES, SEARCH_SUFFIXES = 60, (".py", ".json", ".md", ".txt", ".csv", ".log")
# Tool calls per agent turn (config limits.tool_budget): past two thirds each result
# carries a reminder to hand over, past the budget only check_stage and submit tools run.
FINISHING = {
    "check_stage",
    "submit_receipt",
    "submit_summary",
    "submit_review",
    "propose_plan",
    "show_chart",
}
# A turn's history is re-sent with every model call. Above this size, copies of
# files and run outputs that a later read, write or run supersedes are elided; if it
# is still above it, script outputs older than the last few keep only their summary.
COMPACT_ABOVE_CHARS = 50_000  # the default; config limits.compact_above_chars
KEEP_RECENT_OUTPUTS = 4  # the newest script outputs always stay whole
SUMMARY_CHARS = 600  # what an older output keeps: its last JSON line, or its tail

# What each agent may write, relative to its folder. The model engineer cannot touch
# data.py or features.py: those are approved and frozen in the feature store.
OWNS = {
    "analyst_profile": ("checks/analyst_*.py", "charts/*.json", "notes/analyst.md"),
    "engineer_features": (
        "src/data.py",
        "src/features.py",
        "checks/*.py",
        "notes/engineer.md",
    ),
    "skeptic_features": ("checks/skeptic_*.py", "notes/skeptic.md"),
    "skeptic_model": ("checks/skeptic_*.py", "notes/skeptic.md"),
    "engineer_model": ("src/train.py", "checks/*.py", "notes/engineer.md"),
    "analyst": ("*.py", "charts/*.json", "notes.md"),
}


def root(tool_context: ToolContext) -> Path:
    """The analyst works in its own folder; pipeline agents in the current run's."""
    return home(tool_context.agent_name)


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


def search(
    pattern: str, tool_context: ToolContext, folder_name: str = "."
) -> dict[str, Any]:
    """Finds lines matching a regular expression in the project's text files
    (.py, .json, .md, .txt, .csv, .log), instead of reading whole files to look.

    Args:
      pattern: Python regular expression, e.g. "def build" or "Credit_History".
      folder_name: Folder or file to search, relative to the project root. "." for all.

    Returns:
      dict with up to 60 matches, each with path, line number and the line.
    """
    base = root(tool_context)
    try:
        target = resolve(base, folder_name)
        regex = re.compile(pattern)
    except (ValueError, re.error) as exc:
        return _error(str(exc))
    matches = []
    files = [target] if target.is_file() else sorted(target.rglob("*"))
    for p in files:
        if not p.is_file() or p.suffix not in SEARCH_SUFFIXES:
            continue
        relative = p.relative_to(base)
        if relative.parts[0] in (".home", "logs"):
            continue
        for number, line in enumerate(
            p.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if regex.search(line):
                matches.append(
                    {"path": str(relative), "line": number, "text": line.strip()[:200]}
                )
                if len(matches) >= MAX_MATCHES:
                    return {"status": "ok", "matches": matches, "more": True}
    return {"status": "ok", "matches": matches, "more": False}


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
    relative = str(target.relative_to(root(tool_context).resolve()))
    seen = _turn(tool_context).seen
    if seen.get(relative) == _sha(text):
        seen.pop(relative)  # asked again right after: send it rather than loop
        return {
            "status": "ok",
            "unchanged": True,
            "message": f"{relative} is unchanged since you last read or wrote it in "
            "this task: use that copy (read it once more to get the text again).",
        }
    seen[relative] = _sha(text)
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
    _turn(tool_context).seen[relative] = _sha(content)
    return {"status": "ok", "path": relative}


async def write_and_run(
    path: str, content: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Creates or replaces a Python script you own, then runs it: one call instead of
    write_file followed by run_python.

    Args:
      path: Script path relative to the project root, e.g. "checks/signal.py".
      content: The full content of the script.

    Returns:
      dict with the path written and the run's exit code, seconds, stdout and stderr.
    """
    if not path.endswith(".py"):
        return _error(
            "write_and_run takes a .py script; use write_file for other files."
        )
    written = write_file(path, content, tool_context)
    if written.get("status") != "ok":
        return written
    return {"path": written["path"], **await run_python(written["path"], tool_context)}


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
        return _error(
            f"No Python script {script!r}. run_python takes the path of a script in "
            "the project; to run new code, use write_and_run(path, content)."
        )
    if blocked := screen(target.read_text(encoding="utf-8", errors="replace")):
        return _error(f"Blocked: {', '.join(blocked)}.")
    relative = str(target.relative_to(base.resolve()))
    started = time.time()
    result = await ProjectEnvironment(base, _env(tool_context)).execute(
        f"{shlex.quote(sys.executable)} {shlex.quote(relative)}",
        timeout=settings.load().script_seconds,
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
    if (issues := contracts.unchanged(current(), stage)) is not None:
        return {
            "status": "ok",
            "complete": not issues,
            "problems": issues,
            "unchanged": "Nothing changed since your last check_stage: same result. "
            + (
                "Fix these problems before checking again." if issues else "Submit now."
            ),
        }
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
        e.g. "Stratified split 70/15/15: every part keeps the 31% outcome rate".

    Returns:
      dict with the stored receipt.
    """
    stage = stage_of(tool_context)
    if not findings or len(findings) > MAX_FINDINGS:
        return _error(f"Give between 1 and {MAX_FINDINGS} findings.")
    if issue := _too_long(findings, MAX_FINDING_CHARS, "finding"):
        return _error(issue)
    receipt = {"stage": stage, "findings": _lines(findings)}
    write_json(current() / "receipts" / f"{stage}.json", receipt)
    return {"status": "ok", "receipt": receipt}


def submit_summary(
    headline: str,
    findings: list[str],
    tool_context: ToolContext,
    charts: list[str] | None = None,
) -> dict[str, Any]:
    """Hands the human your data summary: one headline, up to five findings, and the
    charts that show them.

    Args:
      headline: One sentence on what this dataset is, with its size and outcome rate.
      findings: Up to five one-line findings, each with the number you measured, the
        ones that matter most for building a model first.
      charts: Up to three chart files your scripts wrote (same format as show_chart),
        e.g. ["charts/outcome_by_x.json"]: the ones this data calls for, most telling
        first. Empty if no chart adds to the findings.

    Returns:
      dict with the stored summary.
    """
    shown = []
    for path in charts or []:
        chart, issue = _chart(root(tool_context), path)
        if issue:
            return _error(issue)
        shown.append(chart)
    if len(shown) > MAX_SUMMARY_CHARTS:
        return _error(f"Give at most {MAX_SUMMARY_CHARTS} charts.")
    if not headline.strip():
        return _error("Give a one-sentence headline.")
    if not findings or len(findings) > MAX_SUMMARY_FINDINGS:
        return _error(f"Give between 1 and {MAX_SUMMARY_FINDINGS} findings.")
    if issue := _too_long(findings, MAX_FINDING_CHARS, "finding") + _too_long(
        [headline], 200, "headline"
    ):
        return _error(issue)
    summary = {
        "headline": headline.strip(),
        "findings": _lines(findings),
        "charts": shown,
    }
    write_json(current() / "reports" / "summary.json", summary)
    return {"status": "ok", "summary": summary}


def propose_plan(
    models: list[str], metric: str, reason: str, tool_context: ToolContext
) -> dict[str, Any]:
    """Proposes the training plan the human confirms before any model is trained.

    Args:
      models: Model keys to train and compare, from: logistic_regression, decision_tree,
        knn, svm, random_forest, adaboost, extra_trees, gradient_boosting,
        hist_gradient_boosting, xgboost, lightgbm (only those installed are accepted).
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
) -> dict[str, Any]:
    """Hands your review to the human: findings, then recommendations they can apply.

    Args:
      verdict: "pass" if the stage is sound, "concerns" if something should change.
      findings: Up to three one-line findings, each with the number you measured.
      recommendations: Up to three short instructions for the engineer, the most
        impactful first, only those that would change the result: a verb, one
        thing, a short reason, under 110 characters, e.g.
        "Remove `X`: it is filled in after the outcome, so the model would cheat".
        Empty if nothing should change.

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
    if issue := _too_long(findings, MAX_FINDING_CHARS, "finding") + _too_long(
        recommendations, MAX_RECOMMENDATION_CHARS, "recommendation"
    ):
        return _error(issue)
    review = {
        "stage": stage,
        "verdict": verdict,
        "findings": _lines(findings),
        "recommendations": _lines(r for r in recommendations if r.strip()),
    }
    write_json(current() / "reviews" / f"{stage}.json", review)
    append_jsonl(current() / "reviews" / "history.jsonl", review)
    return {"status": "ok", "review": review}


def show_chart(spec_path: str, tool_context: ToolContext) -> dict[str, Any]:
    """Shows the human a chart whose data your script wrote to a JSON file.

    The file holds {"type": "bar" | "hbar" | "line" | "stacked", "title": str,
    "x_label": str, "y_label": str, "categories": [label, ...], "series": [{"name": str,
    "values": [number, ...]}]}: at most 40 categories and 3 series, one value per
    category. Use "hbar" for many or long category names, "line" for values over time,
    "stacked" for shares that add up within each category (e.g. outcome yes / no, in %).

    Args:
      spec_path: The JSON file your script wrote, e.g. "charts/rate_by_area.json".

    Returns:
      dict with the chart as it will be shown, or what to fix.
    """
    chart, issue = _chart(root(tool_context), spec_path)
    return _error(issue) if issue else {"status": "ok", "chart": chart}


def _chart(base: Path, spec_path: str) -> tuple[dict[str, Any], str]:
    """A chart file as it will be shown, or what to fix."""
    try:
        target = resolve(base, spec_path)
        spec = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        written = sorted(
            str(p.relative_to(base)) for p in (base / "charts").glob("*.json")
        )
        return {}, (
            f"No chart file {spec_path!r}. Chart files your scripts wrote: "
            + (", ".join(written[-12:]) if written else "none")
            + ". Pass the path chart() returned."
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        return {}, f"Cannot read {spec_path!r}: {exc}"
    if issues := chart_problems(spec):
        return {}, f"{spec_path}: " + "; ".join(issues)
    return {
        "type": spec["type"],
        "title": str(spec.get("title", ""))[:120],
        "x_label": str(spec.get("x_label", ""))[:60],
        "y_label": str(spec.get("y_label", ""))[:60],
        "categories": [str(c)[:40] for c in spec["categories"]],
        "series": [
            {"name": str(s.get("name", ""))[:40], "values": s["values"]}
            for s in spec["series"]
        ],
    }, ""


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


# --- one agent turn: what it has seen, how many calls it made, a compact history -------


@dataclass
class _Turn:
    seen: dict[str, str] = field(default_factory=dict)  # path -> hash it last saw
    calls: int = 0


_TURNS: dict[str, _Turn] = {}


def _turn_key(context: Any) -> str:
    return f"{getattr(context, 'invocation_id', '')}:{context.agent_name}"


def _turn(context: Any) -> _Turn:
    """The agent's current turn; outside one (a direct call) nothing is remembered."""
    return _TURNS.get(_turn_key(context)) or _Turn()


def start_turn(callback_context: CallbackContext) -> None:
    """before_agent_callback: each activation of an agent starts with a fresh context,
    so it starts with a fresh memory of what it has read and a fresh budget."""
    _TURNS[_turn_key(callback_context)] = _Turn()


def end_turn(callback_context: CallbackContext) -> None:
    """after_agent_callback."""
    _TURNS.pop(_turn_key(callback_context), None)


def budgets(agent: str = "") -> tuple[int, int]:
    """The wrap-up reminder and the hard stop, from config limits.tool_budget (the
    skeptic's from limits.skeptic_tool_budget)."""
    config = settings.load()
    hard = (
        config.skeptic_tool_budget
        if agent.startswith("skeptic")
        else config.tool_budget
    )
    return hard * 2 // 3, hard


def budget(
    tool: Any, args: dict[str, Any], tool_context: ToolContext
) -> dict[str, Any] | None:
    """before_tool_callback: past the hard budget, only finishing tools still run."""
    del args
    turn = _turn(tool_context)
    turn.calls += 1
    _, hard = budgets(tool_context.agent_name)
    if turn.calls > hard and tool.name not in FINISHING:
        return _error(
            f"Tool budget spent ({hard} calls). Hand over now with what you "
            "have: run check_stage, then submit."
        )
    return None


def budget_reminder(
    tool: Any, args: dict[str, Any], tool_context: ToolContext, tool_response: Any
) -> dict[str, Any] | None:
    """after_tool_callback: past the soft budget, every result says how many calls are left."""
    del args
    calls = _turn(tool_context).calls
    soft, hard = budgets(tool_context.agent_name)
    if calls < soft or tool.name in FINISHING or not isinstance(tool_response, dict):
        return None
    return {
        **tool_response,
        "budget": f"{calls} of {hard} tool calls used: wrap up and hand over.",
    }


SUBMITS = {"submit_receipt", "submit_summary", "submit_review"}
HANDED_OVER = "Handed over."


def handed_over(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """before_model_callback: once a submit succeeded, the turn's closing line is
    written here instead of by the model. ADK would otherwise ask the model once more
    only to say it is done: a whole model call nobody reads. The turn still ends with
    a plain final answer, so a paused workflow resumes where it was."""
    del callback_context
    last = (llm_request.contents or [None])[-1]
    for part in (last.parts or []) if last else []:
        response = part.function_response
        if (
            response is not None
            and response.name in SUBMITS
            and (response.response or {}).get("status") == "ok"
        ):
            return LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text=HANDED_OVER)]
                )
            )
    return None


def compact(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
    """before_model_callback: elide what a later step superseded once the turn is long.

    Every model call re-sends the whole turn. An older copy of a file is dead weight
    once the agent has read or written a newer one, and so is an older output of a
    script it has since run again. Those are replaced by a one-line note; the newest
    copy of everything stays. The session itself is never changed.
    """
    del callback_context
    contents = llm_request.contents or []
    limit = settings.load().compact_above_chars
    if sum(_size(c) for c in contents) < limit:
        return None
    calls = {
        part.function_call.id: part.function_call
        for c in contents
        for part in c.parts or []
        if part.function_call is not None and part.function_call.id
    }
    later: set[str] = set()  # "file:<path>" / "run:<script>" seen in a later step
    for index in range(len(contents) - 1, -1, -1):
        parts = list(contents[index].parts or [])
        changed = False
        for k in range(len(parts) - 1, -1, -1):
            part, response, call = (
                parts[k],
                parts[k].function_response,
                parts[k].function_call,
            )
            if response:
                made_by = calls.get(response.id or "")
                if made_by is None:
                    continue
                keys = _superseding(made_by, response.response or {})
                if keys & later:
                    parts[k] = _elided_response(part, response, keys & later)
                    changed = True
                later |= keys
            elif call and call.name in ("write_file", "write_and_run"):
                args = call.args or {}
                key = f"file:{args.get('path', '')}"
                if key in later and len(str(args.get("content", ""))) > 200:
                    elided = {
                        **args,
                        "content": "[elided: a later read or write of this file follows]",
                    }
                    # model_copy keeps the part's thought_signature, which Gemini needs
                    parts[k] = part.model_copy(
                        update={
                            "function_call": call.model_copy(update={"args": elided})
                        }
                    )
                    changed = True
                later.add(key)
        if changed:
            contents[index] = types.Content(role=contents[index].role, parts=parts)
    if sum(_size(c) for c in contents) >= limit:
        _summarise_old_outputs(contents, calls)
    return None


def _summarise_old_outputs(
    contents: list[types.Content], calls: dict[str, types.FunctionCall]
) -> None:
    """Keep only the summary of script outputs older than the last few: scripts print
    a short JSON summary last, which is what later steps build on."""
    seen = 0
    for index in range(len(contents) - 1, -1, -1):
        parts = list(contents[index].parts or [])
        changed = False
        for k in range(len(parts) - 1, -1, -1):
            response = parts[k].function_response
            made_by = calls.get(response.id or "") if response else None
            if made_by is None or made_by.name not in ("run_python", "write_and_run"):
                continue
            seen += 1
            result = dict(response.response or {})
            bulky = {key: str(result.get(key) or "") for key in ("stdout", "stderr")}
            if (
                seen <= KEEP_RECENT_OUTPUTS
                or sum(map(len, bulky.values())) <= SUMMARY_CHARS
            ):
                continue
            result["stdout"] = _summary_of(bulky["stdout"])
            result["stderr"] = (
                bulky["stderr"][-SUMMARY_CHARS // 3 :] if bulky["stderr"] else ""
            )
            result["elided"] = "older output: only its summary is kept"
            parts[k] = parts[k].model_copy(
                update={
                    "function_response": response.model_copy(
                        update={"response": result}
                    )
                }
            )
            changed = True
        if changed:
            contents[index] = types.Content(role=contents[index].role, parts=parts)


def _summary_of(stdout: str) -> str:
    """A script's last JSON line, or the tail of its output."""
    for line in reversed(stdout.strip().splitlines()):
        if line.lstrip().startswith(("{", "[")) and len(line) <= SUMMARY_CHARS:
            return line
    return ("…" + stdout[-SUMMARY_CHARS:]) if len(stdout) > SUMMARY_CHARS else stdout


def _superseding(call: types.FunctionCall, response: dict[str, Any]) -> set[str]:
    """The keys a tool result holds a copy of: a file's text, or a script's output."""
    args = call.args or {}
    if call.name == "read_file" and response.get("text") is not None:
        return {f"file:{args.get('path', '')}"}
    if call.name == "run_python":
        return {f"run:{args.get('script', '')}"}
    if call.name == "write_and_run":
        return {f"run:{args.get('path', '')}"}
    return set()


def _elided_response(
    part: types.Part, response: types.FunctionResponse, keys: set[str]
) -> types.Part:
    kept = dict(response.response or {})
    for bulky in ("text", "stdout", "stderr"):
        if bulky in kept:
            kept[bulky] = "[elided]"
    what = ", ".join(sorted(k.split(":", 1)[1] for k in keys))
    kept["elided"] = f"older copy of {what}: a later step has the current one"
    return part.model_copy(
        update={"function_response": response.model_copy(update={"response": kept})}
    )


def _size(content: types.Content) -> int:
    total = 0
    for part in content.parts or []:
        if part.text:
            total += len(part.text)
        elif part.function_call:
            total += len(json.dumps(part.function_call.args or {}, default=str))
        elif part.function_response:
            total += len(json.dumps(part.function_response.response or {}, default=str))
    return total


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


ANALYSTS = ("analyst", "analyst_profile")
KIT = Path(__file__).with_name("analyst_kit.py")


def _env(tool_context: ToolContext) -> dict[str, str]:
    """Paths an agent's scripts may read besides DATA_DIR. The analysts also get
    kit.py, tested helpers for rates, intervals, drift and the live model."""
    env: dict[str, str] = {}
    if tool_context.agent_name in ANALYSTS:
        env = _analyst_env(root(tool_context))
    if tool_context.agent_name == "analyst":
        return {**env, "RUN_DIR": str(current())}
    ref = read_json(current() / "reports" / "feature_ref.json")
    if ref:
        env["FEATURE_DIR"] = str(feature_store.path(ref["view"], ref["version"]))
    return env


def _analyst_env(folder: Path) -> dict[str, str]:
    from app.harness import registry

    shutil.copy2(KIT, folder / "kit.py")  # fresh every run: edits are overwritten
    config = settings.load()
    env = {
        "PYTHONPATH": str(folder),
        "DATASET": config.dataset,
        "TARGET": config.target,
        "POSITIVE": config.positive_value or "",
        "TRAFFIC_FILE": str(project.traffic_file() or ""),
        "FEATURE_VIEW_DIR": str(feature_store.STORE / config.feature_view),
    }
    if live := registry.production():
        env["LIVE_MODEL_DIR"] = str(registry.REGISTRY / live["version"])
        ref = live.get("feature_view") or {}
        if ref.get("view"):
            env["LIVE_FEATURE_DIR"] = str(
                feature_store.path(ref["view"], ref["version"])
            )
    return env


def _lines(lines: Any) -> list[str]:
    return [line.strip() for line in lines if line.strip()]


def _too_long(lines: list[str], limit: int, what: str) -> str:
    """Overlong lines are sent back to be rewritten, never cut off mid-sentence. Every
    overlong line is named at once, with how much to cut, so one rewrite fixes all."""
    if odd := [line for line in lines if not isinstance(line, str)]:
        return (
            f"Each {what} must be one plain string, not {type(odd[0]).__name__}: "
            f"write it as a sentence, e.g. the text of {str(odd[0])[:60]}. "
        )
    # A line a few characters over is accepted: a rewrite costs a whole model call.
    slack = limit // 10
    long = [line.strip() for line in lines if len(line.strip()) > limit + slack]
    if not long:
        return ""
    return (
        f"Rewrite shorter: each {what} must be one plain line under {limit} "
        f"characters. Too long: "
        + "; ".join(
            f"{i + 1}. ({len(line)} chars, cut {len(line) - limit}) {line[:60]}…"
            for i, line in enumerate(long)
        )
        + ". "
    )


def _clip(text: str) -> str:
    if len(text) <= OUTPUT_HEAD + OUTPUT_TAIL:
        return text
    return f"{text[:OUTPUT_HEAD]}\n…[{len(text) - OUTPUT_HEAD - OUTPUT_TAIL} chars cut]…\n{text[-OUTPUT_TAIL:]}"


def _error(message: str) -> dict[str, Any]:
    return {"status": "error", "message": message}
