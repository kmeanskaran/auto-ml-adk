"""The run index: what every past run tried, what it got, and what was decided.

runs/index.jsonl gets one line when a run starts and one when it finishes. The next
run's engineer and skeptic read a short digest of the finished runs on the same data
(lessons) and the production model's score (the bar to beat), as hypotheses to test,
never as findings: the data may have changed, and so may the question.

A new run also builds on the last finished run's code: carry_over copies it into the
new run's prior/ folder, and the engineer rethinks it with the human's feedback for the
run. Clearing everything (a cold start) is the way to start from nothing.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from app import settings
from app.harness import catalog, project, registry
from app.harness.evaluation import show
from app.harness.project import INDEX, append_jsonl, file_hash, read_json

MAX_LESSONS = 3  # past runs shown to the team
MAX_FINDINGS = 2  # skeptic findings kept per stage of a past run
BRIEF = "reports/brief.json"  # the human's feedback and the run this one builds on
PRIOR = "prior"  # the run-folder subfolder holding the previous run's code
CARRIED = ("src", "notes")  # what a new run gets from the previous one


def started(run: Path, config: settings.Settings, feedback: str = "") -> dict[str, Any]:
    """Record the run's start, carry the last run's code over, and write its brief."""
    brief = {"feedback": feedback.strip(), "builds_on": carry_over(run, config)}
    project.write_json(run / BRIEF, brief)
    append_jsonl(
        project.RUNS / INDEX,
        {
            "run": run.name,
            "event": "started",
            "dataset": config.dataset,
            "data": file_hash(project.LAKE, config.dataset),
            **brief,
        },
    )
    return brief


def builds_on(dataset: str, exclude: str = "") -> str | None:
    """The newest finished run on this dataset that left code to build on."""
    return next(
        (
            r["run"]
            for r in runs(dataset)
            if r["run"] != exclude and (project.RUNS / r["run"] / "src").is_dir()
        ),
        None,
    )


def carry_over(run: Path, config: settings.Settings) -> str | None:
    """Copy that run's code and notes into run/prior/; None means a cold start."""
    prior = builds_on(config.dataset, exclude=run.name)
    for name in CARRIED if prior else ():
        if (project.RUNS / prior / name).is_dir():
            shutil.copytree(
                project.RUNS / prior / name, run / PRIOR / name, dirs_exist_ok=True
            )
    return prior


def finished(run: Path, outcome: str) -> dict[str, Any]:
    """Summarise a run from its own files and add it to the index."""
    config = settings.load()
    plan = read_json(run / "reports" / "plan.json") or {}
    evaluation = read_json(run / "reports" / "evaluation.json") or {}
    feature_ref = read_json(run / "reports" / "feature_ref.json") or {}
    decisions = _jsonl(run / "decisions.jsonl")
    best = evaluation.get("best")
    metric = evaluation.get("metric") or plan.get("metric")
    score = (
        ((evaluation.get("candidates") or {}).get(best) or {})
        .get("test", {})
        .get(metric)
        if best and metric
        else None
    )
    reviews = _jsonl(run / "reviews" / "history.jsonl")
    brief = read_json(run / BRIEF) or {}
    record = {
        "run": run.name,
        "event": "finished",
        "dataset": config.dataset,
        "data": file_hash(project.LAKE, config.dataset),
        "outcome": outcome[:200],
        "feedback": brief.get("feedback") or "",
        "builds_on": brief.get("builds_on"),
        "feature_view": feature_ref.get("version"),
        "features": len(
            (read_json(run / "reports" / "features.json") or {}).get("features") or {}
        ),
        "models": plan.get("models"),
        "metric": metric,
        "best": best,
        "test_score": score,
        "version": next(
            (d.get("version") for d in reversed(decisions) if d.get("version")), None
        ),
        "decisions": [
            f"{d['stage']}:{d.get('action')} by {d.get('by')}" for d in decisions
        ],
        "human_overrides": sum(1 for d in decisions if d.get("by") == "human"),
        "skeptic": {
            stage: [
                f
                for r in reviews
                if r.get("stage") == stage
                for f in r.get("findings", [])
            ][-MAX_FINDINGS:]
            for stage in ("features", "model")
        },
        "warnings": evaluation.get("warnings") or [],
    }
    append_jsonl(project.RUNS / INDEX, record)
    return record


def runs(dataset: str | None = None) -> list[dict[str, Any]]:
    """Finished runs, newest first, optionally on one dataset."""
    done = [
        r
        for r in _jsonl(project.RUNS / INDEX)
        if r.get("event") == "finished"
        and (dataset is None or r.get("dataset") == dataset)
    ]
    return list(reversed(done))


def briefing(current_run: str) -> str:
    """What the team knows before it starts: the bar to beat and past runs' lessons."""
    config = settings.load()
    lines = []
    if live := registry.production():
        metric = live.get("metric") or "roc_auc"
        value = (live.get("test") or {}).get(metric)
        lines.append(
            f"Production now: {live.get('version')} ({live.get('model')}), "
            f"{metric} on its final test {_show(metric, value)}. A new model goes live "
            "only if it beats this."
        )
    data = file_hash(project.LAKE, config.dataset)
    past = [r for r in runs(config.dataset) if r.get("run") != current_run][
        :MAX_LESSONS
    ]
    if past:
        lines.append(
            "Past runs on this dataset, newest first. Hypotheses to test, not findings"
            + (
                ""
                if all(r.get("data") == data for r in past)
                else "; the data has changed since some of them"
            )
            + ":"
        )
        for r in past:
            if r.get("feedback"):
                lines.append(f"- {r['run']} was asked: {r['feedback'][:200]}")
            lines.append(
                f"- {r['run']}: {r.get('features') or '?'} features, models {r.get('models')}, "
                f"best {r.get('best')} with {r.get('metric')} {_show(r.get('metric'), r.get('test_score'))}; "
                f"ended: {r.get('outcome')}"
            )
            for stage, findings in (r.get("skeptic") or {}).items():
                for finding in findings:
                    lines.append(f"  skeptic ({stage}): {finding}")
            for warning in (r.get("warnings") or [])[:1]:
                lines.append(f"  harness warning: {warning}")
    return "\n".join(lines)


def _show(metric: str | None, value: Any) -> str:
    if value is None or metric not in catalog.METRICS:
        return "n/a"
    return show(metric, float(value))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
