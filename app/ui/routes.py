"""HTTP API: the console's JSON API under /api (the Next.js frontend in frontend/
calls it), and served models under /predict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import settings
from app.harness import (
    catalog,
    contracts,
    feature_store,
    history,
    models,
    registry,
    trace,
)
from app.harness.project import RUNS, read_json
from app.ui.runs import CHAT, PIPELINE, clear

console = APIRouter(prefix="/api", tags=["console"])
serving = APIRouter(tags=["serving"])
CHOICES = {
    "features": ("continue", "feedback", "discard"),
    "plan": ("train", "discard"),
    "promote": ("promote", "keep", "retrain", "features", "replan", "discard"),
}


class Answer(BaseModel):
    choice: str
    text: str = ""
    apply: list[int] = Field(default_factory=list)  # indexes of skeptic recommendations
    models: list[str] = Field(default_factory=list)  # training plan
    metric: str = ""  # training plan
    model: str = ""  # candidate to promote or keep


class Question(BaseModel):
    text: str


class ModelChoice(BaseModel):
    model: str
    thinking: str


class Start(BaseModel):
    feedback: str = Field("", max_length=4000)  # optional: what the team should rethink


class Rows(BaseModel):
    rows: list[dict[str, Any]]


@console.post("/pipeline/start")
async def start_pipeline(start: Start | None = None) -> dict:
    try:
        await PIPELINE.start((start or Start()).feedback)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"status": PIPELINE.status}


@console.post("/pipeline/pause")
def pause_pipeline() -> dict:
    """Hold the run after the model call or script in flight; nothing runs until resumed."""
    try:
        PIPELINE.pause_run()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"status": PIPELINE.status}


@console.post("/pipeline/resume")
def resume_pipeline() -> dict:
    try:
        PIPELINE.resume_run()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"status": PIPELINE.status}


@console.post("/pipeline/restart")
async def restart_pipeline() -> dict:
    """Stop the run wherever it is and start a new one with the same feedback."""
    try:
        await PIPELINE.restart()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"status": PIPELINE.status}


@console.post("/model")
def choose_model(choice: ModelChoice) -> dict:
    """The Gemini model and thinking level the team uses from its next call on."""
    if not models.on_gemini():
        raise HTTPException(
            409, f"The team runs on {models.base()}, not Gemini (ML_MODEL)."
        )
    try:
        models.choose(choice.model, choice.thinking)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    trace.note(
        trace.PIPELINE,
        f"⚙ model set to {choice.model}, thinking {choice.thinking}",
        kind="step",
    )
    return models.view()


@console.post("/clear")
async def clear_everything() -> dict:
    """Cold start: remove every run, model, feature view, session and chat message."""
    try:
        await clear()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"status": PIPELINE.status}


@console.post("/pipeline/answer")
async def answer_review(answer: Answer) -> dict:
    stage = (PIPELINE.pause.payload.get("stage") if PIPELINE.pause else None) or ""
    if answer.choice not in CHOICES.get(stage, ()):
        raise HTTPException(
            422, f"{answer.choice!r} is not an answer to the {stage or 'current'} step."
        )
    if stage == "plan" and answer.choice == "train":
        if issues := contracts.plan_problems(answer.models, answer.metric):
            raise HTTPException(422, "; ".join(issues))
    if (
        stage == "features"
        and answer.choice == "feedback"
        and not (answer.text.strip() or answer.apply)
    ):
        raise HTTPException(
            422, "Pick a recommendation or write feedback for the engineer."
        )
    try:
        await PIPELINE.answer(answer.model_dump())
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"status": PIPELINE.status}


@console.post("/registry/{version}/promote")
def make_live(version: str) -> dict:
    """Roll production forward or back to any registered version."""
    try:
        return registry.promote(version, reason="rollback")
    except registry.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc


@console.post("/registry/{version}/remove")
def remove_version(version: str) -> dict:
    """Take a version out of the model library (never the production one)."""
    try:
        return registry.remove(version)
    except registry.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@console.post("/chat")
async def ask_analyst(question: Question) -> dict:
    if not question.text.strip():
        raise HTTPException(422, "Ask a question.")
    try:
        await CHAT.ask(question.text.strip())
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"status": CHAT.status}


@console.get("/runs")
def past_runs(limit: int = 20) -> dict:
    """Finished runs, newest first: what each tried, scored and decided."""
    return {"runs": history.runs()[: max(1, min(limit, 200))]}


@console.get("/view")
def view() -> dict:
    return {
        "pipeline": _pipeline_view(),
        "registry": _registry_view(),
        "feature_store": _feature_store_view(),
        "history": [_run_card(r) for r in history.runs()[:20]],
        "metrics": catalog.describe(),
        "model": models.view(),
        "chat": {"status": CHAT.status, "messages": CHAT.messages[-30:]},
    }


@serving.post("/predict")
def predict(body: Rows) -> dict:
    """Score raw records with the model currently in production."""
    return _score(body.rows, None)


@serving.post("/predict/{version}")
def predict_version(version: str, body: Rows) -> dict:
    """Score raw records with a specific registered version (candidates included)."""
    return _score(body.rows, version)


def _score(rows: list[dict[str, Any]], version: str | None) -> dict:
    try:
        return registry.predict(rows, version)
    except registry.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except (LookupError, ValueError, TypeError) as exc:
        raise HTTPException(
            422, f"Records could not be scored: {type(exc).__name__}: {exc}"
        ) from exc


# --- the pipeline, read from the run folder ------------------------------------------------


def _pipeline_view() -> dict:
    run = RUNS / PIPELINE.project if PIPELINE.project else None
    pause = (PIPELINE.pause.payload.get("stage") if PIPELINE.pause else None) or ""
    files = _Files(run)
    decisions = files.decisions()
    config = settings.load()
    stages = [
        _profile(files),
        _features(files, pause, decisions),
        _plan(files, pause, decisions),
        _model(files, pause),
        _promote(files, pause, decisions),
    ]
    return {
        "status": PIPELINE.status,
        "error": PIPELINE.error,
        "run": PIPELINE.project,
        "waiting_on": pause,
        "why": (PIPELINE.pause.payload.get("why") if PIPELINE.pause else "") or "",
        "activity": trace.activity(run) if run else [],
        "usage": trace.usage(run) if run else None,  # time, tokens, cache, so far
        "brief": files.json(history.BRIEF),
        "next_builds_on": history.builds_on(config.dataset),
        "config": {
            "goal": config.goal,
            "record": config.record,
            "positive": config.positive,
            "false_alarm": config.false_alarm,
            "ask_human": list(config.ask_human),
            "dataset": config.dataset,
            "target": config.target,
            "feature_view": config.feature_view,
            "prediction_moment": config.prediction_moment,
            "costs": {
                "missed": config.cost_missed,
                "false_alarm": config.cost_false_alarm,
            },
        },
        "stages": stages,
    }


class _Files:
    def __init__(self, run: Path | None):
        self.run = run

    def json(self, relative: str) -> dict | None:
        # Agents write these files mid-stage; until they hold the agreed shape, show nothing.
        data = read_json(self.run / relative) if self.run else None
        return data if isinstance(data, dict) else None

    def decisions(self) -> list[dict]:
        path = self.run / "decisions.jsonl" if self.run else None
        if not path or not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]


def _last(decisions: list[dict], stage: str) -> dict:
    return next((d for d in reversed(decisions) if d.get("stage") == stage), {})


def _working(*agents: str) -> str | None:
    if PIPELINE.status == "running" and PIPELINE.active in agents:
        return "reviewing" if PIPELINE.active.startswith("skeptic") else "working"
    return None


def _profile(files: _Files) -> dict:
    summary, profile = (
        files.json("reports/summary.json"),
        files.json("reports/profile.json"),
    )
    state = _working("analyst_profile") or ("done" if summary else "waiting")
    if state == "waiting" and profile and PIPELINE.status == "running":
        state = "working"
    return {
        "key": "profile",
        "label": "Understand the data",
        "agent": "Analyst",
        "state": state,
        "headline": (summary or {}).get("headline", ""),
        "findings": (summary or {}).get("findings", []),
        "charts": (summary or {}).get("charts", []),
        "details": profile,
        # a rerun with no feedback on unchanged data reuses the last run's first look
        "reused_from": (files.json(history.REUSED) or {}).get("profile"),
    }


def _features(files: _Files, pause: str, decisions: list[dict]) -> dict:
    last = _last(decisions, "features")
    state = (
        "your_review"
        if pause == "features"
        else _working("engineer_features", "skeptic_features")
        or {"continue": "done", "discard": "discarded"}.get(
            last.get("action", ""), "waiting"
        )
    )
    report = files.json("reports/features.json") or {}
    stats = files.json("reports/feature_stats.json") or {}
    declared = report.get("features")
    measured = stats.get("features")
    measured = measured if isinstance(measured, dict) else {}
    features = [
        {
            "name": name,
            **meta,
            **(measured.get(name) if isinstance(measured.get(name), dict) else {}),
        }
        for name, meta in (declared.items() if isinstance(declared, dict) else [])
        if isinstance(meta, dict)
    ]
    return {
        "key": "features",
        "label": "Engineer features",
        "agent": "Engineer · Skeptic",
        "state": state,
        "decided_by": last.get("by", ""),
        "rounds": sum(
            1
            for d in decisions
            if d.get("stage") == "features" and d.get("action") == "feedback"
        ),
        "findings": (files.json("receipts/features.json") or {}).get("findings", []),
        "details": {
            "features": features,
            "tables": stats.get("tables"),
            "split": report.get("split"),
            "rows_removed": report.get("rows_removed"),
            "excluded_columns": report.get("excluded_columns"),
            "store": files.json("reports/feature_ref.json"),
        }
        if report
        else None,
        "review": files.json("reviews/features.json"),
        "problems": (files.json("reports/check_features.json") or {}).get(
            "problems", []
        ),
    }


def _plan(files: _Files, pause: str, decisions: list[dict]) -> dict:
    last = _last(decisions, "plan")
    plan = files.json("reports/plan.json")
    state = (
        "your_review"
        if pause == "plan"
        else {"train": "done", "discard": "discarded"}.get(
            last.get("action", ""), "waiting"
        )
    )
    proposal = files.json("reports/plan_proposal.json") or {}
    return {
        "key": "plan",
        "label": "Choose models",
        "agent": "Team · You",
        "state": state,
        "decided_by": last.get("by", ""),
        "findings": [
            f"Models: {', '.join(catalog.models().get(m, m) for m in plan['models'])}",
            f"Winner picked by: {catalog.describe()[plan['metric']]['plain']}",
        ]
        if plan and state == "done"
        else [],
        "proposal": proposal,
        "plan": plan,
        "options": catalog.as_options(),
    }


def _model(files: _Files, pause: str) -> dict:
    evaluation = files.json("reports/evaluation.json")
    receipt = files.json("receipts/model.json")
    state = _working("engineer_model", "skeptic_model") or (
        "done" if evaluation and receipt else "waiting"
    )
    if pause == "promote":
        state = "done"
    return {
        "key": "model",
        "label": "Train and test",
        "agent": "Engineer · Skeptic",
        "state": state,
        "findings": (receipt or {}).get("findings", []),
        "details": _leaderboard(evaluation),
        "problems": (files.json("reports/check_model.json") or {}).get("problems", []),
    }


def _promote(files: _Files, pause: str, decisions: list[dict]) -> dict:
    last = _last(decisions, "promote")
    action = last.get("action", "")
    state = (
        "your_review"
        if pause == "promote"
        else {"promote": "done", "keep": "kept", "discard": "discarded"}.get(
            action, "waiting"
        )
    )
    findings = []
    if action == "promote":
        findings = [
            f"{last['version']} ({last['model']}) is live; POST /predict uses it"
        ]
    elif action == "keep":
        findings = [f"{last['version']} ({last['model']}) is kept in the model library"]
    live = registry.production()
    return {
        "key": "promote",
        "label": "Go live",
        "agent": "Team · You",
        "state": state,
        "decided_by": last.get("by", ""),
        "findings": findings,
        "review": files.json("reviews/model.json"),
        "leaderboard": _leaderboard(files.json("reports/evaluation.json")),
        "live": _version_card(live) if live else None,
    }


def _leaderboard(evaluation: dict | None) -> dict | None:
    if not evaluation:
        return None
    metric = evaluation["metric"]
    rows = [
        {"model": name, "label": catalog.models().get(name, name), **scores}
        for name, scores in evaluation["candidates"].items()
    ]
    sign = 1 if catalog.METRICS[metric].higher_is_better else -1
    rows.sort(
        key=lambda r: (
            r["valid"].get(metric) is not None,
            sign * (r["valid"].get(metric) or 0),
        ),
        reverse=True,
    )
    return {
        "metric": metric,
        "metric_label": catalog.METRICS[metric].label,
        "best": evaluation["best"],
        "baseline": evaluation["baseline"],
        "warnings": evaluation.get("warnings", []),
        "protocol": evaluation.get("protocol", ""),
        "chosen_on": evaluation.get("chosen_on", "valid"),
        "rows": rows,
        "metrics": [{"key": k, "label": m.label} for k, m in catalog.METRICS.items()],
    }


# --- registry ------------------------------------------------------------------------------


def _version_card(version: dict) -> dict:
    metric = version.get("metric") or "roc_auc"
    return {
        "version": version["version"],
        "status": version.get("status"),
        "model": catalog.models().get(version.get("model") or "", version.get("model")),
        "model_key": version.get("model"),
        "metric": metric,
        "metric_label": catalog.METRICS[metric].label
        if metric in catalog.METRICS
        else metric,
        "test": version.get("test") or {},
        "baseline_test": version.get("baseline_test") or {},
        "threshold": version.get("threshold"),
        "feature_view": version.get("feature_view"),
        "run": version.get("run"),
        "created_at": version.get("created_at") or version.get("promoted_at"),
        "promoted_at": version.get("promoted_at"),
    }


def _feature_store_view() -> list[dict]:
    """Every feature view version with its features; which models use it, which
    features come from a protected attribute (fairness.attributes)."""
    protected = set(settings.load().protected)
    used: dict[tuple[str, str], list[dict]] = {}
    for v in registry.versions():
        ref = v.get("feature_view") or {}
        if v.get("status") != "removed" and ref.get("view"):
            used.setdefault((ref["view"], ref.get("version") or ""), []).append(
                {"version": v["version"], "status": v.get("status")}
            )
    views = feature_store.summary()
    for view in views:
        for version in view["versions"]:
            models = used.get((view["view"], version["version"]), [])
            version["used_by"] = models
            version["live"] = any(m["status"] == "production" for m in models)
            for column in version["columns"]:
                column["protected"] = sorted(set(column["source"]) & protected)
    return views


def _run_card(record: dict) -> dict:
    keep = (
        "run",
        "at",
        "outcome",
        "feature_view",
        "features",
        "models",
        "metric",
        "feedback",
        "builds_on",
    )
    card = {k: record.get(k) for k in keep}
    card.update(
        best=catalog.models().get(record.get("best") or "", record.get("best")),
        test_score=record.get("test_score"),
        version=record.get("version"),
        human_overrides=record.get("human_overrides", 0),
        decisions=len(record.get("decisions") or []),
        warnings=len(record.get("warnings") or []),
        usage=record.get("usage"),  # None for runs recorded before usage was
    )
    return card


def _registry_view() -> dict:
    live = registry.production()
    return {
        "production": _version_card(live) if live else None,
        "versions": [
            _version_card(v) for v in registry.versions() if v["status"] != "removed"
        ],
        "metrics": [
            {"key": k, "label": m.label, "higher_is_better": m.higher_is_better}
            for k, m in catalog.METRICS.items()
        ],
        "audit": registry.audit(10),
    }
