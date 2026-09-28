"""HTTP routes: the console under /team, and served models under /predict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import settings
from app.harness import catalog, contracts, feature_store, registry
from app.harness.project import RUNS, read_json
from app.ui.runs import CHAT, PIPELINE

console = APIRouter(prefix="/team", tags=["console"])
serving = APIRouter(tags=["serving"])
PAGE = Path(__file__).parent / "static" / "index.html"
CHOICES = {
    "features": ("continue", "feedback", "discard"),
    "plan": ("train", "discard"),
    "promote": ("promote", "keep", "retrain", "replan", "discard"),
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


class Rows(BaseModel):
    rows: list[dict[str, Any]]


@console.get("", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(PAGE)


@console.post("/api/pipeline/start")
async def start_pipeline() -> dict:
    try:
        await PIPELINE.start()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"status": PIPELINE.status}


@console.post("/api/pipeline/answer")
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


@console.post("/api/registry/{version}/promote")
def make_live(version: str) -> dict:
    """Roll production forward or back to any registered version."""
    try:
        return registry.promote(version, reason="rollback")
    except registry.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc


@console.post("/api/registry/{version}/remove")
def remove_version(version: str) -> dict:
    """Take a version out of the model library (never the production one)."""
    try:
        return registry.remove(version)
    except registry.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@console.post("/api/chat")
async def ask_analyst(question: Question) -> dict:
    if not question.text.strip():
        raise HTTPException(422, "Ask a question.")
    try:
        await CHAT.ask(question.text.strip())
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"status": CHAT.status}


@console.get("/api/view")
def view() -> dict:
    return {
        "pipeline": _pipeline_view(),
        "registry": _registry_view(),
        "feature_store": feature_store.summary(),
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
        "config": {
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
        return read_json(self.run / relative) if self.run else None

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
        "details": profile,
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
    features = [
        {"name": name, **meta, **(stats.get("features") or {}).get(name, {})}
        for name, meta in (report.get("features") or {}).items()
    ]
    return {
        "key": "features",
        "label": "Engineer features",
        "agent": "Engineer · Skeptic",
        "state": state,
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
        "label": "Training plan",
        "agent": "You",
        "state": state,
        "findings": [
            f"Models: {', '.join(catalog.models().get(m, m) for m in plan['models'])}",
            f"Metric: {catalog.METRICS[plan['metric']].label}",
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
        "label": "Train and evaluate",
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
        findings = [f"{last['version']} ({last['model']}) is live at POST /predict"]
    elif action == "keep":
        findings = [f"{last['version']} ({last['model']}) registered as a candidate"]
    live = registry.production()
    return {
        "key": "promote",
        "label": "Promote",
        "agent": "You",
        "state": state,
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
        "threshold": version.get("threshold"),
        "feature_view": version.get("feature_view"),
        "run": version.get("run"),
        "created_at": version.get("created_at") or version.get("promoted_at"),
        "promoted_at": version.get("promoted_at"),
    }


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
