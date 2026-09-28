"""The ML pipeline as an ADK Workflow: a fixed order, agents decide the content.

  START → profile → analyst → check
        → features engineer → check ─fix→ (engineer)        [materializes the feature view]
        → skeptic → REVIEW FEATURES ─feedback→ (engineer)    [continue registers it in the store]
        → TRAINING PLAN (human picks models and metric)
        → model engineer → check ─fix→ (engineer)           [harness evaluates every candidate]
        → skeptic → PROMOTE ─retrain→ (model engineer) ─replan→ (training plan)
                            promote / keep as candidate → registry version

The human answers at the capitalised steps; each can also discard the run. Every
answer is logged to the run's decisions.jsonl. Registering features and models
is idempotent, so a step that runs again when the workflow resumes changes nothing.
"""

from __future__ import annotations

import shutil
from typing import Any

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import Workflow, node

from app import settings
from app.agents import analyst_profile, engineer, skeptic
from app.harness import catalog, contracts, feature_store, registry
from app.harness.profile import profile
from app.harness.project import (
    LAKE,
    append_jsonl,
    current,
    new_run,
    read_json,
    write_json,
)

MAX_FIXES = 2  # times an agent is sent back to complete an incomplete hand-over


def _request(stage: str, instruction: str = "") -> str:
    config = settings.load()
    request = f"{config.brief()}\n\n"
    if stage == "profile":
        request += "Take the first look at the data."
    elif stage == "features":
        request += "Build the features stage."
    else:
        plan = read_json(current() / "reports" / "plan.json") or {}
        request += (
            f"Build the model stage. Approved plan: models {plan.get('models')}, "
            f"metric {plan.get('metric')}."
        )
        if plan.get("note"):
            instruction = "\n".join(filter(None, [plan["note"], instruction]))
    if instruction:
        request += (
            f"\n\nInstruction from the human's review, follow it first:\n{instruction}"
        )
    return request


def _reset(stage: str) -> None:
    """Clear a stage's hand-over so a redo cannot pass on the previous round's files."""
    run = current()
    stale = [run / "receipts" / f"{stage}.json", run / "reviews" / f"{stage}.json"]
    if stage == "features":
        stale += [
            run / "reports" / "feature_stats.json",
            run / "reports" / "plan_proposal.json",
        ]
        shutil.rmtree(run / feature_store.STAGED, ignore_errors=True)
        (run / feature_store.STAGED).mkdir()
    else:
        stale += [run / "reports" / "evaluation.json"]
    for path in stale:
        path.unlink(missing_ok=True)


def _log(stage: str, **record: Any) -> None:
    append_jsonl(current() / "decisions.jsonl", {"stage": stage, **record})


def _instruction(stage: str, answer: dict[str, Any]) -> str:
    """The selected skeptic recommendations plus the human's own words."""
    recommendations = (read_json(current() / "reviews" / f"{stage}.json") or {}).get(
        "recommendations", []
    )
    picked = [
        recommendations[i]
        for i in answer.get("apply") or []
        if isinstance(i, int) and 0 <= i < len(recommendations)
    ]
    text = str(answer.get("text", "")).strip()
    return "\n".join(f"- {line}" for line in [*picked, *([text] if text else [])])


def start(node_input: Any) -> str:
    """A new run gets a fresh project folder and a measured profile of the data."""
    del node_input
    run = new_run()
    config = settings.load()
    write_json(
        run / "reports" / "profile.json", profile(LAKE / config.dataset, config.target)
    )
    return _request("profile")


def check(stage: str, next_stage: str | None):
    """Send the agent back while its hand-over is incomplete, then move on."""

    @node(name=f"check_{stage}", rerun_on_resume=False)
    async def check_node(ctx: Context, node_input: Any):
        del node_input
        tries = ctx.state.get(f"fixes_{stage}", 0)
        issues = await contracts.problems(current(), stage)
        if stage != "profile" and not read_json(
            current() / "receipts" / f"{stage}.json"
        ):
            issues.append(
                "no receipt submitted: call submit_receipt when the stage is complete"
            )
        if issues and tries < MAX_FIXES:
            yield Event(state={f"fixes_{stage}": tries + 1})
            yield Event(
                output=_request(
                    stage, "Your hand-over is incomplete:\n- " + "\n- ".join(issues)
                ),
                route="fix",
            )
            return
        yield Event(state={f"fixes_{stage}": 0})
        write_json(current() / "reports" / f"check_{stage}.json", {"problems": issues})
        if next_stage:
            yield Event(output=_request(next_stage), route="next")
        else:
            note = f"\nThe completeness check still reports: {issues}" if issues else ""
            yield Event(
                output=f"Review the {stage} stage.\n{settings.load().brief()}{note}",
                route="next",
            )

    return check_node


def _needs_review(ctx: Context, stage: str, round_: int) -> bool:
    """The skeptic has not reviewed this round yet and has not been reminded."""
    return not read_json(current() / "reviews" / f"{stage}.json") and not ctx.state.get(
        f"reminded_{stage}_{round_}"
    )


def _remind(stage: str, round_: int):
    yield Event(state={f"reminded_{stage}_{round_}": True})
    yield Event(
        output=f"You have not submitted your review of the {stage} stage. Review it "
        "and call submit_review.",
        route="remind",
    )


@node(name="review_features", rerun_on_resume=True)
async def review_features(ctx: Context, node_input: Any):
    """The human reads the features and the skeptic's recommendations: feedback or continue."""
    del node_input
    round_ = ctx.state.get("round_features", 0)
    key = f"review_features_{round_}"
    answer = (ctx.resume_inputs or {}).get(key)
    if answer is None and _needs_review(ctx, "features", round_):
        for event in _remind("features", round_):
            yield event
        return
    if answer is None:
        yield RequestInput(
            interrupt_id=key,
            message="Review the features",
            payload={"stage": "features"},
        )
        return
    choice = str(answer.get("choice", ""))
    instruction = _instruction("features", answer)
    blocked = (read_json(current() / "reports" / "check_features.json") or {}).get(
        "problems"
    )
    if choice == "continue" and blocked:
        choice, instruction = (
            "feedback",
            "Fix these problems first:\n- " + "\n- ".join(blocked),
        )
    yield Event(state={"round_features": round_ + 1})
    if choice == "discard":
        _log("features", action="discard")
        yield Event(output="Discarded at the features review.", route="discard")
    elif choice == "feedback" and instruction:
        _log("features", action="feedback", text=instruction)
        _reset("features")
        yield Event(output=_request("features", instruction), route="redo")
    else:
        view = settings.load().feature_view
        definition = feature_store.register(current(), view)
        ref = {
            "view": view,
            "version": definition["version"],
            "path": str(feature_store.path(view, definition["version"])),
        }
        write_json(current() / "reports" / "feature_ref.json", ref)
        _log(
            "features",
            action="continue",
            feature_view=f"{view}:{definition['version']}",
        )
        yield Event(output="Confirm the training plan.", route="continue")


@node(name="plan_gate", rerun_on_resume=True)
async def plan_gate(ctx: Context, node_input: Any):
    """The human confirms which models to train and which metric decides."""
    del node_input
    round_ = ctx.state.get("round_plan", 0)
    key = f"plan_{round_}"
    answer = (ctx.resume_inputs or {}).get(key)
    if answer is None:
        yield RequestInput(
            interrupt_id=key,
            message="Confirm the training plan",
            payload={"stage": "plan"},
        )
        return
    yield Event(state={"round_plan": round_ + 1})
    if str(answer.get("choice")) == "discard":
        _log("plan", action="discard")
        yield Event(output="Discarded at the training plan.", route="discard")
        return
    proposal = read_json(current() / "reports" / "plan_proposal.json") or {}
    models = [str(m) for m in answer.get("models") or proposal.get("models") or []]
    metric = str(answer.get("metric") or proposal.get("metric") or "")
    if contracts.plan_problems(
        models, metric
    ):  # the console validates; never train on a bad plan
        models = [m for m in models if m in catalog.models()] or list(catalog.models())[
            :1
        ]
        metric = metric if metric in catalog.METRICS else "roc_auc"
    plan = {
        "models": models,
        "metric": metric,
        "proposed": {k: proposal.get(k) for k in ("models", "metric", "reason")},
        "note": str(answer.get("text", "")).strip(),
    }
    write_json(current() / "reports" / "plan.json", plan)
    _log("plan", action="train", models=models, metric=metric, text=plan["note"])
    _reset("model")
    yield Event(output=_request("model"), route="train")


@node(name="promote_gate", rerun_on_resume=True)
async def promote_gate(ctx: Context, node_input: Any):
    """The human picks a candidate and promotes it, keeps it, retrains, or replans."""
    del node_input
    round_ = ctx.state.get("round_model", 0)
    key = f"promote_{round_}"
    answer = (ctx.resume_inputs or {}).get(key)
    if answer is None and _needs_review(ctx, "model", round_):
        for event in _remind("model", round_):
            yield event
        return
    if answer is None:
        yield RequestInput(
            interrupt_id=key,
            message="Promote to production?",
            payload={"stage": "promote"},
        )
        return
    choice = str(answer.get("choice", ""))
    evaluation = read_json(current() / "reports" / "evaluation.json") or {}
    candidates = evaluation.get("candidates") or {}
    model = str(answer.get("model") or evaluation.get("best") or "")
    if choice in ("promote", "keep") and model not in candidates:
        choice = "retrain"
        answer = {
            "text": "No evaluated candidate could be registered; fix training so every planned model evaluates."
        }
    yield Event(state={"round_model": round_ + 1})
    if choice in ("promote", "keep"):
        ref = read_json(current() / "reports" / "feature_ref.json") or {}
        plan = read_json(current() / "reports" / "plan.json") or {}
        version = registry.register(
            current(),
            model,
            evaluation,
            {"view": ref.get("view"), "version": ref.get("version")},
            feature_store.path(ref["view"], ref["version"]),
            {k: plan.get(k) for k in ("models", "metric")},
        )
        if choice == "promote":
            registry.promote(version["version"])
        _log("promote", action=choice, model=model, version=version["version"])
        state = "is in production" if choice == "promote" else "is kept as a candidate"
        yield Event(output=f"{version['version']} ({model}) {state}.", route="done")
    elif choice == "retrain":
        instruction = _instruction("model", answer)
        _log("promote", action="retrain", text=instruction)
        _reset("model")
        yield Event(output=_request("model", instruction), route="retrain")
    elif choice == "replan":
        _log("promote", action="replan")
        yield Event(output="Change the training plan.", route="replan")
    else:
        _log("promote", action="discard")
        yield Event(output="Discarded at the promote review.", route="done")


def finish(node_input: Any) -> str:
    return str(node_input)


def build(model: Any = None) -> Workflow:
    """The pipeline graph. Tests pass a scripted model; production uses the default."""
    analyst = analyst_profile(model)
    engineer_features, engineer_model = (
        engineer("features", model),
        engineer("model", model),
    )
    skeptic_features, skeptic_model = (
        skeptic("features", model),
        skeptic("model", model),
    )
    check_profile = check("profile", next_stage="features")
    check_features, check_model = check("features", None), check("model", None)
    return Workflow(
        name="ml_team",
        description="Profiles the data, engineers features, trains and promotes a model; "
        "the human reviews features, the training plan and promotion.",
        edges=[
            ("START", start),
            (start, analyst),
            (analyst, check_profile),
            (check_profile, {"fix": analyst, "next": engineer_features}),
            (engineer_features, check_features),
            (check_features, {"fix": engineer_features, "next": skeptic_features}),
            (skeptic_features, review_features),
            (
                review_features,
                {
                    "remind": skeptic_features,
                    "redo": engineer_features,
                    "continue": plan_gate,
                    "discard": finish,
                },
            ),
            (plan_gate, {"train": engineer_model, "discard": finish}),
            (engineer_model, check_model),
            (check_model, {"fix": engineer_model, "next": skeptic_model}),
            (skeptic_model, promote_gate),
            (
                promote_gate,
                {
                    "remind": skeptic_model,
                    "retrain": engineer_model,
                    "replan": plan_gate,
                    "done": finish,
                },
            ),
        ],
    )


pipeline = build()
