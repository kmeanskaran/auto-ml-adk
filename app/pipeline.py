"""The ML pipeline as an ADK Workflow: a fixed order, agents decide the content.

  START → profile → analyst → check
        → features engineer → check ─fix→ (engineer)        [materializes the feature view]
        → skeptic → REVIEW FEATURES ─feedback→ (engineer)    [continue registers it in the store]
        → TRAINING PLAN (human picks models and metric)
        → model engineer → check ─fix→ (engineer)           [harness evaluates every candidate]
        → skeptic → PROMOTE ─retrain→ (model engineer) ─replan→ (training plan)
                            ─features→ (features engineer, then review, plan, train again)
                            promote / keep as candidate → registry version

At the capitalised steps the team answers first, the way an ML team settles its own
reviews: while the skeptic has concerns, the engineer gets every recommendation, for
up to autonomy.self_review_rounds rounds (config/config.yml). After that a step
waits for the human only if it is listed in autonomy.ask_human or the team could not
settle it; the pause says why. Otherwise the team decides: sound features continue,
the proposed plan trains, and a sound model goes live if it beats production.

A run starts with the human's optional feedback (session state "feedback") and builds
on the last finished run: its code is copied to prior/ and the engineer rethinks it
with the feedback. Every answer, the team's or the human's, is logged to the run's decisions.jsonl and
the trace (app/harness/trace.py). Registering features and models is idempotent,
so a step that runs again when the workflow resumes changes nothing.
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
from app.harness import catalog, contracts, feature_store, history, registry, trace
from app.harness.evaluation import fairness_flags
from app.harness.profile import profile
from app.harness.project import (
    FEEDBACK_KEY,
    LAKE,
    RUN_KEY,
    append_jsonl,
    bind,
    current,
    new_run,
    read_json,
    write_json,
)

NO_RECEIPT = "no receipt submitted: call submit_receipt when the stage is complete"
# Only the receipt missing: the work is done, so ask for the receipt alone rather than
# re-sending the stage (which had the engineer rebuild finished work).
RECEIPT_ONLY = (
    "Your hand-over is complete except the receipt: you wrote your findings as text. "
    "Call submit_receipt now with at most three one-line findings. Do not change or "
    "re-run any file."
)


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
    request += _direction(stage)
    if stage != "profile":
        request += _briefing()
    return request


def _direction(stage: str) -> str:
    """The human's feedback for this run and the previous run's code to rethink."""
    brief = read_json(current() / history.BRIEF) or {}
    feedback, prior = brief.get("feedback"), brief.get("builds_on")
    parts = []
    if feedback:
        parts.append(
            "The human's feedback for this run. Weigh it in every decision and say in "
            f"your hand-over how you acted on it:\n{feedback}"
        )
    if prior and stage in ("features", "model"):
        script = (
            "src/data.py and src/features.py" if stage == "features" else "src/train.py"
        )
        parts.append(
            f"This run builds on {prior}: its code is in {history.PRIOR}/src/ and its notes "
            f"in {history.PRIOR}/notes/. Read its {script} first and rethink it"
            + (" with the feedback" if feedback else "")
            + ": it is a draft, not a result. Keep a part only when it still holds on "
            "this data when you run it, change what the feedback or your measurements "
            "say, and write your own src/ (prior/ is never run)."
        )
    reused = read_json(current() / history.REUSED) or {}
    if reused and stage == "features":
        tried = ", ".join(reused.get("models") or []) or "none recorded"
        parts.append(
            f"No feedback this time, so this run explores. Keep the features that "
            f"measured well in {prior} and try at least one change it did not make (a "
            "new feature, a transformation, or dropping a weak one); show its effect in "
            f"your hand-over. Do not repeat {prior}'s training plan: it tried {tried} and "
            f"{reused.get('best') or 'none'} won. Keep the winner as the bar and propose "
            "model families or settings it did not try."
        )
    if reused and stage == "model":
        parts.append(
            f"No feedback this time, so this run explores: tune differently from "
            f"{prior}, whose best was {reused.get('best') or 'none'}, and say what you "
            "changed."
        )
    if prior and stage == "review":
        parts.append(
            f"This run reworked {prior}'s code ({history.PRIOR}/src/). Check that the "
            "changes are sound"
            + (" and that the feedback was acted on" if feedback else "")
            + "."
        )
    return "".join(f"\n\n{p}" for p in parts)


def _briefing() -> str:
    """What the team already knows, handed over with the request so no agent spends
    tool calls re-reading it: the data at a glance, the analyst's findings, the bar
    to beat and past runs' lessons. Files stay the source of truth for details."""
    run = current()
    parts = []
    report = read_json(run / "reports" / "profile.json") or {}
    columns = [
        f"`{c['name']}` {c.get('kind') or c.get('dtype')}, missing {c.get('missing_pct', 0)}%, "
        f"{c.get('unique')} distinct, signal_auc {c.get('signal_auc')}"
        for c in report.get("column_profiles") or []
    ]
    if columns:
        target = report.get("target") or {}
        parts.append(
            f"Data at a glance (reports/profile.json): {report.get('rows')} rows, outcome "
            f"rate {target.get('positive_rate')}, {report.get('duplicate_rows')} duplicate rows.\n"
            + "\n".join(f"- {line}" for line in columns)
        )
    summary = read_json(run / "reports" / "summary.json") or {}
    if summary.get("findings"):
        parts.append(
            "The analyst's findings:\n"
            + "\n".join(f"- {f}" for f in summary["findings"])
        )
    config = settings.load()
    built = (read_json(run / "reports" / "features.json") or {}).get("features") or {}
    protected = sorted(
        f"`{name}` (from {', '.join(sorted(set(spec.get('source') or []) & set(config.protected)))})"
        for name, spec in built.items()
        if isinstance(spec, dict)
        and set(spec.get("source") or []) & set(config.protected)
    )
    if protected:
        parts.append(
            "Features built from a protected attribute (fair lending): "
            + ", ".join(protected)
            + ". Each needs a reason a lender could defend, or it goes."
        )
    if known := history.briefing(run.name):
        parts.append(known)
    return "".join(f"\n\n{p}" for p in parts)


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


def _log(stage: str, by: str = "human", **record: Any) -> None:
    append_jsonl(current() / "decisions.jsonl", {"stage": stage, "by": by, **record})
    detail = record.get("text") or record.get("version") or ""
    trace.note(
        trace.PIPELINE if by == "team" else "human",
        f"◆ {by} decided at {stage}: {record.get('action')}"
        + (f" · {str(detail).splitlines()[0][:120]}" if detail else ""),
        kind="decision",
        stage=stage,
        by=by,
        **record,
    )


def _review(stage: str) -> tuple[dict[str, Any], list[str]]:
    """The skeptic's review of a stage and the problems its completeness check left."""
    review = read_json(current() / "reviews" / f"{stage}.json") or {}
    problems = (read_json(current() / "reports" / f"check_{stage}.json") or {}).get(
        "problems"
    ) or []
    return review, problems


def _sound(stage: str) -> bool:
    review, problems = _review(stage)
    return review.get("verdict") == "pass" and not problems


def _self_review(ctx: Context, stage: str) -> dict[str, Any] | None:
    """The team's own fix while the skeptic has concerns and rounds remain: every
    recommendation, plus any problem the completeness check left. None otherwise."""
    used = ctx.state.get(f"team_rounds_{stage}", 0)
    if _sound(stage) or used >= settings.load().self_review_rounds:
        return None
    review, problems = _review(stage)
    recommendations = review.get("recommendations") or []
    if not recommendations and not problems:
        return None
    # The skeptic repeating what the team just applied: another round would loop.
    if used and [*recommendations, *problems] == ctx.state.get(f"team_asked_{stage}"):
        return None
    text = "Fix these problems first:\n- " + "\n- ".join(problems) if problems else ""
    return {
        "apply": list(range(len(recommendations))),
        "text": text,
        "round": used + 1,
        "asked": [*recommendations, *problems],
    }


def _why(stage: str, review_stage: str) -> str:
    """Why a review waits for the human, shown with the question."""
    if stage in settings.load().ask_human:
        return f"Your call: {stage} is a human review (autonomy.ask_human)."
    rounds = settings.load().self_review_rounds
    return (
        f"The team could not settle this on its own: the skeptic still has concerns "
        f"after {rounds} round{'s' if rounds != 1 else ''} of fixes."
        if not _sound(review_stage)
        else "The team needs your decision here."
    )


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


@node(name="start", rerun_on_resume=False)
async def start(ctx: Context, node_input: Any):
    """A new run gets a fresh project folder, named in the session state so every
    later step (and a resume on another server) finds it, and a measured profile."""
    del node_input
    run = bind(ctx.state) if ctx.state.get(RUN_KEY) else new_run()
    yield Event(state={RUN_KEY: run.name})
    config = settings.load()
    report = profile(LAKE / config.dataset, config.target, config.positive_value)
    write_json(run / "reports" / "profile.json", report)
    brief = history.started(run, config, str(ctx.state.get(FEEDBACK_KEY) or ""))
    trace.note(
        trace.PIPELINE,
        f"● new run {run.name}: {config.dataset}, {report.get('rows')} rows profiled"
        + (
            f", builds on {brief['builds_on']}"
            if brief["builds_on"]
            else ", cold start"
        )
        + (
            f" · feedback: {brief['feedback'].splitlines()[0][:100]}"
            if brief["feedback"]
            else ""
        ),
        kind="step",
        step="start",
        **brief,
    )
    if reused := history.reuse_profile(run, config):
        trace.note(
            trace.PIPELINE,
            f"↺ no feedback and the same data as {reused}: its first look at the data "
            "is reused, and the team explores new features and models",
            kind="step",
            step="start",
            reused=reused,
        )
        yield Event(output=_request("features"), route="reuse")
        return
    yield Event(output=_request("profile"), route="profile")


def check(stage: str, next_stage: str | None):
    """Send the agent back while its hand-over is incomplete, then move on."""

    @node(name=f"check_{stage}", rerun_on_resume=False)
    async def check_node(ctx: Context, node_input: Any):
        del node_input
        bind(ctx.state)
        tries = ctx.state.get(f"fixes_{stage}", 0)
        issues = await contracts.problems(current(), stage)
        if stage != "profile" and not read_json(
            current() / "receipts" / f"{stage}.json"
        ):
            issues.append(NO_RECEIPT)
        # A round that fixed nothing would only repeat itself: stop sending it back.
        stuck = bool(issues) and issues == ctx.state.get(f"issues_{stage}")
        fix_rounds = settings.load().fix_rounds  # limits.fix_rounds
        if issues and tries < fix_rounds and not stuck:
            trace.note(
                trace.PIPELINE,
                f"✗ check {stage}: {len(issues)} problems, sent back · {issues[0][:100]}",
                kind="step",
                step=f"check_{stage}",
                problems=issues,
            )
            yield Event(state={f"fixes_{stage}": tries + 1, f"issues_{stage}": issues})
            yield Event(
                output=RECEIPT_ONLY
                if issues == [NO_RECEIPT]
                else _request(
                    stage, "Your hand-over is incomplete:\n- " + "\n- ".join(issues)
                ),
                route="fix",
            )
            return
        trace.note(
            trace.PIPELINE,
            f"✓ check {stage}: complete"
            if not issues
            else f"! check {stage}: the same {len(issues)} problems after a fix round, not sent back again"
            if stuck
            else f"! check {stage}: still {len(issues)} problems after {fix_rounds} fixes",
            kind="step",
            step=f"check_{stage}",
            problems=issues,
        )
        yield Event(state={f"fixes_{stage}": 0, f"issues_{stage}": None})
        write_json(current() / "reports" / f"check_{stage}.json", {"problems": issues})
        if next_stage:
            yield Event(output=_request(next_stage), route="next")
        else:
            note = f"\nThe completeness check still reports: {issues}" if issues else ""
            yield Event(
                output=f"Review the {stage} stage.\n{settings.load().brief()}{note}"
                + _direction("review")
                + _briefing(),
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


def _ask(
    key: str, message: str, stage: str, review_stage: str, why: str = ""
) -> RequestInput:
    why = why or _why(stage, review_stage)
    trace.note(
        trace.PIPELINE, f"✋ waiting for the human at {stage}: {why}", kind="pause"
    )
    return RequestInput(
        interrupt_id=key, message=message, payload={"stage": stage, "why": why}
    )


@node(name="review_features", rerun_on_resume=True)
async def review_features(ctx: Context, node_input: Any):
    """Sound features continue; concerns go back to the engineer; else the human decides."""
    del node_input
    bind(ctx.state)
    round_ = ctx.state.get("round_features", 0)
    key = f"review_features_{round_}"
    answer, by = (ctx.resume_inputs or {}).get(key), "human"
    if answer is None and _needs_review(ctx, "features", round_):
        for event in _remind("features", round_):
            yield event
        return
    state: dict[str, Any] = {"round_features": round_ + 1}
    if answer is None:
        fix = _self_review(ctx, "features")
        if fix:
            answer, by = {"choice": "feedback", **fix}, "team"
            state["team_rounds_features"] = fix["round"]
            state["team_asked_features"] = fix.pop("asked")
        elif "features" not in settings.load().ask_human and _sound("features"):
            answer, by = {"choice": "continue"}, "team"
        else:
            yield _ask(key, "Review the features", "features", "features")
            return
    choice = str(answer.get("choice", ""))
    instruction = _instruction("features", answer)
    blocked = _review("features")[1]
    if choice == "continue" and blocked:
        choice, instruction = (
            "feedback",
            "Fix these problems first:\n- " + "\n- ".join(blocked),
        )
    if choice == "discard":
        yield Event(state=state)
        _log("features", by, action="discard")
        yield Event(output="Discarded at the features review.", route="discard")
    elif choice == "feedback" and instruction:
        yield Event(state=state)
        _log("features", by, action="feedback", text=instruction)
        _reset("features")
        yield Event(output=_request("features", instruction), route="redo")
    else:
        yield Event(state={**state, "team_rounds_features": 0})
        view = settings.load().feature_view
        definition = feature_store.register(current(), view)
        ref = {
            "view": view,
            "version": definition["version"],
        }
        write_json(current() / "reports" / "feature_ref.json", ref)
        _log(
            "features",
            by,
            action="continue",
            feature_view=f"{view}:{definition['version']}",
        )
        yield Event(output="Confirm the training plan.", route="continue")


@node(name="plan_gate", rerun_on_resume=True)
async def plan_gate(ctx: Context, node_input: Any):
    """Which models to train and which metric decides: the team's proposal, or the human's."""
    del node_input
    bind(ctx.state)
    round_ = ctx.state.get("round_plan", 0)
    key = f"plan_{round_}"
    answer, by = (ctx.resume_inputs or {}).get(key), "human"
    proposal = read_json(current() / "reports" / "plan_proposal.json") or {}
    if answer is None:
        proposed_ok = proposal and not contracts.plan_problems(
            proposal.get("models") or [], proposal.get("metric") or ""
        )
        if "plan" not in settings.load().ask_human and proposed_ok:
            answer, by = {"choice": "train"}, "team"
        else:
            yield _ask(
                key,
                "Confirm the training plan",
                "plan",
                "features",
                ""
                if proposed_ok
                else "The engineer's proposed plan is missing or invalid; choose one.",
            )
            return
    yield Event(state={"round_plan": round_ + 1})
    if str(answer.get("choice")) == "discard":
        _log("plan", by, action="discard")
        yield Event(output="Discarded at the training plan.", route="discard")
        return
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
        "by": by,
    }
    write_json(current() / "reports" / "plan.json", plan)
    _log("plan", by, action="train", models=models, metric=metric, text=plan["note"])
    _reset("model")
    yield Event(output=_request("model"), route="train")


def _beats_production(model: str, evaluation: dict[str, Any]) -> bool:
    """Better than doing nothing and than the live version, on the plan's metric (test)."""
    metric = evaluation.get("metric") or "roc_auc"
    score = ((evaluation.get("candidates") or {}).get(model) or {}).get("test", {})
    baseline = (evaluation.get("baseline") or {}).get("test", {})
    if baseline.get(metric) is not None and not catalog.better(
        metric, score.get(metric), baseline.get(metric)
    ):
        return False
    live = registry.production()
    return not live or catalog.better(
        metric, score.get(metric), (live.get("test") or {}).get(metric)
    )


@node(name="promote_gate", rerun_on_resume=True)
async def promote_gate(ctx: Context, node_input: Any):
    """Concerns go back to the engineer; a sound model goes live if it beats production,
    or the human picks: promote, keep, retrain, change features, or replan."""
    del node_input
    bind(ctx.state)
    round_ = ctx.state.get("round_model", 0)
    key = f"promote_{round_}"
    answer, by = (ctx.resume_inputs or {}).get(key), "human"
    if answer is None and _needs_review(ctx, "model", round_):
        for event in _remind("model", round_):
            yield event
        return
    evaluation = read_json(current() / "reports" / "evaluation.json") or {}
    candidates = evaluation.get("candidates") or {}
    state: dict[str, Any] = {"round_model": round_ + 1}
    if answer is None:
        fix = _self_review(ctx, "model")
        if fix:
            answer, by = {"choice": "retrain", **fix}, "team"
            state["team_rounds_model"] = fix["round"]
            state["team_asked_model"] = fix.pop("asked")
        else:
            pick = _review("model")[0].get("recommended_model") or evaluation.get(
                "best"
            )
            # A fair-lending warning is never the team's call to wave through.
            unfair = fairness_flags(candidates.get(str(pick)) or {})
            if (
                "promote" not in settings.load().ask_human
                and _sound("model")
                and not unfair
            ):
                choice = (
                    "promote" if _beats_production(str(pick), evaluation) else "keep"
                )
                answer, by = {"choice": choice, "model": pick}, "team"
            else:
                why = (
                    f"Fair-lending check on {pick}: {unfair[0]} A human decides "
                    "whether it may go live."
                    if unfair
                    else ""
                )
                yield _ask(key, "Promote to production?", "promote", "model", why)
                return
    choice = str(answer.get("choice", ""))
    model = str(answer.get("model") or evaluation.get("best") or "")
    if choice in ("promote", "keep") and model not in candidates:
        choice = "retrain"
        answer = {
            "text": "No evaluated candidate could be registered; fix training so every planned model evaluates."
        }
    if choice in ("promote", "keep"):
        state["team_rounds_model"] = 0
    yield Event(state=state)
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
        _log("promote", by, action=choice, model=model, version=version["version"])
        state_ = "is in production" if choice == "promote" else "is kept as a candidate"
        yield Event(output=f"{version['version']} ({model}) {state_}.", route="done")
    elif choice == "retrain":
        instruction = _instruction("model", answer)
        _log("promote", by, action="retrain", text=instruction)
        _reset("model")
        yield Event(output=_request("model", instruction), route="retrain")
    elif choice == "features":
        instruction = _instruction("model", answer)
        _log("promote", by, action="features", text=instruction)
        _reset("model")
        _reset("features")
        yield Event(output=_request("features", instruction), route="features")
    elif choice == "replan":
        _log("promote", by, action="replan")
        yield Event(output="Change the training plan.", route="replan")
    else:
        _log("promote", by, action="discard")
        yield Event(output="Discarded at the promote review.", route="done")


@node(name="finish", rerun_on_resume=False)
async def finish(ctx: Context, node_input: Any):
    """Close the run in the run index, where the next run's team reads its lessons."""
    history.finished(bind(ctx.state), str(node_input))
    yield Event(output=str(node_input))


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
            # No feedback and unchanged data: the last run's first look still holds.
            (start, {"profile": analyst, "reuse": engineer_features}),
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
                    "features": engineer_features,
                    "replan": plan_gate,
                    "done": finish,
                },
            ),
        ],
    )


pipeline = build()
