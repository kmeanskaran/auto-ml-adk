"""Harness, feature store, registry and the pipeline graph, without a real model."""

import asyncio
import dataclasses
import json
import types
from collections.abc import AsyncGenerator
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from google.adk.apps import App, ResumabilityConfig
from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types as genai
from pydantic import Field
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app import pipeline, settings
from app.harness import (
    contracts,
    environment,
    evaluation,
    feature_store,
    history,
    profile,
    project,
    registry,
    scope,
    tools,
    trace,
)


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(project, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(environment, "LAKE", tmp_path / "lake")
    monkeypatch.setattr(
        project, "traffic_file", lambda: tmp_path / "traffic" / "2017-01.csv"
    )
    monkeypatch.setattr(feature_store, "STORE", tmp_path / "store")
    monkeypatch.setattr(registry, "REGISTRY", tmp_path / "registry")
    (tmp_path / "lake").mkdir()
    (tmp_path / "traffic").mkdir()
    raw(40, seed=9).drop(columns=["y"]).to_csv(
        tmp_path / "traffic" / "2017-01.csv", index=False
    )
    project.use("t")


def raw(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "id": [f"r{seed}-{i}" for i in range(n)],
            "x": x,
            "kind": rng.choice(["a", "b"], size=n),
            "y": (x + rng.normal(scale=0.8, size=n) > 0.3).astype(int),
        }
    )


def ctx(agent: str):
    return types.SimpleNamespace(agent_name=agent)


def write(path, content, agent="engineer_features"):
    return tools.write_file(path, content, ctx(agent))


def run(script, agent="engineer_features"):
    return asyncio.run(tools.run_python(script, ctx(agent)))


# --- tools -----------------------------------------------------------------------------


def test_each_agent_writes_only_what_it_owns():
    assert write("src/features.py", "x = 1\n")["status"] == "ok"
    assert write("src/features.py", "x = 1\n", "engineer_model")["status"] == "error"
    assert write("src/train.py", "x = 1\n", "engineer_model")["status"] == "ok"
    assert write("src/data.py", "x = 1\n", "skeptic_features")["status"] == "error"
    assert (
        write("checks/skeptic_leak.py", "x = 1\n", "skeptic_features")["status"] == "ok"
    )
    assert write("../escape.py", "x = 1\n")["status"] == "error"
    assert write("src/data.py", "import requests\n")["status"] == "error"


def test_code_runs_without_server_secrets(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    write(
        "checks/env.py",
        "import os, json\nprint(json.dumps({'key': os.environ.get('GEMINI_API_KEY')}))\n",
    )
    result = run("checks/env.py")
    assert result["status"] == "ok" and '"key": null' in result["stdout"]


def test_analyst_works_in_its_own_folder():
    tools.write_file("q1.py", "print(1)\n", ctx("analyst"))
    assert (project.RUNS / project.ANALYSIS / "q1.py").is_file()
    assert not (project.current() / "q1.py").exists()


def test_reports_are_short_and_validated():
    assert (
        tools.submit_receipt(["a", "b", "c", "d"], ctx("engineer_features"))["status"]
        == "error"
    )
    assert (
        tools.submit_receipt(["Split by date"], ctx("engineer_features"))["status"]
        == "ok"
    )
    assert (
        tools.propose_plan(["magic_model"], "roc_auc", "r", ctx("engineer_features"))[
            "status"
        ]
        == "error"
    )
    assert (
        tools.propose_plan(
            ["logistic_regression"], "vibes", "r", ctx("engineer_features")
        )["status"]
        == "error"
    )
    good = tools.propose_plan(
        ["logistic_regression"], "pr_auc", "imbalanced", ctx("engineer_features")
    )
    assert good["proposal"]["metric"] == "pr_auc"
    review = tools.submit_review(
        "concerns", ["Leak: AUC 0.99"], ["Remove z"], ctx("skeptic_features")
    )
    assert review["review"]["recommendations"] == ["Remove z"]
    assert (
        tools.submit_review(
            "pass", ["ok"], [], ctx("skeptic_model"), recommended_model="nope"
        )["status"]
        == "error"
    )


def test_overlong_lines_are_sent_back_to_be_rewritten_not_cut():
    long = "x" * 200
    receipt = tools.submit_receipt([long], ctx("engineer_features"))
    assert receipt["status"] == "error" and "Rewrite shorter" in receipt["message"]
    review = tools.submit_review(
        "concerns", ["short"], ["Remove `x`: " + long], ctx("skeptic_features")
    )
    assert review["status"] == "error" and "under 110 characters" in review["message"]


def test_charts_are_validated_from_files_the_script_wrote():
    folder = project.folder(project.ANALYSIS) / "charts"
    folder.mkdir(exist_ok=True)
    spec = {
        "type": "bar",
        "title": "Rate",
        "categories": ["a", "b"],
        "series": [{"name": "rate", "values": [0.1, 0.2]}],
    }
    (folder / "ok.json").write_text(json.dumps(spec))
    assert tools.show_chart("charts/ok.json", ctx("analyst"))["chart"][
        "categories"
    ] == ["a", "b"]
    (folder / "bad.json").write_text(
        json.dumps({**spec, "series": [{"name": "r", "values": [1]}]})
    )
    assert tools.show_chart("charts/bad.json", ctx("analyst"))["status"] == "error"


def test_profile_measures_columns_and_signal(tmp_path):
    frame = raw()
    frame["after"] = frame["y"]  # a column that already knows the outcome
    frame.to_csv(tmp_path / "lake" / "d.csv", index=False)
    report = profile.profile(tmp_path / "lake" / "d.csv", "y")
    by_name = {c["name"]: c for c in report["column_profiles"]}
    assert report["rows"] == 400 and by_name["id"]["kind"] == "identifier"
    assert by_name["after"]["signal_auc"] == 1.0 and by_name["x"]["signal_auc"] > 0.7
    assert report["charts"][0]["categories"][0] == "after"


def test_profile_maps_a_text_outcome_with_its_positive_value(tmp_path):
    frame = raw()
    frame["status"] = frame["y"].map({1: "Y", 0: "N"})
    frame.drop(columns=["y"]).to_csv(tmp_path / "lake" / "d.csv", index=False)
    report = profile.profile(tmp_path / "lake" / "d.csv", "status", "Y")
    assert report["target"]["positives"] == int(frame["y"].sum())
    assert report["target"]["unlabelled"] == 0
    assert {c["name"]: c for c in report["column_profiles"]}["x"]["signal_auc"] > 0.7


# --- feature store, evaluation and registry ------------------------------------------------

FEATURES_REPORT = {
    "outcome_column": "y",
    "entity_key": "id",
    "split": {"method": "by id", "reason": "test", "ordered_by": "id"},
    "rows_removed": {},
    "excluded_columns": {},
    "features": {
        "x": {"source": ["x"], "description": "x", "known_at_prediction": "yes"},
        "x_sq": {
            "source": ["x"],
            "description": "x squared",
            "known_at_prediction": "yes",
        },
        "is_a": {
            "source": ["kind"],
            "description": "kind is a",
            "known_at_prediction": "yes",
        },
    },
}
ROW_WISE = (
    "import pandas as pd\n"
    "def build(raw):\n"
    "    return pd.DataFrame({'x': raw['x'], 'x_sq': raw['x'] ** 2, 'is_a': (raw['kind'] == 'a').astype(int)})\n"
)


def features_stage(source=ROW_WISE, report=None):
    root = project.current()
    frame = raw()
    for name, part in zip(
        ("train", "valid", "test"),
        (frame[:200], frame[200:300], frame[300:]),
        strict=True,
    ):
        part.to_parquet(root / "artifacts" / f"{name}.parquet")
    (root / "reports" / "features.json").write_text(
        json.dumps(report or FEATURES_REPORT)
    )
    tools.propose_plan(
        ["logistic_regression"], "roc_auc", "r", ctx("engineer_features")
    )
    write("src/features.py", source)
    return root


def test_features_must_be_row_wise_and_serve_production_records():
    features_stage(
        "import pandas as pd\n"
        "def build(raw):\n"
        "    x = raw['x']\n"
        "    return pd.DataFrame({'x': x - x.mean(), 'x_sq': x ** 2, 'is_a': (raw['kind'] == 'a').astype(int)})\n"
    )
    issues = asyncio.run(contracts.problems(project.current(), "features"))
    assert any("not row-wise" in i for i in issues)
    features_stage()
    text = pd.read_parquet(project.current() / "artifacts" / "train.parquet")
    text["y"] = text["y"].map({1: "Y", 0: "N"})  # the outcome left unencoded
    text.to_parquet(project.current() / "artifacts" / "train.parquet")
    issues = asyncio.run(contracts.problems(project.current(), "features"))
    assert any("must hold only 0 and 1" in i for i in issues)
    features_stage()
    assert asyncio.run(contracts.problems(project.current(), "features")) == []
    stats = project.read_json(project.current() / "reports" / "feature_stats.json")
    assert stats["features"]["x"]["signal_auc"] > 0.7


def trained_run(report=None):
    """A run with approved features in the store and one trained model."""
    root = features_stage(report=report)
    assert asyncio.run(contracts.problems(root, "features")) == []
    definition = feature_store.register(root, "view")
    assert (
        feature_store.register(root, "view")["version"] == definition["version"] == "v1"
    )
    ref = {
        "view": "view",
        "version": "v1",
        "path": str(feature_store.path("view", "v1")),
    }
    project.write_json(root / "reports" / "feature_ref.json", ref)
    project.write_json(
        root / "reports" / "plan.json",
        {"models": ["logistic_regression"], "metric": "pr_auc"},
    )
    train = pd.read_parquet(
        feature_store.path("view", "v1") / "offline" / "train.parquet"
    )
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(
        train[["x", "x_sq", "is_a"]], train["y"]
    )
    (root / "artifacts" / "models").mkdir()
    joblib.dump(model, root / "artifacts" / "models" / "logistic_regression.joblib")
    return root, ref


def test_harness_evaluates_every_candidate_and_serving_bundle_works():
    root, _ = trained_run()
    assert asyncio.run(contracts.problems(root, "model")) == []
    report = project.read_json(root / "reports" / "evaluation.json")
    scores = report["candidates"]["logistic_regression"]
    assert report["best"] == "logistic_regression" and report["metric"] == "pr_auc"
    assert report["chosen_on"] == "valid"  # a time-ordered split: compared on valid
    assert set(scores["test"]) == set(contracts.catalog.METRICS)
    assert scores["valid"]["roc_auc"] > report["baseline"]["valid"]["roc_auc"]


def test_harness_refits_so_training_on_valid_cannot_inflate_validation():
    root, _ = trained_run()
    path = feature_store.path("view", "v1") / "offline"
    cols = ["x", "x_sq", "is_a"]
    train = pd.read_parquet(path / "train.parquet")
    valid = pd.read_parquet(path / "valid.parquet")
    both = pd.concat([train, valid])
    # the engineer hands over a model that has already seen the validation rows
    joblib.dump(
        make_pipeline(StandardScaler(), LogisticRegression()).fit(
            both[cols], both["y"]
        ),
        root / "artifacts" / "models" / "logistic_regression.joblib",
    )
    assert asyncio.run(contracts.problems(root, "model")) == []
    scores = project.read_json(root / "reports" / "evaluation.json")["candidates"][
        "logistic_regression"
    ]
    honest = make_pipeline(StandardScaler(), LogisticRegression()).fit(
        train[cols], train["y"]
    )
    from sklearn.metrics import roc_auc_score

    expected = roc_auc_score(valid["y"], honest.predict_proba(valid[cols])[:, 1])
    assert scores["valid"]["roc_auc"] == round(expected, 4)
    assert set(scores["calibration"]) == {"valid", "test"}
    assert scores["drift"][0]["feature"] in cols
    assert (root / "artifacts" / "final" / "logistic_regression.joblib").is_file()


def test_the_split_must_say_whether_it_follows_time():
    report = {**FEATURES_REPORT, "split": {"method": "random", "reason": "r"}}
    features_stage(report=report)
    issues = asyncio.run(contracts.problems(project.current(), "features"))
    assert any("split.ordered_by is required" in i for i in issues)


def test_few_unordered_rows_are_compared_by_cross_validation():
    unordered = {
        **FEATURES_REPORT,
        "split": {**FEATURES_REPORT["split"], "ordered_by": None},
    }
    root, _ = trained_run(unordered)
    assert asyncio.run(contracts.problems(root, "model")) == []
    report = project.read_json(root / "reports" / "evaluation.json")
    assert report["chosen_on"] == "cv5" and report["selection"]["rows"] == 300
    assert "5-fold cross-validation over the 300" in report["protocol"]


def test_drift_is_only_a_warning_when_the_split_follows_time():
    scores = {
        "valid": {"roc_auc": 0.8},
        "test": {"roc_auc": 0.8},
        "calibration": {"test": {"mean_score": 0.3, "outcome_rate": 0.3}},
        "drift": [{"feature": "income", "score_shift": 0.2}],
    }
    assert evaluation.warnings(scores, "roc_auc", ordered=False) == []
    assert "income" in evaluation.warnings(scores, "roc_auc", ordered=True)[0]


def test_the_analyst_hands_over_the_charts_it_chose():
    folder = project.current() / "charts"
    spec = {
        "type": "stacked",
        "title": "Outcome by kind",
        "categories": ["a", "b"],
        "series": [
            {"name": "yes", "values": [60, 30]},
            {"name": "no", "values": [40, 70]},
        ],
    }
    (folder / "by_kind.json").write_text(json.dumps(spec))
    ok = tools.submit_summary(
        "400 rows", ["kind matters"], ctx("analyst_profile"), ["charts/by_kind.json"]
    )
    assert ok["summary"]["charts"][0]["title"] == "Outcome by kind"
    bad = tools.submit_summary(
        "400 rows", ["x"], ctx("analyst_profile"), ["charts/none.json"]
    )
    assert bad["status"] == "error"


def test_warnings_speak_plainly_about_calibration_gaps_and_drift():
    scores = {
        "valid": {"roc_auc": 0.86},
        "test": {"roc_auc": 0.80},
        "calibration": {"test": {"mean_score": 0.22, "outcome_rate": 0.37}},
        "drift": [{"feature": "income", "score_shift": 0.075}],
    }
    found = evaluation.warnings(scores, "roc_auc")
    assert "predicts a 22% approval rate, but the real rate was 37%" in found[0]
    assert "underestimates the risk, so it flags too few" in found[0]
    assert "0.860 when compared, 0.800 on the final test" in found[1]
    assert found[2].endswith("hold up: income.")
    fine = {**scores, "test": {"roc_auc": 0.85}, "drift": []}
    fine["calibration"] = {"test": {"mean_score": 0.36, "outcome_rate": 0.37}}
    assert evaluation.warnings(fine, "roc_auc") == []


def test_registry_versions_promotes_idempotently_and_rolls_back():
    root, ref = trained_run()
    asyncio.run(contracts.problems(root, "model"))
    report = project.read_json(root / "reports" / "evaluation.json")
    path = feature_store.path("view", "v1")
    with pytest.raises(LookupError):
        registry.predict([{"id": "a"}])
    first = registry.register(
        root, "logistic_regression", report, ref, path, {"metric": "pr_auc"}
    )
    assert (
        registry.register(root, "logistic_regression", report, ref, path, {})["version"]
        == "v1"
    )
    assert first["status"] == "candidate" and first["feature_view"] == ref
    registry.promote("v1")
    registry.promote("v1")  # a resumed workflow promoting again changes nothing
    assert [e["event"] for e in registry.audit()] == ["promoted", "register"]
    rows = raw(3, seed=5).drop(columns=["y"]).to_dict("records")
    served = registry.predict(rows)
    assert served["version"] == "v1" and [p["id"] for p in served["predictions"]] == [
        "r5-0",
        "r5-1",
        "r5-2",
    ]
    # a second trained candidate becomes v2; rolling back to v1 archives it
    joblib.dump(
        make_pipeline(StandardScaler(), LogisticRegression(C=0.01)).fit(
            pd.read_parquet(path / "offline" / "train.parquet")[["x", "x_sq", "is_a"]],
            pd.read_parquet(path / "offline" / "train.parquet")["y"],
        ),
        root / "artifacts" / "models" / "logistic_regression.joblib",
    )
    asyncio.run(contracts.problems(root, "model"))  # the harness refits the new model
    report = project.read_json(root / "reports" / "evaluation.json")
    registry.register(root, "logistic_regression", report, ref, path, {})
    registry.promote("v2")
    registry.promote("v1", reason="rollback")
    assert {v["version"]: v["status"] for v in registry.versions()} == {
        "v2": "archived",
        "v1": "production",
    }
    assert registry.predict(rows, "v2")["version"] == "v2"
    # removing: never the live version; a removed one can't be served or made live
    with pytest.raises(ValueError):
        registry.remove("v1")
    registry.remove("v2")
    with pytest.raises(registry.NotFound):
        registry.predict(rows, "v2")
    with pytest.raises(registry.NotFound):
        registry.promote("v2")
    assert registry.predict(rows)["version"] == "v1"


# --- the pipeline graph --------------------------------------------------------------------


class Scripted(BaseLlm):
    """Plays each agent with a fixed tool call, and remembers what each was asked."""

    seen: list = Field(default_factory=list)
    verdicts: list = Field(
        default_factory=list
    )  # the skeptic's, in order; then concerns

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        del stream
        last = llm_request.contents[-1] if llm_request.contents else None
        if last and any(p.function_response for p in last.parts):
            yield LlmResponse(
                content=genai.Content(role="model", parts=[genai.Part(text="done")])
            )
            return
        self.seen.append("".join(p.text or "" for p in (last.parts if last else [])))
        role = str(llm_request.config.system_instruction or "")
        if "Your role: analyst" in role:
            calls = [
                ("submit_summary", {"headline": "400 rows", "findings": ["x ranks y"]})
            ]
        elif "Your role: skeptic" in role:
            verdict = self.verdicts.pop(0) if self.verdicts else "concerns"
            calls = [
                (
                    "submit_review",
                    {
                        "verdict": verdict,
                        "findings": ["x leaks" if verdict == "concerns" else "sound"],
                        "recommendations": ["Remove x"]
                        if verdict == "concerns"
                        else [],
                    },
                )
            ]
        elif "Build the features stage" in self.seen[-1]:
            calls = [
                (
                    "propose_plan",
                    {
                        "models": ["logistic_regression"],
                        "metric": "pr_auc",
                        "reason": "r",
                    },
                ),
                ("submit_receipt", {"findings": ["3 features"]}),
            ]
        else:
            calls = [("submit_receipt", {"findings": ["trained"]})]
        parts = [
            genai.Part(function_call=genai.FunctionCall(name=n, args=a))
            for n, a in calls
        ]
        yield LlmResponse(content=genai.Content(role="model", parts=parts))


def autonomy(monkeypatch, ask_human=settings.REVIEWS, rounds=0):
    """Pin how much the team decides alone, whatever config/pipeline.yml says."""
    config = settings.load()
    monkeypatch.setattr(
        settings,
        "load",
        lambda: dataclasses.replace(
            config, ask_human=tuple(ask_human), self_review_rounds=rounds
        ),
    )


def scripted_stages(monkeypatch, ask_human=settings.REVIEWS, rounds=0):
    """Stub the harness around the agents so the workflow can run on a scripted model."""
    autonomy(monkeypatch, ask_human, rounds)

    async def complete(root, stage):
        if stage == "model":
            project.write_json(
                root / "reports" / "evaluation.json",
                {
                    "metric": "pr_auc",
                    "best": "logistic_regression",
                    "candidates": {"logistic_regression": {}},
                },
            )
        return []

    promoted = []
    monkeypatch.setattr(pipeline, "profile", lambda *args: {"rows": 1})
    monkeypatch.setattr(pipeline.contracts, "problems", complete)
    monkeypatch.setattr(
        pipeline.feature_store, "register", lambda run, view: {"version": "v1"}
    )
    monkeypatch.setattr(pipeline.registry, "register", lambda *a: {"version": "v9"})
    monkeypatch.setattr(pipeline.registry, "promote", promoted.append)

    return promoted


@pytest.mark.parametrize("backend", ["memory", "sqlite-restart"])
def test_pipeline_walks_features_plan_and_promote_with_human_answers(
    monkeypatch, tmp_path, backend
):
    """Every human answer resumes the paused workflow. With sqlite-restart, each answer
    goes to a brand-new Runner and session service over the same database file, the
    way a server restarted between reviews would see it."""
    promoted = scripted_stages(monkeypatch)
    model = Scripted(model="scripted")
    app = App(
        name="t",
        root_agent=pipeline.build(model),
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    memory = InMemorySessionService()

    def fresh_runner():
        if backend == "memory":
            return Runner(app=app, session_service=memory)
        return Runner(
            app=app, session_service=SqliteSessionService(str(tmp_path / "s.db"))
        )

    async def scenario():
        session = await fresh_runner().session_service.create_session(
            app_name="t", user_id="u"
        )

        async def send(message, invocation=None):
            pause = None
            runner = fresh_runner()
            async for event in runner.run_async(
                user_id="u",
                session_id=session.id,
                new_message=message,
                invocation_id=invocation,
            ):
                for call in event.get_function_calls():
                    if call.name == "adk_request_input":
                        pause = (
                            call.id,
                            event.invocation_id,
                            call.args.get("payload", {}),
                        )
            return pause

        async def answer(pause, **reply):
            part = genai.Part(
                function_response=genai.FunctionResponse(
                    id=pause[0], name="adk_request_input", response=reply
                )
            )
            return await send(genai.Content(role="user", parts=[part]), pause[1])

        pause = await send(genai.Content(role="user", parts=[genai.Part(text="run")]))
        assert pause[2]["stage"] == "features"
        assert any("first look at the data" in s for s in model.seen)
        pause = await answer(
            pause, choice="feedback", apply=[0], text="Add lead-time buckets"
        )
        assert pause[2]["stage"] == "features"
        assert any("- Remove x\n- Add lead-time buckets" in s for s in model.seen)
        pause = await answer(pause, choice="continue")
        assert pause[2]["stage"] == "plan"
        pause = await answer(
            pause, choice="train", models=["logistic_regression"], metric="roc_auc"
        )
        assert pause[2]["stage"] == "promote"
        assert any("metric roc_auc" in s for s in model.seen)
        pause = await answer(pause, choice="retrain", text="Tune C")
        assert pause[2]["stage"] == "promote"
        pause = await answer(pause, choice="features", text="Drop drifting features")
        assert pause[2]["stage"] == "features"
        assert any("Drop drifting features" in s for s in model.seen)
        pause = await answer(pause, choice="continue")
        assert pause[2]["stage"] == "plan"
        pause = await answer(
            pause, choice="train", models=["logistic_regression"], metric="roc_auc"
        )
        assert pause[2]["stage"] == "promote"
        assert (
            await answer(pause, choice="promote", model="logistic_regression") is None
        )

    asyncio.run(scenario())
    decisions = [
        json.loads(line)
        for line in (project.current() / "decisions.jsonl").read_text().splitlines()
    ]
    assert [d["action"] for d in decisions] == [
        "feedback",
        "continue",
        "train",
        "retrain",
        "features",
        "continue",
        "train",
        "promote",
    ]
    assert promoted == ["v9"] and decisions[-1]["version"] == "v9"


def test_console_resumes_an_open_review_after_a_server_restart(monkeypatch, tmp_path):
    from app.ui import runs

    scripted_stages(monkeypatch)
    app = App(
        name="t",
        root_agent=pipeline.build(Scripted(model="scripted")),
        resumability_config=ResumabilityConfig(is_resumable=True),
    )

    def fresh_runner():
        return Runner(
            app=app, session_service=SqliteSessionService(str(tmp_path / "s.db"))
        )

    async def scenario():
        before = runs.PipelineRun(runner=fresh_runner())
        await before.start()
        await before.task
        assert (
            before.status == "waiting" and before.pause.payload["stage"] == "features"
        )

        project.use("somewhere-else")  # a new process knows nothing
        after = runs.PipelineRun.restore()
        assert after.status == "waiting" and after.project == before.project
        assert project.current_name() == before.project
        after.runner = fresh_runner()
        await after.answer({"choice": "continue"})
        await after.task
        assert after.status == "waiting" and after.pause.payload["stage"] == "plan"

        after.status = "running"  # killed mid-stage: reported, never silently resumed
        after._save()
        crashed = runs.PipelineRun.restore()
        assert crashed.status == "error" and "restarted" in crashed.error

    asyncio.run(scenario())


# --- the team decides; the human is asked only when needed ---------------------------------


class Session:
    """Drives one pipeline session: send a message, get the pause it stops at (or None)."""

    def __init__(self, model):
        self.runner = Runner(
            app=App(
                name="t",
                root_agent=pipeline.build(model),
                resumability_config=ResumabilityConfig(is_resumable=True),
                plugins=[scope.RunScope(), trace.TracePlugin()],
            ),
            session_service=InMemorySessionService(),
        )
        self.id = ""

    async def send(self, message, invocation=None):
        if not self.id:
            session = await self.runner.session_service.create_session(
                app_name="t", user_id="u"
            )
            self.id = session.id
        pause = None
        async for event in self.runner.run_async(
            user_id="u",
            session_id=self.id,
            new_message=message,
            invocation_id=invocation,
        ):
            for call in event.get_function_calls():
                if call.name == "adk_request_input":
                    pause = (call.id, event.invocation_id, call.args.get("payload", {}))
        return pause

    async def start(self):
        return await self.send(
            genai.Content(role="user", parts=[genai.Part(text="run")])
        )

    async def answer(self, pause, **reply):
        part = genai.Part(
            function_response=genai.FunctionResponse(
                id=pause[0], name="adk_request_input", response=reply
            )
        )
        return await self.send(genai.Content(role="user", parts=[part]), pause[1])


def decisions():
    path = project.current() / "decisions.jsonl"
    return [
        (d["stage"], d["action"], d["by"])
        for d in map(json.loads, path.read_text().splitlines())
    ]


def test_team_settles_the_skeptic_itself_and_asks_only_to_go_live(monkeypatch):
    scripted_stages(monkeypatch, ask_human=["promote"], rounds=1)
    model = Scripted(model="scripted", verdicts=["concerns", "pass", "pass"])
    session = Session(model)

    async def scenario():
        pause = await session.start()
        assert pause[2]["stage"] == "promote" and "Your call" in pause[2]["why"]
        assert await session.answer(pause, choice="keep") is None

    asyncio.run(scenario())
    # the skeptic's recommendation went to the engineer without asking anyone
    assert any("- Remove x" in s for s in model.seen)
    assert decisions() == [
        ("features", "feedback", "team"),
        ("features", "continue", "team"),
        ("plan", "train", "team"),
        ("promote", "keep", "human"),
    ]
    log = "\n".join(trace.activity(project.current()))
    assert "engineer_features" in log and "▶ started" in log and "■ finished" in log
    assert "◆ team decided at features: feedback" in log
    assert "✋ waiting for the human at promote" in log


def test_concerns_the_team_cannot_settle_go_to_the_human(monkeypatch):
    scripted_stages(monkeypatch, ask_human=[], rounds=1)
    session = Session(Scripted(model="scripted"))  # the skeptic never passes
    pause = asyncio.run(session.start())
    assert pause[2]["stage"] == "features"
    assert "could not settle" in pause[2]["why"]
    assert decisions() == [("features", "feedback", "team")]


def test_a_fully_autonomous_team_puts_a_sound_model_live(monkeypatch):
    promoted = scripted_stages(monkeypatch, ask_human=[], rounds=1)
    session = Session(Scripted(model="scripted", verdicts=["pass", "pass"]))
    assert asyncio.run(session.start()) is None
    assert promoted == ["v9"]
    assert [d[2] for d in decisions()] == ["team", "team", "team"]


def test_trace_keeps_every_version_of_the_code_an_agent_writes():
    plugin = trace.TracePlugin()
    tool = types.SimpleNamespace(name="write_file")

    def written(content, call):
        context = types.SimpleNamespace(
            agent_name="engineer_features", invocation_id="i", function_call_id=call
        )
        args = {"path": "src/data.py", "content": content}
        result = tools.write_file(args["path"], content, context)
        asyncio.run(
            plugin.before_tool_callback(tool=tool, tool_args=args, tool_context=context)
        )
        asyncio.run(
            plugin.after_tool_callback(
                tool=tool, tool_args=args, tool_context=context, result=result
            )
        )

    written('"""Load."""\nx = 1\n', "c1")
    written('"""Load, fixed."""\nx = 2\n', "c2")
    code = sorted((project.current() / trace.CODE).iterdir())
    assert [c.name for c in code] == [
        "001-engineer_features-src__data.py",
        "002-engineer_features-src__data.py",
    ]
    assert code[0].read_text().endswith("x = 1\n")
    records = [
        json.loads(line)
        for line in (project.current() / trace.TRACE).read_text().splitlines()
    ]
    assert records[-1]["tool"] == "write_file"
    assert records[-1]["args"]["content"]["chars"] == len('"""Load, fixed."""\nx = 2\n')
    assert records[-1]["snapshot"].endswith("002-engineer_features-src__data.py")
    assert "wrote src/data.py (3 lines)" in trace.activity(project.current())[-1]


# --- one run per session -----------------------------------------------------------------


def test_sessions_side_by_side_each_keep_their_own_run(monkeypatch):
    """The run lives in the session state, so two sessions in one process never share
    a folder, and each one's decisions land in its own run."""
    scripted_stages(monkeypatch, ask_human=["features"], rounds=0)
    first = Session(Scripted(model="scripted", verdicts=["pass"]))
    second = Session(Scripted(model="scripted", verdicts=["pass"]))

    async def scenario():
        return await asyncio.gather(first.start(), second.start())

    pauses = asyncio.run(scenario())
    assert [p[2]["stage"] for p in pauses] == ["features", "features"]

    async def state(session):
        stored = await session.runner.session_service.get_session(
            app_name="t", user_id="u", session_id=session.id
        )
        return stored.state[project.RUN_KEY]

    runs = [asyncio.run(state(first)), asyncio.run(state(second))]
    assert runs[0] != runs[1]
    for run in runs:
        assert (project.RUNS / run / "reports" / "profile.json").is_file()
        assert (project.RUNS / run / "reviews" / "features.json").is_file()
        assert "◆" not in "".join(
            trace.activity(project.RUNS / run)
        )  # nothing decided yet


# --- fewer tool calls ------------------------------------------------------------------------


def turn(agent="engineer_features"):
    context = types.SimpleNamespace(agent_name=agent, invocation_id="i")
    tools.start_turn(context)
    return context


def test_an_unchanged_file_is_not_sent_twice_in_one_turn():
    context = turn()
    tools.write_file("src/data.py", '"""Load."""\nx = 1\n', context)
    assert tools.read_file("src/data.py", context)["unchanged"] is True
    (project.current() / "src" / "data.py").write_text("x = 2\n")
    assert tools.read_file("src/data.py", context)["text"] == "x = 2\n"
    assert tools.read_file("src/data.py", context)["unchanged"] is True
    tools.start_turn(context)  # a new activation has a fresh context: send it again
    assert tools.read_file("src/data.py", context)["text"] == "x = 2\n"
    tools.end_turn(context)


def test_write_and_run_and_search_save_calls():
    context = turn()
    result = asyncio.run(
        tools.write_and_run(
            "checks/sig.py", '"""Signal."""\nprint(\'{"auc": 0.7}\')\n', context
        )
    )
    assert result["status"] == "ok" and result["path"] == "checks/sig.py"
    assert '"auc": 0.7' in result["stdout"]
    found = tools.search(r"auc", context)
    assert found["matches"][0]["path"] == "checks/sig.py"
    assert tools.search("(", context)["status"] == "error"
    hint = asyncio.run(tools.run_python("import os\nprint(1)", context))
    assert "write_and_run" in hint["message"]


def test_every_overlong_line_is_named_at_once():
    review = tools.submit_review(
        "concerns",
        ["f" * 150],
        ["r" * 120, "ok", "s" * 130],
        ctx("skeptic_features"),
    )
    message = review["message"]
    assert "150 chars, cut 10" in message
    assert "120 chars, cut 10" in message and "130 chars, cut 20" in message


def test_the_tool_budget_lets_an_agent_only_hand_over_at_the_end():
    context = turn()
    read = types.SimpleNamespace(name="read_file")
    submit = types.SimpleNamespace(name="submit_receipt")
    for _ in range(tools.SOFT_BUDGET):
        assert tools.budget(read, {}, context) is None
    reminded = tools.budget_reminder(read, {}, context, {"status": "ok"})
    assert "wrap up" in reminded["budget"]
    for _ in range(tools.HARD_BUDGET - tools.SOFT_BUDGET):
        tools.budget(read, {}, context)
    assert "budget spent" in tools.budget(read, {}, context)["message"]
    assert tools.budget(submit, {}, context) is None


def test_long_turns_drop_superseded_copies_but_keep_the_newest():
    def call(i, name, **args):
        return genai.Content(
            role="model",
            parts=[
                genai.Part(
                    function_call=genai.FunctionCall(id=f"c{i}", name=name, args=args)
                )
            ],
        )

    def answer(i, name, **response):
        return genai.Content(
            role="user",
            parts=[
                genai.Part(
                    function_response=genai.FunctionResponse(
                        id=f"c{i}", name=name, response=response
                    )
                )
            ],
        )

    big = "x" * (tools.COMPACT_ABOVE_CHARS // 3)
    contents = [
        genai.Content(
            role="user", parts=[genai.Part(text="Build the features stage.")]
        ),
        call(1, "read_file", path="src/features.py"),
        answer(1, "read_file", status="ok", text=big),
        call(2, "write_file", path="src/features.py", content=big),
        answer(2, "write_file", status="ok", path="src/features.py"),
        call(3, "run_python", script="src/data.py"),
        answer(3, "run_python", status="ok", stdout=big),
        call(4, "run_python", script="src/data.py"),
        answer(4, "run_python", status="ok", stdout="latest"),
        call(5, "read_file", path="src/features.py"),
        answer(5, "read_file", status="ok", text="newest"),
    ]
    session = list(contents)  # the session's own objects, which must stay whole
    request = LlmRequest(contents=contents)
    tools.compact(None, request)
    out = request.contents
    assert out[2].parts[0].function_response.response["text"] == "[elided]"
    assert "elided" in out[3].parts[0].function_call.args["content"]
    assert out[6].parts[0].function_response.response["stdout"] == "[elided]"
    assert out[8].parts[0].function_response.response["stdout"] == "latest"
    assert out[10].parts[0].function_response.response["text"] == "newest"
    assert session[2].parts[0].function_response.response["text"] == big
    assert session[3].parts[0].function_call.args["content"] == big
    small = LlmRequest(contents=session[:3])
    tools.compact(None, small)  # short turns are left exactly as they are
    assert small.contents[2].parts[0].function_response.response["text"] == big


# --- fair lending ----------------------------------------------------------------------------


def test_fair_lending_gaps_are_plain_warnings():
    fair = {
        "fairness": {
            "Gender": [
                {"group": "Female", "rows": 40, "flag_rate": 0.70, "recall": 0.90},
                {"group": "Male", "rows": 160, "flag_rate": 0.75, "recall": 0.92},
            ]
        }
    }
    unfair = {
        "fairness": {
            "Gender": [
                {"group": "Female", "rows": 40, "flag_rate": 0.40, "recall": 0.60},
                {"group": "Male", "rows": 160, "flag_rate": 0.75, "recall": 0.92},
                {"group": "(missing)", "rows": 5, "flag_rate": 0.0, "recall": None},
            ]
        }
    }
    assert evaluation.fairness_flags(fair) == []
    flags = evaluation.fairness_flags(unfair)
    assert len(flags) == 2  # the 5-row group is too small to judge
    assert "Female" in flags[0] and "ratio 0.53" in flags[0]
    assert "60%" in flags[1] and "92%" in flags[1]


def test_an_unfair_model_never_goes_live_without_a_human(monkeypatch):
    promoted = scripted_stages(monkeypatch, ask_human=[], rounds=0)
    monkeypatch.setattr(
        pipeline,
        "fairness_flags",
        lambda scores: [
            "Fair lending, `Gender`: Female get an approval 40% of the time."
        ],
    )
    session = Session(Scripted(model="scripted", verdicts=["pass", "pass"]))
    pause = asyncio.run(session.start())
    assert pause[2]["stage"] == "promote" and "Fair-lending check" in pause[2]["why"]
    assert promoted == []


# --- the team learns from past runs ----------------------------------------------------------


def test_finished_runs_brief_the_next_one(monkeypatch):
    promoted = scripted_stages(monkeypatch, ask_human=[], rounds=0)
    first = Session(Scripted(model="scripted", verdicts=["pass", "pass"]))
    assert asyncio.run(first.start()) is None and promoted == ["v9"]
    done = history.runs()
    assert len(done) == 1 and done[0]["version"] == "v9"
    assert done[0]["decisions"][-1] == "promote:promote by team"

    model = Scripted(model="scripted", verdicts=["pass", "pass"])
    asyncio.run(Session(model).start())
    briefed = next(s for s in model.seen if "Build the features stage" in s)
    assert "Past runs on this dataset" in briefed and done[0]["run"] in briefed
    assert "Hypotheses to test, not findings" in briefed


# --- feature store listing and the analyst's kit -------------------------------------------


def test_the_feature_store_lists_every_feature_of_every_version():
    run = project.current()
    feature_store_dir = run / feature_store.STAGED
    feature_store_dir.mkdir(parents=True, exist_ok=True)
    (run / "src").mkdir(exist_ok=True)
    (run / "src" / "features.py").write_text("def build(raw):\n    return raw\n")
    for split in feature_store.SPLITS:
        raw(20).to_parquet(feature_store_dir / f"{split}.parquet")
    project.write_json(
        run / "reports" / "features.json",
        {
            "outcome_column": "y",
            "features": {
                "x": {"source": ["x"], "description": "x as is"},
                "kind_a": {"source": ["kind"], "description": "kind is a"},
            },
        },
    )
    feature_store.register(run, "view")
    listed = feature_store.summary()[0]["versions"][0]
    assert [c["name"] for c in listed["columns"]] == ["x", "kind_a"]
    assert listed["columns"][1]["source"] == ["kind"]


def test_the_kit_reports_rates_with_intervals_and_real_gaps():
    from app.harness import analyst_kit as kit

    df = raw(400).rename(columns={"y": "outcome"})
    rates = kit.rate_by(df, "kind")
    assert set(rates["group"]) == {"a", "b"}
    assert (rates["low"] <= rates["rate"]).all() and (
        rates["rate"] <= rates["high"]
    ).all()
    binned = kit.rate_by(df, "x")
    assert len(binned) == 5 and binned["rows"].sum() == 400
    assert binned["rate"].iloc[-1] > binned["rate"].iloc[0]  # x drives y
    assert kit.compare(df, "x")["p_value"] < 0.05
    assert kit.compare(df, "kind")["effect"] in ("negligible", "small")
    moved = kit.drift(df, df.assign(x=df["x"] + 3))
    assert moved.iloc[0]["column"] == "x" and moved.iloc[0]["level"] == "large"
    assert "id" not in set(moved["column"])  # identifiers are not drift


def test_a_new_run_rethinks_the_last_runs_code_with_the_humans_feedback(monkeypatch):
    from app.ui import runs

    scripted_stages(monkeypatch, ask_human=[], rounds=0)
    model = Scripted(model="scripted", verdicts=["pass"] * 4)
    app = App(
        name="t",
        root_agent=pipeline.build(model),
        resumability_config=ResumabilityConfig(is_resumable=True),
        plugins=[scope.RunScope()],
    )
    console = runs.PipelineRun(
        runner=Runner(app=app, session_service=InMemorySessionService())
    )

    async def scenario():
        async def run(feedback=""):
            await console.start(feedback)
            await console.task
            assert console.status == "done", console.error
            return project.RUNS / console.project

        first = await run()
        (first / "src" / "data.py").write_text("# the first run's split\n")
        second = await run("Try income per household member.")
        return first, second

    first, second = asyncio.run(scenario())
    assert project.read_json(first / history.BRIEF)["builds_on"] is None  # cold start
    brief = project.read_json(second / history.BRIEF)
    assert brief["builds_on"] == first.name
    assert brief["feedback"] == "Try income per household member."
    assert (second / "prior" / "src" / "data.py").read_text().startswith("# the first")
    features_request = next(
        s for s in reversed(model.seen) if "Build the features stage" in s
    )
    assert "Try income per household member." in features_request
    assert f"builds on {first.name}" in features_request
    assert history.runs()[0]["feedback"] == "Try income per household member."
    # an agent may read prior/ but never write it
    assert (
        tools.write_file("prior/src/data.py", "x", ctx("engineer_features"))["status"]
        == "error"
    )


def test_clear_is_a_cold_start(monkeypatch):
    from app.ui import runs

    service = InMemorySessionService()
    monkeypatch.setattr(runs.services, "get_session_service", lambda: service)
    app = App(name="t", root_agent=pipeline.build(Scripted(model="scripted")))
    console = runs.PipelineRun(
        runner=Runner(app=app, session_service=service), status="done", project="r1"
    )
    chat = runs.AnalystChat(messages=[{"role": "you", "text": "hi"}])
    monkeypatch.setattr(runs, "PIPELINE", console)
    monkeypatch.setattr(runs, "CHAT", chat)
    for folder in (
        project.RUNS / "r1" / "src",
        registry.REGISTRY / "v1",
        feature_store.STORE / "view" / "v1",
    ):
        folder.mkdir(parents=True)
        (folder / "file").write_text("x")

    async def scenario():
        for name in ("t", "analyst"):
            await service.create_session(app_name=name, user_id=runs.USER)
        console.status = "running"
        with pytest.raises(RuntimeError):
            await runs.clear()
        console.status = "waiting"  # an open review is abandoned
        await runs.clear()
        return [
            (await service.list_sessions(app_name=n, user_id=runs.USER)).sessions
            for n in ("t", "analyst")
        ]

    assert asyncio.run(scenario()) == [[], []]
    for folder in (project.RUNS, registry.REGISTRY, feature_store.STORE):
        assert folder.is_dir() and not any(folder.iterdir())
    assert (console.status, console.project, chat.messages) == ("idle", "", [])
    assert history.runs() == []


def test_build_must_not_read_project_files_it_is_served_alone():
    features_stage(
        "import json\n"
        "import pandas as pd\n"
        "def build(raw):\n"
        "    names = list(json.load(open('reports/features.json'))['features'])\n"
        "    x = raw['x']\n"
        "    return pd.DataFrame({'x': x, 'x_sq': x ** 2, 'is_a': (raw['kind'] == 'a').astype(int)})[names]\n"
    )
    issues = asyncio.run(contracts.problems(project.current(), "features"))
    assert any("served alone" in i for i in issues)


def test_a_registered_feature_version_is_frozen_and_a_damaged_one_skipped():
    trained_run()
    folder = feature_store.path("view", "v1")
    with pytest.raises(PermissionError):  # what a training script writing there gets
        (folder / "definition.json").write_text("{}")
    with pytest.raises(PermissionError):
        (folder / "notes.json").write_text("{}")
    folder.chmod(0o755)  # damaged by hand: the console skips it instead of failing
    (folder / "definition.json").chmod(0o644)
    (folder / "definition.json").write_text('{"features": ["x"]}')
    assert feature_store.versions("view") == []
    assert feature_store.summary() == [{"view": "view", "versions": []}]
    project.empty(feature_store.STORE)  # a cold start removes frozen versions too
    assert not any(feature_store.STORE.iterdir())


def test_charts_the_analyst_drew_are_shown_even_without_show_chart():
    from app.ui import runs

    started = __import__("time").time() - 1
    folder = project.folder(project.ANALYSIS)
    spec = {
        "type": "bar",
        "title": "t",
        "categories": ["a", "b"],
        "series": [{"name": "rate", "values": [1, 2]}],
    }
    (folder / "charts" / "rate.json").write_text(json.dumps(spec))
    (folder / "charts" / "broken.json").write_text("{}")
    assert [c["title"] for c in runs._drawn_since(started)] == ["t"]
    assert runs._drawn_since(__import__("time").time() + 60) == []


def test_old_script_outputs_keep_only_their_summary_when_the_turn_is_long():
    noise = "row\n" * (tools.COMPACT_ABOVE_CHARS // 20)
    contents = [genai.Content(role="user", parts=[genai.Part(text="Build it.")])]
    for i in range(7):
        args = {"script": f"checks/c{i}.py"}
        contents += [
            genai.Content(
                role="model",
                parts=[
                    genai.Part(
                        function_call=genai.FunctionCall(
                            id=f"r{i}", name="run_python", args=args
                        )
                    )
                ],
            ),
            genai.Content(
                role="user",
                parts=[
                    genai.Part(
                        function_response=genai.FunctionResponse(
                            id=f"r{i}",
                            name="run_python",
                            response={
                                "status": "ok",
                                "stdout": f'{noise}{{"auc": 0.{i}}}',
                            },
                        )
                    )
                ],
            ),
        ]
    request = LlmRequest(contents=list(contents))
    tools.compact(None, request)
    outputs = [c.parts[0].function_response.response for c in request.contents[2::2]]
    assert [o["stdout"] for o in outputs[:3]] == [
        '{"auc": 0.0}',
        '{"auc": 0.1}',
        '{"auc": 0.2}',
    ]
    assert all(
        o["stdout"].startswith(noise) for o in outputs[3:]
    )  # the newest four stay
    assert contents[2].parts[0].function_response.response["stdout"].startswith(noise)


def test_an_unchanged_hand_over_is_not_checked_twice(monkeypatch):
    root = features_stage()
    checked = []
    real = contracts._problems

    async def counting(root, stage):
        checked.append(stage)
        return await real(root, stage)

    monkeypatch.setattr(contracts, "_problems", counting)
    assert asyncio.run(contracts.problems(root, "features")) == []
    result = asyncio.run(tools.check_stage(ctx("engineer_features")))
    assert result["complete"] and "Nothing changed" in result["unchanged"]
    assert checked == ["features"]
    (root / "reports" / "feature_stats.json").unlink()  # an output removed: check again
    asyncio.run(contracts.problems(root, "features"))
    write("src/features.py", ROW_WISE + "\n# edited\n")  # the code changed: check again
    asyncio.run(contracts.problems(root, "features"))
    assert checked == ["features"] * 3


def test_a_fix_round_that_changes_nothing_is_not_repeated(monkeypatch):
    scripted_stages(monkeypatch, ask_human=[], rounds=0)

    async def same(root, stage):
        return ["artifacts/test.parquet does not exist"] if stage == "features" else []

    monkeypatch.setattr(pipeline.contracts, "problems", same)
    model = Scripted(model="scripted", verdicts=["pass"])
    pause = asyncio.run(Session(model).start())
    assert pause[2]["stage"] == "features"
    # the first try and one fix; a second fix would only repeat the first
    assert sum("Build the features stage" in s for s in model.seen) == 2
    assert "the same 1 problems" in "\n".join(trace.activity(project.current()))


def test_a_skeptic_repeating_itself_goes_to_the_human_not_another_round(monkeypatch):
    scripted_stages(monkeypatch, ask_human=[], rounds=3)
    pause = asyncio.run(
        Session(Scripted(model="scripted")).start()
    )  # always "Remove x"
    assert pause[2]["stage"] == "features"
    assert decisions() == [("features", "feedback", "team")]


def test_a_chart_named_with_its_extension_lands_where_the_agent_expects(
    monkeypatch, tmp_path
):
    from app.harness import analyst_kit

    monkeypatch.chdir(tmp_path)
    for name in ("rate", "rate.json", "charts/rate.json"):
        assert analyst_kit.chart(name, {"a": 0.2, "b": 0.4}) == "charts/rate.json"
    chart, issue = tools._chart(tmp_path, "charts/missing.json")
    assert not chart and "charts/rate.json" in issue and "chart() returned" in issue


def test_a_missing_receipt_alone_asks_for_the_receipt_not_the_stage(monkeypatch):
    scripted_stages(monkeypatch, ask_human=["features"], rounds=0)
    model = Scripted(model="scripted", verdicts=["pass"])
    original = Scripted.generate_content_async
    skipped = []

    async def forgetful(self, llm_request, stream=False):
        async for response in original(self, llm_request, stream):
            calls = response.content.parts if response.content else []
            if (
                self.seen
                and "Build the features stage" in self.seen[-1]
                and not skipped
            ):
                skipped.append(1)  # first features turn: the work, but no receipt
                response.content.parts = [
                    p
                    for p in calls
                    if not (
                        p.function_call and p.function_call.name == "submit_receipt"
                    )
                ]
            yield response

    monkeypatch.setattr(Scripted, "generate_content_async", forgetful)
    pause = asyncio.run(Session(model).start())
    assert pause[2]["stage"] == "features"
    assert sum(s == pipeline.RECEIPT_ONLY for s in model.seen) == 1
    assert sum("Build the features stage" in s for s in model.seen) == 1


def test_an_unchanged_file_asked_for_twice_is_sent():
    write("src/data.py", "x = 1\n")
    context = ctx("engineer_features")
    tools.start_turn(context)
    assert tools.read_file("src/data.py", context)["text"] == "x = 1\n"
    assert tools.read_file("src/data.py", context)["unchanged"]
    assert tools.read_file("src/data.py", context)["text"] == "x = 1\n"


def test_analyst_answers_are_plain_markdown_without_math_markup():
    from app.ui.runs import clean

    latex = (
        "cut\u2011off at\u202f0.667:\n\\[\n\\text{threshold} = "
        "\\frac{\\text{cost}_{FP}}{\\text{cost}_{FP}+\\text{cost}_{FN}} = \\frac{2}{2+1}\n\\]\n"
        "| cost | \\$422 (421 \\times 1.0) |\nwhen $x \\geq 3$, $5 and $7, 50\\% here"
    )
    assert clean(latex) == (
        "cut-off at 0.667:\n\nthreshold = cost_FP / (cost_FP+cost_FN) = 2 / (2+1)\n\n"
        "| cost | $422 (421 \u00d7 1.0) |\nwhen x \u2265 3, $5 and $7, 50% here"
    )
    plain = "**68.7%** approved\n\n### Income\n| a | b |\n|---|---|\n| 1 | 2 |"
    assert clean(plain) == plain


def test_kit_charts_accept_what_the_analyst_writes(monkeypatch, tmp_path):
    from app.harness import analyst_kit

    monkeypatch.chdir(tmp_path)
    spec = {
        "type": "bar",
        "title": "t",
        "categories": ["a"],
        "series": [{"name": "s", "values": [1]}],
    }
    assert analyst_kit.chart("whole", spec) == "charts/whole.json"  # a whole spec
    assert (
        analyst_kit.show_chart("rates", {"a": 0.5}) == "charts/rates.json"
    )  # imported by mistake
    for name in ("whole", "rates"):
        assert tools._chart(tmp_path, f"charts/{name}.json")[1] == ""


def test_search_takes_a_file_as_well_as_a_folder():
    write("src/data.py", "def load():\n    pass\n")
    context = ctx("engineer_features")
    assert tools.search("def load", context, "src/data.py")["matches"][0]["line"] == 1
    assert len(tools.search("def load", context, "src")["matches"]) == 1


def console_run(monkeypatch, verdicts, ask_human=()):
    from app.ui import runs

    scripted_stages(monkeypatch, ask_human=list(ask_human), rounds=0)
    model = Scripted(model="scripted", verdicts=list(verdicts))
    app = App(
        name="t",
        root_agent=pipeline.build(model),
        resumability_config=ResumabilityConfig(is_resumable=True),
        plugins=[scope.RunScope()],
    )
    return runs.PipelineRun(
        runner=Runner(app=app, session_service=InMemorySessionService())
    ), model


def test_pause_holds_the_run_until_it_is_resumed(monkeypatch):
    console, model = console_run(monkeypatch, ["pass", "pass"])

    async def scenario():
        await console.start()
        console.pause_run()
        await asyncio.sleep(0.3)
        held = len(model.seen)
        await asyncio.sleep(0.3)
        assert console.status == "paused" and len(model.seen) == held <= 1
        with pytest.raises(RuntimeError):
            console.pause_run()  # already paused
        console.resume_run()
        await console.task
        assert console.status == "done" and len(model.seen) > held

    asyncio.run(scenario())


def test_restart_stops_the_run_and_starts_again_with_the_same_feedback(monkeypatch):
    console, _ = console_run(monkeypatch, [], ask_human=["features"])

    async def scenario():
        await console.start("Try income per member.")
        await console.task  # waiting at the features review
        first = console.project
        await console.restart()
        await console.task
        return first, console.project

    first, second = asyncio.run(scenario())
    assert first != second and console.status == "waiting"
    brief = project.read_json(project.RUNS / second / history.BRIEF)
    assert brief["feedback"] == "Try income per member."


def test_the_chosen_model_and_thinking_level_go_into_every_request(
    monkeypatch, tmp_path
):
    from app.harness import models

    monkeypatch.setattr(models, "STATE", tmp_path / "model.json")
    monkeypatch.delenv("ML_MODEL", raising=False)  # config models.team decides
    assert models.current() == {"model": "gemini-3.7-flash", "thinking": "low"}
    models.choose("gemini-3.8-flash", "high")  # the console's menu
    team, chat = (
        SimpleNamespace(agent_name="engineer_model"),
        SimpleNamespace(agent_name="analyst"),
    )
    request = LlmRequest(model="gemini-3.7-flash")
    models.apply(team, request)
    assert request.model == "gemini-3.8-flash"
    assert request.config.thinking_config.thinking_level == genai.ThinkingLevel.HIGH
    answer = LlmRequest(model="gemini-3.7-flash")  # the chat analyst keeps its own
    models.apply(chat, answer)
    assert answer.model == "gemini-3.5-flash"
    assert answer.config.thinking_config.thinking_level == genai.ThinkingLevel.LOW
    with pytest.raises(ValueError):
        models.choose("gemini-1.0-pro", "low")
    monkeypatch.setenv(
        "ML_MODEL", "ollama_chat/gpt-oss:120b-cloud"
    )  # not Gemini: untouched
    other = LlmRequest(model="ollama_chat/gpt-oss:120b-cloud")
    models.apply(team, other)
    assert (
        other.model == "ollama_chat/gpt-oss:120b-cloud"
        and models.view()["available"] is False
    )


def test_limits_come_from_the_config(monkeypatch):
    config = settings.load()
    assert (config.tool_budget, config.fix_rounds, config.compact_above_chars) == (
        45,
        2,
        50_000,
    )
    assert tools.budgets() == (tools.SOFT_BUDGET, tools.HARD_BUDGET)
    monkeypatch.setattr(
        settings, "load", lambda: dataclasses.replace(config, tool_budget=12)
    )
    assert tools.budgets() == (8, 12)
