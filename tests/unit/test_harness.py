"""Harness, feature store, registry and the pipeline graph, without a real model."""

import asyncio
import json
import types
from collections.abc import AsyncGenerator

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
from google.genai import types as genai
from pydantic import Field
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app import pipeline
from app.harness import (
    contracts,
    environment,
    evaluation,
    feature_store,
    profile,
    project,
    registry,
    tools,
)


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(project, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(environment, "LAKE", tmp_path / "lake")
    monkeypatch.setattr(feature_store, "TRAFFIC", tmp_path / "traffic")
    monkeypatch.setattr(evaluation, "TRAFFIC", tmp_path / "traffic")
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


# --- feature store, evaluation and registry ------------------------------------------------

FEATURES_REPORT = {
    "outcome_column": "y",
    "entity_key": "id",
    "split": {"method": "random", "reason": "test"},
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


def features_stage(source=ROW_WISE):
    root = project.current()
    frame = raw()
    for name, part in zip(
        ("train", "valid", "test"),
        (frame[:200], frame[200:300], frame[300:]),
        strict=True,
    ):
        part.to_parquet(root / "artifacts" / f"{name}.parquet")
    (root / "reports" / "features.json").write_text(json.dumps(FEATURES_REPORT))
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
    assert asyncio.run(contracts.problems(project.current(), "features")) == []
    stats = project.read_json(project.current() / "reports" / "feature_stats.json")
    assert stats["features"]["x"]["signal_auc"] > 0.7


def trained_run():
    """A run with approved features in the store and one trained model."""
    root = features_stage()
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
    assert set(scores["test"]) == set(contracts.catalog.METRICS)
    assert scores["valid"]["roc_auc"] > report["baseline"]["valid"]["roc_auc"]


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
            calls = [
                (
                    "submit_review",
                    {
                        "verdict": "concerns",
                        "findings": ["x leaks"],
                        "recommendations": ["Remove x"],
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


def test_pipeline_walks_features_plan_and_promote_with_human_answers(monkeypatch):
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
    monkeypatch.setattr(pipeline, "profile", lambda path, target: {"rows": 1})
    monkeypatch.setattr(pipeline.contracts, "problems", complete)
    monkeypatch.setattr(
        pipeline.feature_store, "register", lambda run, view: {"version": "v1"}
    )
    monkeypatch.setattr(pipeline.registry, "register", lambda *a: {"version": "v9"})
    monkeypatch.setattr(pipeline.registry, "promote", promoted.append)

    model = Scripted(model="scripted")
    runner = Runner(
        app=App(
            name="t",
            root_agent=pipeline.build(model),
            resumability_config=ResumabilityConfig(is_resumable=True),
        ),
        session_service=InMemorySessionService(),
    )

    async def scenario():
        session = await runner.session_service.create_session(app_name="t", user_id="u")

        async def send(message, invocation=None):
            pause = None
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
        "promote",
    ]
    assert promoted == ["v9"] and decisions[-1]["version"] == "v9"
