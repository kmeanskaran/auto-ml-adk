"""A rerun with no feedback on unchanged data reuses the analyst's first look.

Anything else (feedback, changed data, no earlier run) looks at the data again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import settings
from app.harness import history, project
from app.harness.project import file_hash


@pytest.fixture
def world(tmp_path: Path, monkeypatch):
    config = settings.load()
    lake, runs = tmp_path / "lake", tmp_path / "runs"
    (lake / config.dataset).parent.mkdir(parents=True)
    (lake / config.dataset).write_text("Loan_ID,Loan_Status\nLP1,Y\n")
    monkeypatch.setattr(project, "LAKE", lake)
    monkeypatch.setattr(project, "RUNS", runs)

    prior = runs / "run-1"
    for name, text in {
        "reports/summary.json": '{"headline": "614 applications"}',
        "notes/analyst.md": "notes",
        "charts/approval_by_credit.json": "{}",
        "checks/analyst_first_look.py": "print(1)",
        "src/train.py": "not part of the first look",
    }.items():
        (prior / name).parent.mkdir(parents=True, exist_ok=True)
        (prior / name).write_text(text)
    record = {
        "run": "run-1",
        "event": "finished",
        "dataset": config.dataset,
        "data": file_hash(lake, config.dataset),
        "models": ["logistic_regression", "xgboost"],
        "best": "xgboost",
    }
    (runs / project.INDEX).write_text(json.dumps(record) + "\n")
    return config, lake, runs


def new_run(runs: Path, feedback: str = "") -> Path:
    run = runs / "run-2"
    project.write_json(
        run / history.BRIEF, {"feedback": feedback, "builds_on": "run-1"}
    )
    return run


def test_no_feedback_and_same_data_reuses_the_first_look(world):
    config, _, runs = world
    run = new_run(runs)

    assert history.reuse_profile(run, config) == "run-1"
    assert json.loads((run / "reports/summary.json").read_text())["headline"]
    assert (run / "charts/approval_by_credit.json").is_file()
    assert (run / "checks/analyst_first_look.py").is_file()
    assert not (run / "src").exists()  # only the first look, not the code
    reused = json.loads((run / history.REUSED).read_text())
    assert reused["models"] == ["logistic_regression", "xgboost"]
    assert reused["best"] == "xgboost"


def test_feedback_means_a_fresh_look(world):
    config, _, runs = world
    assert history.reuse_profile(new_run(runs, "drop Gender"), config) is None


def test_changed_data_means_a_fresh_look(world):
    config, lake, runs = world
    (lake / config.dataset).write_text("Loan_ID,Loan_Status\nLP1,N\n")
    assert history.reuse_profile(new_run(runs), config) is None
