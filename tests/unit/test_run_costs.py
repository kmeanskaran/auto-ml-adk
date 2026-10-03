"""Fewer wasted steps, and what a run cost.

A script in checks/ imports the run's own src/ (it used to fail with "No module named
'src'"), and usage() totals a run's time, tokens, cache share and tool calls.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from app.harness.environment import ProjectEnvironment, cpus
from app.harness.trace import TRACE, usage


def test_a_check_script_imports_the_runs_own_code(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "features.py").write_text("NAME = 'from src'\n")
    (tmp_path / "checks").mkdir()
    (tmp_path / "checks" / "check.py").write_text(
        "from src.features import NAME\nprint(NAME)\n"
    )
    result = asyncio.run(
        ProjectEnvironment(tmp_path).execute(f"{sys.executable} checks/check.py")
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip() == "from src"


def test_scripts_are_told_the_real_cpu_count():
    assert cpus() >= 1


def test_usage_totals_a_run(tmp_path: Path):
    lines = [
        {"at": "2026-10-03T08:40:00", "agent": "pipeline", "summary": "▶ started"},
        {
            "at": "2026-10-03T08:41:00",
            "agent": "analyst_profile",
            "kind": "agent",
            "event": "end",
            "seconds": 60.0,
            "tool_calls": 6,
            "tokens": 40_000,
            "cached_tokens": 16_000,
        },
        {
            "at": "2026-10-03T08:52:00",
            "agent": "engineer_features",
            "kind": "agent",
            "event": "end",
            "seconds": 600.0,
            "tool_calls": 20,
            "tokens": 60_000,
            "cached_tokens": 24_000,
        },
    ]
    (tmp_path / TRACE).parent.mkdir(parents=True)
    (tmp_path / TRACE).write_text("\n".join(json.dumps(x) for x in lines) + "\n")

    u = usage(tmp_path)
    assert u["minutes"] == 12.0  # 08:40 to 08:52, waits included
    assert u["agent_minutes"] == 11.0
    assert (u["tokens"], u["cached_tokens"], u["cached_pct"]) == (100_000, 40_000, 40)
    assert u["tool_calls"] == 26
    assert [t["agent"] for t in u["turns"]] == ["analyst_profile", "engineer_features"]


def test_usage_of_a_run_without_a_trace(tmp_path: Path):
    assert usage(tmp_path)["tokens"] == 0
