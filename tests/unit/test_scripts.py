"""The shell scripts parse, and the teardown script changes nothing without --apply."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[2] / "scripts").glob("*.sh"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_parses(script: Path):
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_has_help_text(script: Path):
    assert script.read_text().startswith("#!/usr/bin/env bash\n# ")


def test_nuke_is_a_dry_run_unless_applied():
    script = (SCRIPTS[0].parent / "nuke_gcp.sh").read_text()
    assert "APPLY=false" in script
    assert "if ! $APPLY; then" in script
    assert 'read -r -p "Type the project ID to confirm: "' in script
