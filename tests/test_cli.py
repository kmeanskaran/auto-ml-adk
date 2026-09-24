"""The runtime CLI is what the bash scripts call."""

from __future__ import annotations

from runtime.cli import main


def test_cli_run_prints_a_completed_experiment(tmp_path, monkeypatch, capsys) -> None:
  monkeypatch.setenv("ML_DATA_ROOT", str(tmp_path))
  monkeypatch.setenv("ML_RUNTIME_MODEL", "scripted")
  code = main(["run", "--target", "0.0", "--no-approve"])
  captured = capsys.readouterr()
  assert code == 0, captured.out
  assert "status: completed" in captured.out
  assert "experiment:" in captured.out
