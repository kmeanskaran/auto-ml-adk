"""Write the files that sit beside experiment.json.

The experiment record stays in the store. This saver owns the other files
for one run: the column configuration, the cleaned table, reports, and models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from team.store import jsonable


class ArtifactSaver:
  """Atomic writes scoped to one experiment directory."""

  def __init__(self, directory: Path):
    self.directory = Path(directory)
    self.directory.mkdir(parents=True, exist_ok=True)

  def save_feature_pipeline(self, spec: dict[str, Any]) -> Path:
    """Persist the column configuration the trainer and predict path share."""
    return self.save_json("feature_pipeline.json", spec)

  def load_feature_pipeline(self) -> dict[str, Any]:
    return self.load_json("feature_pipeline.json")

  def save_clean_table(self, frame: pd.DataFrame) -> Path:
    return self.save_table("clean.csv", frame)

  def save_statistical_report(self, text: str) -> Path:
    return self.save_text("statistical_report.md", text)

  def save_design(self, text: str) -> Path:
    return self.save_text("DESIGN.md", text)

  def save_model(self, model: Any, name: str = "model.joblib") -> Path:
    path = self._path(name)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(model, temporary)
    temporary.replace(path)
    return path

  def save_json(self, name: str, payload: dict[str, Any]) -> Path:
    path = self._path(name)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path

  def load_json(self, name: str) -> dict[str, Any]:
    path = self._path(name)
    if not path.is_file():
      raise FileNotFoundError(f"Artifact {name} was not found in {self.directory}.")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
      raise ValueError(f"Artifact {name} is not a JSON object.")
    return loaded

  def save_text(self, name: str, text: str) -> Path:
    path = self._path(name)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path

  def save_table(self, name: str, frame: pd.DataFrame) -> Path:
    path = self._path(name)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return path

  def _path(self, name: str) -> Path:
    if not name or Path(name).name != name:
      raise ValueError(f"Artifact name must be a file name, got {name!r}.")
    return self.directory / name
