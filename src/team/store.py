"""Filesystem source of truth for one ML experiment.

ADK session state is convenient inside a run. This store is what survives a
crash: phase results, the chosen artifact, and the approval flag.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import threading
from pathlib import Path
from typing import Any
from typing import Callable

_LOCKS: dict[str, threading.RLock] = {}
_LOCK_GUARD = threading.Lock()


def _lock_for(root: Path) -> threading.RLock:
  key = str(root.resolve())
  with _LOCK_GUARD:
    lock = _LOCKS.get(key)
    if lock is None:
      lock = threading.RLock()
      _LOCKS[key] = lock
    return lock


def jsonable(value: Any) -> Any:
  """Convert numpy and pandas scalars into JSON values."""
  if value is None or isinstance(value, (str, bool, int)):
    return value
  if isinstance(value, float):
    if math.isnan(value) or math.isinf(value):
      return None
    return value
  if isinstance(value, dict):
    return {str(key): jsonable(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [jsonable(item) for item in value]
  if hasattr(value, "item") and callable(value.item):
    try:
      return jsonable(value.item())
    except Exception:
      return str(value)
  return str(value)


class ExperimentStore:
  """One JSON document per experiment, plus artifact files beside it."""

  def __init__(self, root: Path):
    self.root = Path(root)
    self.root.mkdir(parents=True, exist_ok=True)
    self._lock = _lock_for(self.root)

  def experiment_dir(self, experiment_id: str) -> Path:
    return self.root / "experiments" / experiment_id

  def record_path(self, experiment_id: str) -> Path:
    return self.experiment_dir(experiment_id) / "experiment.json"

  def create(self, record: dict[str, Any]) -> dict[str, Any]:
    experiment_id = record["id"]
    directory = self.experiment_dir(experiment_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "checkpoints").mkdir(exist_ok=True)
    with self._lock:
      path = self.record_path(experiment_id)
      if path.exists():
        raise FileExistsError(f"Experiment {experiment_id} already exists.")
      self._write(record)
    return self.load(experiment_id)

  def load(self, experiment_id: str) -> dict[str, Any]:
    with self._lock:
      return self._read(experiment_id)

  def save(self, record: dict[str, Any]) -> None:
    with self._lock:
      self._write(record)

  def update(
      self,
      experiment_id: str,
      mutator: Callable[[dict[str, Any]], None],
  ) -> dict[str, Any]:
    """Read, mutate, and write one experiment under the store lock."""
    with self._lock:
      record = self._read(experiment_id)
      mutator(record)
      self._write(record)
      return copy.deepcopy(record)

  def list_ids(self) -> list[str]:
    directory = self.root / "experiments"
    if not directory.exists():
      return []
    return sorted(path.name for path in directory.iterdir() if path.is_dir())

  def clear(self) -> int:
    """Delete every experiment directory, including runs waiting for review."""
    directory = self.root / "experiments"
    with self._lock:
      if not directory.exists():
        return 0
      removed = 0
      for path in list(directory.iterdir()):
        if not path.is_dir():
          continue
        shutil.rmtree(path)
        removed += 1
      return removed

  def _read(self, experiment_id: str) -> dict[str, Any]:
    path = self.record_path(experiment_id)
    if not path.exists():
      raise FileNotFoundError(f"Experiment {experiment_id} was not found.")
    return json.loads(path.read_text(encoding="utf-8"))

  def _write(self, record: dict[str, Any]) -> None:
    path = self.record_path(record["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(jsonable(record), indent=2, sort_keys=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
