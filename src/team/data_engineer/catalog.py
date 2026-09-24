"""Named sample tables the SME can pick instead of uploading a file."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from team.config import PROJECT_ROOT
from team.data_engineer.sample import write_churn_csv

DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"

SAMPLES: dict[str, dict[str, Any]] = {
    "churn": {
        "id": "churn",
        "title": "Customers",
        "file": None,
        "target_column": "churn",
        "metric": "f1",
        "target_value": 0.75,
        "objective": "Which customers will cancel?",
        "rows_hint": 400,
    },
    "breast_cancer": {
        "id": "breast_cancer",
        "title": "Breast cancer",
        "file": "breast_cancer.csv",
        "target_column": "diagnosis",
        "metric": "f1",
        "target_value": 0.9,
        "objective": "Predict tumor diagnosis (malignant vs benign). Optimize for F1.",
        "rows_hint": 569,
    },
    "iris": {
        "id": "iris",
        "title": "Iris",
        "file": "iris.csv",
        "target_column": "species",
        "metric": "f1",
        "target_value": 0.9,
        "objective": "Predict iris species from sepal and petal measurements. Optimize for F1.",
        "rows_hint": 150,
    },
}


def list_samples() -> list[dict[str, Any]]:
  return [
      {
          "id": item["id"],
          "title": item["title"],
          "target_column": item["target_column"],
          "metric": item["metric"],
          "target_value": item["target_value"],
          "objective": item["objective"],
          "rows_hint": item["rows_hint"],
      }
      for item in SAMPLES.values()
  ]


def get_sample(sample_id: str) -> dict[str, Any]:
  sample = SAMPLES.get(sample_id)
  if sample is None:
    known = ", ".join(SAMPLES)
    raise KeyError(f"Unknown sample {sample_id!r}. Choose one of: {known}.")
  return sample


def write_sample(sample_id: str, destination: Path) -> dict[str, Any]:
  sample = get_sample(sample_id)
  destination.parent.mkdir(parents=True, exist_ok=True)
  if sample["id"] == "churn":
    write_churn_csv(destination)
    return sample
  source = DATASETS_DIR / str(sample["file"])
  if not source.is_file():
    raise FileNotFoundError(f"Sample file missing: {source}")
  shutil.copyfile(source, destination)
  return sample
