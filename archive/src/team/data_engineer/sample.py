"""Synthetic churn table used when the caller does not upload a dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from team.config import load_config


def make_churn_frame(rows: int = 400, seed: int = 7) -> pd.DataFrame:
  """Build a binary churn table with a recoverable signal and a 0 majority."""
  rng = np.random.default_rng(seed)
  tenure = rng.integers(1, 72, rows)
  monthly = rng.normal(70, 22, rows).clip(20, 160)
  support = rng.poisson(1.4, rows)
  contract = rng.choice(
      ["month-to-month", "one-year", "two-year"],
      rows,
      p=[0.55, 0.25, 0.20],
  )
  internet = rng.choice(["dsl", "fiber", "none"], rows, p=[0.34, 0.46, 0.20])
  senior = rng.integers(0, 2, rows)
  paperless = rng.integers(0, 2, rows)
  total = tenure * monthly * rng.uniform(0.9, 1.1, rows)
  logit = (
      -2.3
      + 0.055 * (monthly - 70)
      + 0.5 * support
      - 0.038 * tenure
      + np.where(contract == "month-to-month", 1.45, 0.0)
      + np.where(contract == "two-year", -0.7, 0.0)
      + np.where(internet == "fiber", 0.85, 0.0)
      + 0.35 * senior
  )
  probability = 1.0 / (1.0 + np.exp(-logit))
  churn = (rng.random(rows) < probability).astype(int)
  return pd.DataFrame(
      {
          "customer_id": [f"C{index:04d}" for index in range(rows)],
          "tenure_months": tenure.astype(int),
          "monthly_charges": np.round(monthly, 2),
          "total_charges": np.round(total, 2),
          "contract": contract,
          "internet": internet,
          "support_calls": support.astype(int),
          "senior": senior.astype(int),
          "paperless": paperless.astype(int),
          "churn": churn.astype(int),
      }
  )


def write_churn_csv(path: Path, rows: int | None = None, seed: int | None = None) -> Path:
  sample = load_config().sample_data
  frame = make_churn_frame(
      rows=sample.rows if rows is None else rows,
      seed=sample.seed if seed is None else seed,
  )
  path.parent.mkdir(parents=True, exist_ok=True)
  frame.to_csv(path, index=False)
  return path


def inject_missing(
    frame: pd.DataFrame,
    rate: float | None = None,
    columns: tuple[str, ...] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
  """Blank random cells so the quality agent has something to find."""
  sample = load_config().sample_data
  rate = sample.missing_rate if rate is None else rate
  columns = sample.missing_columns if columns is None else columns
  seed = sample.missing_seed if seed is None else seed
  rng = np.random.default_rng(seed)
  out = frame.copy()
  for column in columns:
    if column not in out.columns:
      continue
    mask = rng.random(len(out)) < rate
    out.loc[mask, column] = np.nan
  return out
