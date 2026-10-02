"""Split the loan applications into what the team sees and later production traffic.

Run once, before any agent:

    uv run python scripts/make_environment.py

- data/lake/loan_applications.csv    labelled applications (Loan_Status Y/N), every column
- data/traffic/applications-001.csv  new applications as they reach the model when the
                                     form is submitted: same columns, no Loan_Status

The source is the Loan Prediction dataset (data/datasets/lending-loan): train.csv is
labelled, test.csv is not, so it plays the part of production traffic. Nothing is
cleaned or encoded here: that is the team's job.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data"
SOURCE = ROOT / "datasets" / "lending-loan"
LAKE = ROOT / "lake"
TRAFFIC = ROOT / "traffic"
LABEL = "Loan_Status"


def read(path: Path) -> pd.DataFrame:
    """The raw file as text, without the unnamed index column some exports carry."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frame.loc[:, ~frame.columns.str.startswith("Unnamed")]


def main() -> None:
    labelled, unlabelled = read(SOURCE / "train.csv"), read(SOURCE / "test.csv")
    missing = set(labelled.columns) - {LABEL} - set(unlabelled.columns)
    if missing:
        raise SystemExit(
            f"test.csv lacks columns the model will need: {sorted(missing)}"
        )

    for folder in (LAKE, TRAFFIC):
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
    labelled.to_csv(LAKE / "loan_applications.csv", index=False)
    unlabelled[[c for c in labelled.columns if c != LABEL]].to_csv(
        TRAFFIC / "applications-001.csv", index=False
    )

    approved = (labelled[LABEL] == "Y").mean()
    print(f"lake:    {len(labelled):>4} applications, {approved:.0%} approved")
    print(f"traffic: {len(unlabelled):>4} new applications, no outcome")


if __name__ == "__main__":
    main()
