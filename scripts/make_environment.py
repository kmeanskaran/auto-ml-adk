"""Split the hotel bookings into what the team sees and later production traffic.

Run once, before any agent:

    uv run python scripts/make_environment.py

- data/lake/hotel_bookings.csv   bookings arriving Jul 2015 - Dec 2016, every column
- data/traffic/2017-MM.csv       bookings arriving in 2017, one file per month, as
                                 they would reach the model at booking time: no
                                 label and no reservation outcome columns
- data/traffic/2017-MM-labels.csv  the true outcome, released a month later

The split is by arrival date, so traffic is real later data and any drift in it
is real, not injected.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data"
RAW = ROOT / "datasets" / "hotels" / "hotels.csv"
LAKE = ROOT / "lake"
TRAFFIC = ROOT / "traffic"

LABEL = "is_canceled"
# Written when the stay ends or the booking is cancelled, so absent at booking time.
OUTCOME_COLUMNS = ["reservation_status", "reservation_status_date"]
CUTOFF_YEAR = 2017


def main() -> None:
    raw = pd.read_csv(RAW, dtype=str, keep_default_na=False)
    raw.insert(0, "booking_id", [f"B{i:06d}" for i in range(1, len(raw) + 1)])
    arrival = pd.to_datetime(
        raw["arrival_date_year"] + "-" + raw["arrival_date_month"] + "-01",
        format="%Y-%B-%d",
    )
    history = raw[arrival.dt.year < CUTOFF_YEAR]
    future = raw[arrival.dt.year >= CUTOFF_YEAR]

    for folder in (LAKE, TRAFFIC):
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
    history.to_csv(LAKE / "hotel_bookings.csv", index=False)

    months = arrival[future.index].dt.strftime("%Y-%m")
    for month in sorted(months.unique()):
        batch = future[months == month]
        batch.drop(columns=[LABEL, *OUTCOME_COLUMNS]).to_csv(
            TRAFFIC / f"{month}.csv", index=False
        )
        batch[["booking_id", LABEL]].to_csv(
            TRAFFIC / f"{month}-labels.csv", index=False
        )

    print(f"lake:    {len(history):>6} bookings, arrivals before {CUTOFF_YEAR}")
    print(f"traffic: {len(future):>6} bookings in {months.nunique()} monthly batches")


if __name__ == "__main__":
    main()
