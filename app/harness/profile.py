"""The data profile: measured by harness code, so the analyst starts from facts.

The analyst reads this report, checks what it finds surprising with its own
scripts, and writes the short summary the human sees. Nothing here interprets
the data; it only measures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MISSING_TOKENS = ("NULL", "null", "NA", "N/A", "n/a", "None", "?", "")
MAX_LEVELS = 50  # categories beyond this are pooled when measuring signal
TOP_CHARTED = 10


def profile(path, target: str, positive_value: str | None = None) -> dict:
    frame = pd.read_csv(path, low_memory=False)
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} is not in {path.name}.")
    y = outcome(frame[target], positive_value)
    labelled = y.notna()
    columns = [
        _column(frame[name], y, labelled) for name in frame.columns if name != target
    ]
    positives = int((y == 1).sum())
    report = {
        "dataset": path.name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "target": {
            "column": target,
            "positive_value": positive_value,
            "positive_rate": round(positives / max(int(labelled.sum()), 1), 4),
            "positives": positives,
            "negatives": int((y == 0).sum()),
            "unlabelled": int((~labelled).sum()),
        },
        "column_profiles": columns,
    }
    report["charts"] = _charts(report)
    return report


def outcome(series: pd.Series, positive_value: str | None) -> pd.Series:
    """The target as 1 / 0 / NaN: 1 where it equals positive_value, or numeric as is."""
    if positive_value is None:
        return pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip()
    missing = series.isna() | text.isin(MISSING_TOKENS)
    return (text == str(positive_value)).astype(float).where(~missing)


def _column(series: pd.Series, y: pd.Series, labelled: pd.Series) -> dict:
    tokens = (
        series.astype(str).str.strip().isin(MISSING_TOKENS) & series.notna()
        if series.dtype == object
        else pd.Series(False, index=series.index)
    )
    missing = series.isna() | tokens
    numeric = pd.api.types.is_numeric_dtype(series)
    unique = int(series.nunique(dropna=True))
    entry: dict = {
        "name": series.name,
        "dtype": str(series.dtype),
        "kind": _kind(series, numeric, unique),
        "missing_pct": round(100 * float(missing.mean()), 2),
        "missing_as_text": int(tokens.sum()),
        "unique": unique,
    }
    if numeric:
        values = series[~missing]
        entry["stats"] = {
            k: _round(v)
            for k, v in {
                "min": values.min(),
                "median": values.median(),
                "mean": values.mean(),
                "max": values.max(),
            }.items()
        }
    else:
        top = series[~missing].astype(str).value_counts().head(3)
        entry["top"] = [[str(k), int(v)] for k, v in top.items()]
    entry["signal_auc"] = (
        None
        if entry["kind"] == "identifier"
        else _signal(series.where(~missing), y, labelled, numeric)
    )
    return entry


def _kind(series: pd.Series, numeric: bool, unique: int) -> str:
    if unique >= 0.95 * len(series) and not pd.api.types.is_float_dtype(series):
        return "identifier"
    if numeric:
        return "binary" if unique <= 2 else "numeric"
    if unique <= MAX_LEVELS:
        return "categorical"
    parsed = pd.to_datetime(series.dropna().head(200), errors="coerce", format="mixed")
    return "date" if parsed.notna().mean() > 0.9 else "high-cardinality text"


def _signal(
    values: pd.Series, y: pd.Series, labelled: pd.Series, numeric: bool
) -> float | None:
    """How well this column alone ranks the outcome: max(AUC, 1 - AUC), in-sample."""
    x, target = values[labelled], y[labelled]
    if target.nunique() < 2:
        return None
    if numeric:
        score = x.fillna(x.median() if x.notna().any() else 0)
    else:
        levels = x.astype(str).where(x.notna(), "<missing>")
        keep = levels.value_counts().index[:MAX_LEVELS]
        levels = levels.where(levels.isin(keep), "<other>")
        score = levels.map(target.groupby(levels).mean())
    if score.nunique() < 2:
        return None
    auc = float(roc_auc_score(target, score))
    return round(max(auc, 1 - auc), 3)


def _charts(report: dict) -> list[dict]:
    columns = report["column_profiles"]
    missing = sorted(
        (c for c in columns if c["missing_pct"] > 0), key=lambda c: -c["missing_pct"]
    )[:TOP_CHARTED]
    signal = sorted(
        (c for c in columns if c["signal_auc"] is not None),
        key=lambda c: -c["signal_auc"],
    )[:TOP_CHARTED]
    charts = []
    if signal:
        charts.append(
            {
                "type": "hbar",
                "title": "Columns that predict the outcome on their own (0.5 = no signal, 1 = perfect)",
                "categories": [c["name"] for c in signal],
                "series": [
                    {"name": "AUC", "values": [c["signal_auc"] for c in signal]}
                ],
                "domain": [0.5, 1.0],
            }
        )
    if missing:
        charts.append(
            {
                "type": "hbar",
                "title": "Columns with missing values (% of rows)",
                "categories": [c["name"] for c in missing],
                "series": [
                    {"name": "% missing", "values": [c["missing_pct"] for c in missing]}
                ],
            }
        )
    return charts


def _round(value) -> float | int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value = value.item() if hasattr(value, "item") else value
    return round(value, 4) if isinstance(value, float) else value
