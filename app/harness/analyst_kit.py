"""kit.py: tested helpers for the analysts' scripts. The harness copies this file into
the analyst's folder before every run, so edits to the copy are overwritten.

    from kit import *
    df = lake()                          # the dataset, with `outcome` as 0/1
    rate_by(df, "Property_Area")         # outcome rate per group: 95% interval, lift, rows
    compare(df, "Married")               # is the gap between groups real? p-value, effect size
    drift(lake(), traffic())             # what production records do differently (PSI)
    importance()                         # what the live model relies on (final test)
    score(traffic().head(20))            # the live model's scores for raw records
    chart("rate_by_area", rate_by(df, "Property_Area"))   # writes charts/rate_by_area.json
    chart("top_features", importance(), value="auc_drop", label="feature", type="hbar")
    print(table(rate_by(df, "Education")))                # markdown, 8 rows
"""

from __future__ import annotations

import glob
import importlib.util
import json
import math
import os

import numpy as np
import pandas as pd

TARGET = os.environ.get("TARGET", "")
POSITIVE = os.environ.get("POSITIVE") or None
MIN_ROWS = 30  # a group smaller than this gives a shaky rate

__all__ = [
    "TARGET",
    "chart",
    "compare",
    "drift",
    "features",
    "importance",
    "lake",
    "live",
    "rate_by",
    "score",
    "show_chart",
    "table",
    "traffic",
]


# --- data ------------------------------------------------------------------------------


def lake() -> pd.DataFrame:
    """The labelled dataset, plus `outcome`: 1 for the positive value, 0 otherwise,
    missing where the target is missing."""
    df = pd.read_csv(os.path.join(os.environ["DATA_DIR"], os.environ["DATASET"]))
    if TARGET in df.columns:
        y = df[TARGET]
        hit = y.astype(str) == POSITIVE if POSITIVE else y.astype(float) == 1
        df["outcome"] = hit.astype(float).where(y.notna())
    return df


def traffic() -> pd.DataFrame:
    """Production records (no outcome), with a `batch` column naming their file."""
    path = os.environ.get("TRAFFIC_FILE", "")
    if not path or not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
    return df.assign(batch=os.path.basename(path).removesuffix(".csv"))


def features(version: str | None = None) -> tuple[pd.DataFrame, dict]:
    """A feature view version's tables (train, valid, test stacked, with a `split`
    column) and its definition. The newest version unless one is named."""
    root = os.environ.get("FEATURE_VIEW_DIR", "")
    found = sorted(
        (p for p in glob.glob(os.path.join(root, "v*")) if os.path.isdir(p)),
        key=lambda p: int(os.path.basename(p)[1:] or 0),
    )
    if version:
        found = [p for p in found if os.path.basename(p) == version]
    if not found:
        raise FileNotFoundError(f"no feature view version {version or ''} in {root}")
    folder = found[-1]
    definition = json.load(open(os.path.join(folder, "definition.json")))
    parts = [
        pd.read_parquet(os.path.join(folder, "offline", f"{s}.parquet")).assign(split=s)
        for s in ("train", "valid", "test")
    ]
    return pd.concat(parts, ignore_index=True), definition


# --- statistics --------------------------------------------------------------------------


def _interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a rate of k in n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _groups(values: pd.Series, bins: int) -> pd.Series:
    """Group labels: numbers with many values in quantile bins; missing as its own group."""
    if pd.api.types.is_numeric_dtype(values) and values.nunique() > 12:
        labels = pd.qcut(values, bins, duplicates="drop").astype(str)
    else:
        labels = values.astype(str)
    return labels.where(values.notna(), "(missing)")


def rate_by(
    df: pd.DataFrame, column: str, bins: int = 5, outcome: str = "outcome"
) -> pd.DataFrame:
    """Outcome rate per group of `column`: rows, rate, 95% interval, lift over the
    overall rate, and `shaky` when the group has fewer than 30 rows."""
    d = df[df[outcome].notna()]
    groups = _groups(d[column], bins)
    overall = float(d[outcome].mean())
    rows = []
    for value, part in d.groupby(groups, sort=True)[outcome]:
        n, k = len(part), int(part.sum())
        low, high = _interval(k, n)
        rows.append(
            {
                "group": str(value),
                "rows": n,
                "rate": round(k / n, 4),
                "low": round(low, 4),
                "high": round(high, 4),
                "lift": round((k / n) / overall, 2) if overall else math.nan,
                "shaky": n < MIN_ROWS,
            }
        )
    return pd.DataFrame(rows)


def compare(
    df: pd.DataFrame, column: str, bins: int = 5, outcome: str = "outcome"
) -> dict:
    """Whether the outcome rate really differs across `column`: chi-square p-value over
    all groups, Cramér's V (0 none, 0.1 small, 0.3 medium, 0.5 large), and the highest
    and lowest groups with 30+ rows."""
    from scipy.stats import chi2_contingency

    d = df[df[outcome].notna()]
    groups = _groups(d[column], bins)
    counts = pd.crosstab(groups, d[outcome])
    if counts.shape[0] < 2 or counts.shape[1] < 2:
        return {
            "column": column,
            "verdict": "only one group or one outcome: nothing to compare",
        }
    chi2, p, _, _ = chi2_contingency(counts)
    v = math.sqrt(chi2 / (counts.values.sum() * (min(counts.shape) - 1)))
    rates = rate_by(df, column, bins, outcome)
    solid = rates[~rates["shaky"]] if (~rates["shaky"]).sum() >= 2 else rates
    high, low = solid.loc[solid["rate"].idxmax()], solid.loc[solid["rate"].idxmin()]
    size = (
        "large"
        if v >= 0.5
        else "medium"
        if v >= 0.3
        else "small"
        if v >= 0.1
        else "negligible"
    )
    return {
        "column": column,
        "p_value": round(float(p), 4),
        "cramers_v": round(v, 3),
        "effect": size,
        "highest": {
            "group": high["group"],
            "rate": high["rate"],
            "rows": int(high["rows"]),
        },
        "lowest": {
            "group": low["group"],
            "rate": low["rate"],
            "rows": int(low["rows"]),
        },
        "verdict": (
            f"{column}: {size} effect (V={v:.2f}), "
            + ("unlikely to be chance" if p < 0.05 else "could be chance")
            + f" (p={p:.3f}); {high['group']} {high['rate']:.0%} vs {low['group']} {low['rate']:.0%}"
        ),
    }


def drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """How far each shared column moved from reference to current: PSI (under 0.1
    stable, 0.1-0.25 moderate, over 0.25 large) and the change in missing share."""
    skip = {"outcome", TARGET, "batch"}
    shared = columns or [
        c
        for c in reference.columns
        if c in current.columns
        and c not in skip
        # identifiers: nearly every value is new, so their drift means nothing
        and not (
            not pd.api.types.is_numeric_dtype(reference[c])
            and reference[c].nunique() > 0.9 * len(reference)
        )
    ]
    rows = []
    for col in shared:
        ref, cur = reference[col], current[col]
        if pd.api.types.is_numeric_dtype(ref) and ref.nunique() > 12:
            edges = np.unique(np.nanquantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
            edges[0], edges[-1] = -np.inf, np.inf
            a = pd.cut(ref, edges).astype(str).where(ref.notna(), "(missing)")
            b = pd.cut(cur, edges).astype(str).where(cur.notna(), "(missing)")
        else:
            a, b = (
                ref.astype(str).where(ref.notna(), "(missing)"),
                cur.astype(str).where(cur.notna(), "(missing)"),
            )
        pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
        keys = pa.index.union(pb.index)
        pa, pb = (
            pa.reindex(keys, fill_value=0) + 1e-4,
            pb.reindex(keys, fill_value=0) + 1e-4,
        )
        psi = float(((pb - pa) * np.log(pb / pa)).sum())
        rows.append(
            {
                "column": col,
                "psi": round(psi, 3),
                "level": "large"
                if psi > 0.25
                else "moderate"
                if psi > 0.1
                else "stable",
                "missing_before": round(float(ref.isna().mean()), 3),
                "missing_now": round(float(cur.isna().mean()), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


# --- the live model ----------------------------------------------------------------------


def live():
    """The model in production: (predict, meta), or None when nothing is live."""
    folder = os.environ.get("LIVE_MODEL_DIR", "")
    source = os.path.join(folder, "src", "predict.py")
    if not folder or not os.path.isfile(source):
        return None
    spec = importlib.util.spec_from_file_location("live_predict", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.predict, module.META


def score(records: pd.DataFrame) -> pd.DataFrame:
    """The live model's score and flag for raw records (as in the lake or traffic)."""
    model = live()
    if model is None:
        raise RuntimeError("no model is live yet")
    predict, _ = model
    clean = records.drop(
        columns=[c for c in ("outcome", TARGET, "batch") if c in records.columns]
    )
    rows = [
        {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in r.items()
        }
        for r in clean.to_dict("records")
    ]
    return pd.DataFrame(predict(rows))


def importance(top: int = 10, repeats: int = 5) -> pd.DataFrame:
    """What the live model relies on: how much ROC AUC drops on its final-test rows when
    one feature's values are shuffled (permutation importance)."""
    import joblib
    from sklearn.inspection import permutation_importance

    folder = os.environ.get("LIVE_MODEL_DIR", "")
    if not folder:
        raise RuntimeError("no model is live yet")
    meta = json.load(open(os.path.join(folder, "artifacts", "model.json")))
    model = joblib.load(os.path.join(folder, "artifacts", "model.joblib"))
    test = pd.read_parquet(
        os.path.join(os.environ["LIVE_FEATURE_DIR"], "offline", "test.parquet")
    )
    X, y = test[meta["features"]], test[meta["outcome_column"]].astype(int)
    result = permutation_importance(
        model, X, y, scoring="roc_auc", n_repeats=repeats, random_state=0
    )
    out = pd.DataFrame(
        {
            "feature": meta["features"],
            "auc_drop": result.importances_mean.round(4),
            "spread": result.importances_std.round(4),
        }
    )
    return out.sort_values("auc_drop", ascending=False, ignore_index=True).head(top)


# --- showing results ---------------------------------------------------------------------


def chart(
    name: str,
    data,
    value: str = "rate",
    label: str = "group",
    type: str = "bar",
    title: str = "",
    series: str = "",
) -> str:
    """Write charts/<name>.json for show_chart from a table (e.g. rate_by's) or a dict
    {label: value}. Rates (0-1) are shown as percentages. Returns the path: pass that
    path on (to show_chart or submit_summary). "x", "x.json" and "charts/x.json" all
    name the same file."""
    name = os.path.basename(str(name)).removesuffix(".json")
    if isinstance(data, dict) and "series" in data and "categories" in data:
        spec = {"x_label": "", "y_label": "", "title": title or name, **data}
        return _write_chart(name, spec)  # a whole chart spec, written as it is
    if isinstance(data, dict):
        categories, values = [str(k) for k in data], list(data.values())
    else:
        categories, values = data[label].astype(str).tolist(), data[value].tolist()
    as_pct = value in ("rate", "low", "high") or all(
        isinstance(v, float) and 0 <= v <= 1 for v in values
    )
    values = [
        None
        if v is None or (isinstance(v, float) and math.isnan(v))
        else round(100 * v if as_pct else float(v), 2)
        for v in values
    ]
    spec = {
        "type": type if type != "bar" or len(categories) <= 12 else "hbar",
        "title": title or f"{series or value} by {label}",
        "x_label": "",
        "y_label": "%" if as_pct else "",
        "categories": categories[:40],
        "series": [
            {
                "name": series or (f"{value} (%)" if as_pct else value),
                "values": values[:40],
            }
        ],
    }
    return _write_chart(name, spec)


def show_chart(name: str, data=None, **options) -> str:
    """For scripts: same as chart(name, data, ...). Every chart a script writes during
    an answer is shown with it; the show_chart tool is how you show one explicitly."""
    if data is None:
        return os.path.join(
            "charts", os.path.basename(str(name)).removesuffix(".json") + ".json"
        )
    return chart(name, data, **options)


def _write_chart(name: str, spec: dict) -> str:
    os.makedirs("charts", exist_ok=True)
    path = os.path.join("charts", f"{name}.json")
    with open(path, "w") as handle:
        json.dump(spec, handle, default=float)
    return path


def table(df: pd.DataFrame, n: int = 8, digits: int = 3) -> str:
    """A markdown table of the first n rows, numbers rounded."""
    shown = df.head(n).copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].round(digits)
    header = "| " + " | ".join(map(str, shown.columns)) + " |"
    rule = "|" + "---|" * len(shown.columns)
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in shown.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])
