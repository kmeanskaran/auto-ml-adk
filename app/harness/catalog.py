"""What the human can choose from at the training-plan review: models and metrics.

The engineer proposes from these lists, the human confirms or changes the choice,
and the harness (not the engineer) computes every metric for every candidate.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    label: str
    higher_is_better: bool
    needs_threshold: bool  # computed at the cost-minimising threshold
    about: str


METRICS: dict[str, Metric] = {
    "roc_auc": Metric("ROC-AUC", True, False, "ranking quality across all thresholds"),
    "pr_auc": Metric("PR-AUC", True, False, "ranking quality on the positive class"),
    "f1": Metric("F1", True, True, "balance of precision and recall"),
    "precision": Metric("Precision", True, True, "share of flags that are right"),
    "recall": Metric("Recall", True, True, "share of positives that are flagged"),
    "accuracy": Metric(
        "Accuracy", True, True, "share of all predictions that are right"
    ),
    "log_loss": Metric("Log loss", False, False, "how well calibrated the scores are"),
    "brier": Metric("Brier score", False, False, "mean squared error of the scores"),
    "cost_per_1000": Metric(
        "Cost per 1,000", False, True, "business cost of errors per 1,000 records"
    ),
}

_MODELS = {
    "logistic_regression": ("Logistic regression", "sklearn"),
    "random_forest": ("Random forest", "sklearn"),
    "extra_trees": ("Extra trees", "sklearn"),
    "gradient_boosting": ("Gradient boosting", "sklearn"),
    "hist_gradient_boosting": ("Histogram gradient boosting", "sklearn"),
    "xgboost": ("XGBoost", "xgboost"),
}


def models() -> dict[str, str]:
    """Model key → label, for the libraries installed in this environment."""
    return {
        key: label
        for key, (label, package) in _MODELS.items()
        if importlib.util.find_spec(package) is not None
    }


def better(metric: str, a: float | None, b: float | None) -> bool:
    """Whether a beats b on metric. A missing value never wins."""
    if a is None:
        return False
    if b is None:
        return True
    return a > b if METRICS[metric].higher_is_better else a < b


def as_options() -> dict:
    """The catalog in the shape the UI renders."""
    return {
        "models": [{"key": k, "label": v} for k, v in models().items()],
        "metrics": [
            {
                "key": k,
                "label": m.label,
                "about": m.about,
                "higher_is_better": m.higher_is_better,
            }
            for k, m in METRICS.items()
        ],
    }
