"""What the human can choose from at the training-plan review: models and metrics.

The engineer proposes from these lists, the human confirms or changes the choice,
and the harness (not the engineer) computes every metric for every candidate.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    label: str  # the technical name, shown small
    higher_is_better: bool
    needs_threshold: bool  # computed at the cost-minimising threshold
    kind: str  # "rate" (shown as %), "score" (0-1, 3 decimals) or "cost"
    plain: str  # the business name; {record}, {positive}, {Positive} are filled in
    about: str  # one line on what it means; also {missed} and {false_alarm}


METRICS: dict[str, Metric] = {
    "cost_per_1000": Metric(
        "Cost per 1,000",
        False,
        True,
        "cost",
        "Loss per 1,000 {record}s",
        "What its mistakes cost per 1,000 {record}s: each missed {positive} costs "
        "{missed}, each {false_alarm_word} costs {false_alarm}. Lower is better.",
    ),
    "recall": Metric(
        "Recall",
        True,
        True,
        "rate",
        "{Positive}s caught",
        "Of the {record}s that ended in a {positive}, the share it flagged.",
    ),
    "precision": Metric(
        "Precision",
        True,
        True,
        "rate",
        "Right when it flags",
        "Of the {record}s it flags, the share that really ended in a {positive}.",
    ),
    "f1": Metric(
        "F1",
        True,
        True,
        "rate",
        "Balance of caught and right",
        "One number that rewards catching {positive}s and being right when flagging.",
    ),
    "accuracy": Metric(
        "Accuracy",
        True,
        True,
        "rate",
        "Right overall",
        "Share of all {record}s it gets right. Misleading when {positive}s are rare.",
    ),
    "roc_auc": Metric(
        "ROC-AUC",
        True,
        False,
        "score",
        "Ranking quality",
        "How well it ranks likely {positive}s above the rest, whatever the cut-off. "
        "0.5 is guessing, 1 is perfect.",
    ),
    "pr_auc": Metric(
        "PR-AUC",
        True,
        False,
        "score",
        "Ranking quality on {positive}s",
        "Like ranking quality, but judged on the {record}s that ended in a {positive}. "
        "Guessing scores the {positive} rate.",
    ),
    "log_loss": Metric(
        "Log loss",
        False,
        False,
        "score",
        "Probability error",
        "How far its probabilities are from what happened. Lower is better.",
    ),
    "brier": Metric(
        "Brier score",
        False,
        False,
        "score",
        "Probability error (squared)",
        "Average squared gap between its probability and what happened. Lower is better.",
    ),
}


def words() -> dict[str, str]:
    """The business words and costs the plain texts are written with."""
    from app import settings

    config = settings.load()
    return {
        "record": config.record,
        "positive": config.positive,
        "Positive": config.positive[:1].upper() + config.positive[1:],
        "false_alarm_word": config.false_alarm,
        "missed": f"{config.cost_missed:g}",
        "false_alarm": f"{config.cost_false_alarm:g}",
    }


def describe() -> dict[str, dict]:
    """Every metric with its plain name and meaning, in the order the console shows them."""
    fill = words()
    return {
        key: {
            "label": m.label,
            "plain": m.plain.format(**fill),
            "about": m.about.format(**fill),
            "kind": m.kind,
            "higher_is_better": m.higher_is_better,
        }
        for key, m in METRICS.items()
    }


_MODELS = {
    "logistic_regression": ("Logistic regression", "sklearn"),
    "decision_tree": ("Decision tree", "sklearn"),
    "knn": ("Nearest neighbours", "sklearn"),
    "svm": ("Support vector machine", "sklearn"),
    "random_forest": ("Random forest", "sklearn"),
    "adaboost": ("AdaBoost", "sklearn"),
    "extra_trees": ("Extra trees", "sklearn"),
    "gradient_boosting": ("Gradient boosting", "sklearn"),
    "hist_gradient_boosting": ("Histogram gradient boosting", "sklearn"),
    "xgboost": ("XGBoost", "xgboost"),
    "lightgbm": ("LightGBM", "lightgbm"),
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
        "metrics": [{"key": k, **m} for k, m in describe().items()],
    }
