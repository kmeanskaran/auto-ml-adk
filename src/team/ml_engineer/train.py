"""Fit strategy candidates in parallel and keep the best validation F1."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from team.config import load_config
from team.data_scientist.features import engineer
from team.data_scientist.features import feature_columns


def train_candidates(
    frame: pd.DataFrame,
    target: str,
    strategy_level: int,
    models: list[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
  """Fit the chosen candidates in parallel and keep the best validation F1."""
  if target not in frame.columns:
    raise ValueError(f"Target column {target!r} is not in the dataset.")
  names = [name for name in (models or []) if name]
  if not names:
    raise ValueError("Pass the models the SME chose.")
  engineered = engineer(frame, strategy_level)
  usable = engineered.dropna(subset=[target])
  training = load_config().training
  if len(usable) < training.min_rows:
    raise ValueError(f"Need at least {training.min_rows} labeled rows to train.")
  y = usable[target]
  positive = _positive_label(y)
  numeric, categorical = feature_columns(usable, target)
  features = usable[numeric + categorical]
  stratify = y if y.nunique() > 1 and int(y.value_counts().min()) >= 2 else None
  x_train, x_test, y_train, y_test = train_test_split(
      features,
      y,
      test_size=training.test_size,
      random_state=training.random_state,
      stratify=stratify,
  )
  candidates = _estimators(names)

  def _fit_one(spec: tuple[str, Any]) -> dict[str, Any]:
    name, estimator = spec
    pipeline = _make_pipeline(numeric, categorical, estimator)
    try:
      return _score(
          name,
          pipeline,
          x_train.copy(),
          x_test.copy(),
          y_train.copy(),
          y_test.copy(),
          positive,
      )
    except Exception as exc:  # one bad candidate should not sink the panel
      return {"name": name, "error": str(exc), "f1": -1.0}

  workers = max(1, len(candidates))
  with ThreadPoolExecutor(max_workers=workers) as pool:
    scored = list(pool.map(_fit_one, candidates))

  viable = [item for item in scored if item.get("f1", -1) >= 0]
  if not viable:
    errors = "; ".join(f"{item['name']}: {item.get('error')}" for item in scored)
    raise RuntimeError(f"Every candidate failed. {errors}")
  winner = max(viable, key=lambda item: (item["f1"], item["name"] == "logistic_regression"))
  winner_estimator = next(
      estimator for name, estimator in candidates if name == winner["name"]
  )
  final = _make_pipeline(numeric, categorical, clone(winner_estimator))
  final.fit(features, y)
  negatives = [value for value in pd.unique(y) if value != positive]
  model = TabularModel(
      pipeline=final,
      strategy_level=strategy_level,
      target_column=target,
      model_name=winner["name"],
      metrics={key: winner[key] for key in ("f1", "precision", "recall", "accuracy")},
      positive_label=positive,
      negative_label=negatives[0] if negatives else 0,
      decision_threshold=winner.get("threshold"),
      feature_columns=numeric + categorical,
  )
  report = {
      "strategy_level": strategy_level,
      "positive_label": positive,
      "decision_threshold": winner.get("threshold"),
      "candidates": scored,
      "winner": winner["name"],
      "metrics": model.metrics,
      "numeric": numeric,
      "categorical": categorical,
  }
  return model, report


class TabularModel:
  """Self-contained artifact: engineer rows, then run the fitted pipeline."""

  def __init__(
      self,
      pipeline: Pipeline,
      strategy_level: int,
      target_column: str,
      model_name: str,
      metrics: dict[str, Any],
      positive_label: Any,
      negative_label: Any,
      decision_threshold: float | None,
      feature_columns: list[str],
  ):
    self.pipeline = pipeline
    self.strategy_level = strategy_level
    self.target_column = target_column
    self.model_name = model_name
    self.metrics = metrics
    self.positive_label = positive_label
    self.negative_label = negative_label
    self.decision_threshold = decision_threshold
    self.feature_columns = feature_columns

  def predict_records(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = engineer(pd.DataFrame(rows), self.strategy_level)
    missing = [column for column in self.feature_columns if column not in frame.columns]
    if missing:
      raise ValueError(f"Prediction rows are missing columns: {missing}")
    matrix = frame[self.feature_columns]
    probabilities = self.pipeline.predict_proba(matrix)
    classes = list(self.pipeline.named_steps["model"].classes_)
    if self.positive_label in classes:
      positive_index = classes.index(self.positive_label)
    else:
      positive_index = -1
    hard_labels = self.pipeline.predict(matrix)
    predictions = []
    for hard_label, distribution in zip(hard_labels, probabilities):
      probability = float(distribution[positive_index])
      if self.decision_threshold is None:
        label = hard_label.item() if hasattr(hard_label, "item") else hard_label
      elif probability >= self.decision_threshold:
        label = self.positive_label
      else:
        label = self.negative_label
      if hasattr(label, "item"):
        label = label.item()
      predictions.append({"label": label, "probability": probability})
    return predictions


def _positive_label(values: pd.Series) -> Any:
  classes = list(pd.unique(values.dropna()))
  for preferred in (1, 1.0, "1", True, "yes", "Yes", "true", "True", "churn"):
    if preferred in classes:
      return preferred
  counts = values.value_counts()
  return counts.idxmin()


def _estimators(names: list[str]) -> list[tuple[str, Any]]:
  training = load_config().training
  linear = training.logistic_regression
  forest = training.random_forest
  boosting = training.gradient_boosting
  factory: dict[str, Any] = {
      "majority_baseline": lambda: DummyClassifier(strategy="most_frequent"),
      "logistic_regression": lambda: LogisticRegression(
          max_iter=linear.max_iter,
          class_weight=linear.class_weight,
      ),
      "random_forest": lambda: RandomForestClassifier(
          n_estimators=forest.n_estimators,
          max_depth=forest.max_depth,
          min_samples_leaf=forest.min_samples_leaf,
          random_state=training.random_state,
          n_jobs=forest.n_jobs,
          class_weight=forest.class_weight,
      ),
      "gradient_boosting": lambda: GradientBoostingClassifier(
          random_state=training.random_state,
          max_depth=boosting.max_depth,
          n_estimators=boosting.n_estimators,
      ),
  }
  missing = [name for name in names if name not in factory]
  if missing:
    raise ValueError(f"Unknown model(s): {missing}.")
  return [(name, factory[name]()) for name in names]


def _make_pipeline(numeric: list[str], categorical: list[str], estimator: Any) -> Pipeline:
  transformers = []
  if numeric:
    transformers.append(
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            numeric,
        )
    )
  if categorical:
    transformers.append(
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "one_hot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            categorical,
        )
    )
  if not transformers:
    raise ValueError("No usable feature columns after dropping ids and the target.")
  preprocessor = ColumnTransformer(transformers, remainder="drop")
  return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def _binary_scores(
    y_test: pd.Series,
    predicted: Any,
    positive: Any,
) -> dict[str, float]:
  kwargs = {"average": "binary", "zero_division": 0, "pos_label": positive}
  return {
      "f1": float(f1_score(y_test, predicted, **kwargs)),
      "precision": float(precision_score(y_test, predicted, **kwargs)),
      "recall": float(recall_score(y_test, predicted, **kwargs)),
      "accuracy": float(accuracy_score(y_test, predicted)),
  }


def _threshold_for_f1(
    y_test: pd.Series,
    positive_probability: Any,
    positive: Any,
) -> tuple[float, Any]:
  """Pick the validation threshold that maximizes positive-class F1."""
  classes = [value for value in pd.unique(y_test) if value != positive]
  negative = classes[0] if classes else 0
  training = load_config().training
  best_threshold = 0.5
  best_predicted = None
  best_f1 = -1.0
  threshold = training.threshold_start
  while threshold <= training.threshold_stop + 1e-9:
    predicted = [
        positive if probability >= threshold else negative
        for probability in positive_probability
    ]
    score = float(
        f1_score(y_test, predicted, average="binary", pos_label=positive, zero_division=0)
    )
    if score > best_f1:
      best_f1 = score
      best_threshold = threshold
      best_predicted = predicted
    threshold = round(threshold + training.threshold_step, 10)
  return best_threshold, best_predicted


def _score(
    name: str,
    pipeline: Pipeline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    positive: Any,
) -> dict[str, Any]:
  fitted = pipeline.fit(train, y_train)
  binary = int(y_test.nunique()) == 2
  threshold = None
  if binary:
    classes = list(fitted.named_steps["model"].classes_)
    if positive in classes:
      probabilities = fitted.predict_proba(test)[:, classes.index(positive)]
      threshold, predicted = _threshold_for_f1(y_test, probabilities, positive)
      metrics = _binary_scores(y_test, predicted, positive)
    else:
      predicted = fitted.predict(test)
      metrics = _binary_scores(y_test, predicted, positive)
  else:
    predicted = fitted.predict(test)
    kwargs = {"average": "macro", "zero_division": 0}
    metrics = {
        "f1": float(f1_score(y_test, predicted, **kwargs)),
        "precision": float(precision_score(y_test, predicted, **kwargs)),
        "recall": float(recall_score(y_test, predicted, **kwargs)),
        "accuracy": float(accuracy_score(y_test, predicted)),
    }
  return {"name": name, "threshold": threshold, **metrics}
