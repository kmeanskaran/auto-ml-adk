"""Pipeline settings from config/pipeline.yml. The pipeline reads them; it never asks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config" / "pipeline.yml"


@dataclass(frozen=True)
class Settings:
    dataset: str
    target: str
    feature_view: str
    prediction_moment: str
    cost_missed: float
    cost_false_alarm: float

    def brief(self) -> str:
        return (
            f"Dataset: {self.dataset} (in DATA_DIR). Target: {self.target} (1 = positive).\n"
            f"Prediction moment: {self.prediction_moment}.\n"
            f"Costs: a missed positive costs {self.cost_missed}, a false alarm costs "
            f"{self.cost_false_alarm}. Choose the threshold that minimises total cost."
        )


def load() -> Settings:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return Settings(
        dataset=raw["dataset"],
        target=raw["target"],
        feature_view=raw.get("feature_view") or Path(raw["dataset"]).stem,
        prediction_moment=raw["prediction_moment"],
        cost_missed=float(raw["costs"]["missed_cancellation"]),
        cost_false_alarm=float(raw["costs"]["false_alarm"]),
    )
