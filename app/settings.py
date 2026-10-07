"""Settings from config/config.yml, the one config file. The pipeline reads them; it
never asks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yml"
REVIEWS = ("features", "plan", "promote")


@dataclass(frozen=True)
class Settings:
    dataset: str
    target: str
    feature_view: str
    prediction_moment: str
    cost_missed: float
    cost_false_alarm: float
    goal: str = ""
    traffic: str = ""  # production records (no outcome), relative to data/
    positive_value: str | None = None  # target value meaning 1; None: already 0/1
    record: str = "record"  # the console's word for one row, e.g. "application"
    positive: str = "positive"  # its word for the outcome, e.g. "approval"
    false_alarm: str = "false alarm"  # its word for a wrong flag, e.g. "wrong approval"
    ask_human: tuple[str, ...] = REVIEWS  # reviews that always wait for the human
    protected: tuple[str, ...] = ()  # raw columns whose groups must be treated alike
    min_group_ratio: float = 0.8  # lowest group's approval rate / highest, at least
    team_model: str = "gemini-3.7-flash"  # models.team
    team_thinking: str = "low"
    analyst_model: str = "gemini-3.5-flash"  # models.analyst: the chat analyst
    analyst_thinking: str = "low"
    context_cache: bool = True  # models.context_cache: cache each role's prompt head
    tool_budget: int = 20  # limits.tool_budget: tool calls per agent turn
    skeptic_tool_budget: int = 10  # limits.skeptic_tool_budget: the skeptic's
    fix_rounds: int = 1  # limits.fix_rounds
    cv_folds: int = 3  # limits.cv_folds: folds when the harness compares by CV
    compact_above_chars: int = 50_000  # limits.compact_above_chars
    script_seconds: int = 300  # limits.script_seconds: a script is stopped after this

    def brief(self) -> str:
        outcome = (
            f"Target: {self.target}; the value {self.positive_value!r} is the outcome. "
            f"Write the outcome column as 1 for {self.positive_value!r} and 0 otherwise."
            if self.positive_value is not None
            else f"Target: {self.target} (1 = positive)."
        )
        from app.harness.environment import cpus  # here: settings loads before harness

        return (
            f"Dataset: {self.dataset}, read it as "
            f"os.path.join(os.environ['DATA_DIR'], {self.dataset!r}). {outcome}\n"
            f"Compute: {cpus()} CPUs; a script is stopped after {self.script_seconds} s. "
            "Size searches to fit: a few candidate settings, not exhaustive grids; time "
            "one fit before looping over many.\n"
            f"Prediction moment: {self.prediction_moment}.\n"
            f"Costs: a missed positive costs {self.cost_missed}, a false alarm costs "
            f"{self.cost_false_alarm}. Choose the threshold that minimises total cost."
            + (
                f"\nProtected attributes (fair lending): {', '.join(self.protected)}. "
                "Their groups must be treated alike; the harness compares them at promote."
                if self.protected
                else ""
            )
        )


def load() -> Settings:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    words, costs = raw.get("words") or {}, raw["costs"]
    autonomy = raw.get("autonomy") or {}
    ask = autonomy.get("ask_human", list(REVIEWS))
    unknown = set(ask) - set(REVIEWS)
    if unknown:
        raise ValueError(f"autonomy.ask_human: unknown reviews {sorted(unknown)}")
    positive_value = raw.get("positive_value")
    fairness = raw.get("fairness") or {}
    llms, limits = raw.get("models") or {}, raw.get("limits") or {}
    return Settings(
        dataset=raw["dataset"],
        target=raw["target"],
        feature_view=raw.get("feature_view") or Path(raw["dataset"]).stem,
        prediction_moment=raw["prediction_moment"],
        cost_missed=float(costs.get("missed_positive", costs.get("missed", 1.0))),
        cost_false_alarm=float(costs["false_alarm"]),
        goal=raw.get("goal", ""),
        traffic=str(raw.get("traffic") or ""),
        positive_value=None if positive_value is None else str(positive_value),
        record=words.get("record", "record"),
        positive=words.get("positive", "positive"),
        false_alarm=words.get("false_alarm", "false alarm"),
        ask_human=tuple(ask),
        protected=tuple(str(a) for a in fairness.get("attributes") or ()),
        min_group_ratio=float(fairness.get("min_ratio", 0.8)),
        team_model=str(llms.get("team") or Settings.team_model),
        team_thinking=str(llms.get("team_thinking") or Settings.team_thinking),
        analyst_model=str(llms.get("analyst") or Settings.analyst_model),
        analyst_thinking=str(llms.get("analyst_thinking") or Settings.analyst_thinking),
        context_cache=bool(llms.get("context_cache", True)),
        tool_budget=max(5, int(limits.get("tool_budget", Settings.tool_budget))),
        skeptic_tool_budget=max(
            5, int(limits.get("skeptic_tool_budget", Settings.skeptic_tool_budget))
        ),
        cv_folds=max(2, int(limits.get("cv_folds", Settings.cv_folds))),
        fix_rounds=max(0, int(limits.get("fix_rounds", Settings.fix_rounds))),
        compact_above_chars=max(
            10_000, int(limits.get("compact_above_chars", Settings.compact_above_chars))
        ),
        script_seconds=max(
            30, int(limits.get("script_seconds", Settings.script_seconds))
        ),
    )
