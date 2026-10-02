"""Which Gemini model each agent thinks with, and how hard.

config/config.yml (models) sets the defaults: `team` for the analyst's first look, the
engineer and the skeptic, and `analyst` for the chat analyst, each with a thinking level.
The console's model menu can switch the team's model during a run: every agent is
built once, and the choice is applied to each model request (a before_model_callback),
so it takes effect from the next call. That choice is saved in .adk/model.json, which
survives restarts and "Clear runs". All this applies when the agents run on Gemini; with
ML_MODEL set to another LiteLLM model (e.g. Ollama) requests are left as they are.
"""

from __future__ import annotations

import json
import os
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from app import settings
from app.harness.project import APP_ROOT

MODELS = {
    "gemini-3.8-flash": "Gemini 3.8 Flash",
    "gemini-3.7-flash": "Gemini 3.7 Flash",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
}
THINKING = ("low", "medium", "high")  # how long the model reasons before it answers
STATE = APP_ROOT / ".adk" / "model.json"


def base() -> str:
    """The model the agents are built with: ML_MODEL if set, else models.team."""
    return (os.environ.get("ML_MODEL") or settings.load().team_model).removeprefix(
        "gemini/"
    )


def on_gemini() -> bool:
    return base().startswith("gemini")


def current() -> dict[str, str]:
    """The team's model and thinking level: the console's choice, else config."""
    config = settings.load()
    model = base() if base() in MODELS else config.team_model
    thinking = config.team_thinking
    try:
        saved = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {}
    except (OSError, json.JSONDecodeError):
        saved = {}
    if saved.get("model") in MODELS:
        model = saved["model"]
    if saved.get("thinking") in THINKING:
        thinking = saved["thinking"]
    return {"model": model, "thinking": thinking if thinking in THINKING else "low"}


def choose(model: str, thinking: str) -> dict[str, str]:
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}: pick one of {', '.join(MODELS)}.")
    if thinking not in THINKING:
        raise ValueError(f"Unknown thinking level {thinking!r}: {', '.join(THINKING)}.")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        json.dumps({"model": model, "thinking": thinking}), encoding="utf-8"
    )
    return current()


def analyst() -> dict[str, str]:
    """The chat analyst's model and thinking level (config models.analyst)."""
    config = settings.load()
    thinking = config.analyst_thinking
    return {
        "model": config.analyst_model,
        "thinking": thinking if thinking in THINKING else "low",
    }


def view() -> dict[str, Any]:
    return {
        **current(),
        "analyst": analyst(),
        "available": on_gemini(),
        "base": base(),
        "models": [{"key": k, "label": v} for k, v in MODELS.items()],
        "thinking_levels": list(THINKING),
    }


def apply(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
    """before_model_callback: send this request to the agent's model and thinking
    level, the chat analyst's from config, the team's from the console or config."""
    if not on_gemini():
        return None
    agent = getattr(callback_context, "agent_name", "")
    choice = analyst() if agent == "analyst" else current()
    llm_request.model = choice["model"]
    config = llm_request.config or types.GenerateContentConfig()
    config.thinking_config = types.ThinkingConfig(
        thinking_level=types.ThinkingLevel(choice["thinking"].upper())
    )
    llm_request.config = config
    return None
