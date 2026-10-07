"""Unit tests never reach Gemini: the prompt cache gets a client that refuses."""

import pytest

from app.harness import prompt_cache


@pytest.fixture(autouse=True)
def no_real_prompt_cache(monkeypatch):
    def offline():
        raise RuntimeError("unit tests do not call Gemini")

    prompt_cache.reset()
    monkeypatch.setattr(prompt_cache, "_gemini", offline)
