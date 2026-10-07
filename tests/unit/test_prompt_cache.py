"""The turn's growing prefix is cached in the background and sent by reference."""

import asyncio
import types as pytypes

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from app.harness import prompt_cache

CONTEXT = pytypes.SimpleNamespace(agent_name="engineer_features", invocation_id="i1")
BIG = "x" * prompt_cache.MIN_CHARS


class FakeCaches:
    def __init__(self, error=None):
        self.created, self.error = [], error

    async def create(self, model, config):
        if self.error:
            raise ValueError(self.error)
        self.created.append(config)
        return pytypes.SimpleNamespace(name=f"cachedContents/{len(self.created)}")


def turn(steps):
    """The request, then `steps` rounds of a model call and its tool result."""
    contents = [types.Content(role="user", parts=[types.Part(text=BIG)])]
    for i in range(steps):
        contents.append(
            types.Content(role="model", parts=[types.Part(text=f"call {i}")])
        )
        contents.append(
            types.Content(role="user", parts=[types.Part(text=f"result {i}")])
        )
    tool = types.Tool(
        function_declarations=[types.FunctionDeclaration(name="read_file")]
    )
    return LlmRequest(
        model="gemini-3.7-flash",
        contents=contents,
        config=types.GenerateContentConfig(system_instruction="role", tools=[tool]),
    )


def run(monkeypatch, caches, *requests):
    prompt_cache.reset()
    client = pytypes.SimpleNamespace(aio=pytypes.SimpleNamespace(caches=caches))
    monkeypatch.setattr(prompt_cache, "_gemini", lambda: client)
    monkeypatch.setattr(prompt_cache.models, "on_gemini", lambda: True)
    said = []
    monkeypatch.setattr(prompt_cache, "_say", lambda agent, text: said.append(text))

    async def calls():
        for request in requests:
            prompt_cache.use(CONTEXT, request)  # never waits for a cache
            await asyncio.sleep(0)
            await asyncio.sleep(0)

    asyncio.run(calls())
    return said


def test_later_calls_send_only_what_came_after_the_cached_prefix(monkeypatch):
    caches = FakeCaches()
    first, second = turn(1), turn(2)
    run(monkeypatch, caches, first, second)
    # the first call goes out whole; its prefix (all but the newest message) is cached
    assert first.config.cached_content is None and len(first.contents) == 3
    assert len(caches.created) == 1 and len(caches.created[0].contents) == 2
    assert caches.created[0].system_instruction == "role"
    # the next call refers to it and sends the rest, starting with a user turn
    assert second.config.cached_content == "cachedContents/1"
    assert second.config.system_instruction is None and second.config.tools is None
    assert [c.role for c in second.contents] == ["user", "model", "user"]


def test_a_prefix_too_small_to_cache_is_tried_again_once_it_grew(monkeypatch):
    caches = FakeCaches(error="400 The minimum token count to start caching is 4096")
    said = run(monkeypatch, caches, turn(1), turn(2))
    assert not said  # not an error worth showing: the turn just keeps going
    assert prompt_cache._turns["i1:engineer_features"].too_small > 0


def test_a_refused_cache_leaves_the_calls_whole(monkeypatch):
    second = turn(2)
    said = run(
        monkeypatch, FakeCaches(error="403 caching not allowed"), turn(1), second
    )
    assert second.config.cached_content is None and len(second.contents) == 5
    assert "not used this turn" in said[0]


def test_a_request_rejected_with_a_cache_is_sent_whole(monkeypatch):
    second = turn(2)
    run(monkeypatch, FakeCaches(), turn(1), second)
    assert second.config.cached_content
    assert prompt_cache.undo(second)  # what the model wrapper does on a rejection
    assert second.config.cached_content is None and len(second.contents) == 5
    assert second.config.system_instruction == "role" and second.config.tools
    later = turn(3)
    prompt_cache.use(CONTEXT, later)
    assert later.config.cached_content is None  # no more caching this turn
