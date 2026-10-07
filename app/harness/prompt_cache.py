"""Gemini prompt caching that never makes an agent wait.

Within one agent turn every model call re-sends the same growing prefix: the role
prompt, the tool declarations, the request (with the data at a glance) and every
earlier step. Once that prefix is big enough for Gemini to cache (about 4096 tokens;
a role prompt alone is smaller), it is stored as a cached content in the background,
and the following calls send only what came after it, so they are cheaper and faster.
As the turn grows, a longer prefix is cached the same way.

ADK's own ContextCacheConfig creates the cache inside the request, so the agent waits
for it (6 to 47 s per creation on Vertex, once a 6.5-minute step). Here no call ever
waits: until a cache is ready, requests go out whole (Gemini's implicit caching still
applies). A call whose start no longer matches the cached prefix (the history was
compacted) is sent whole too, and a turn whose cache Gemini refuses for any reason
but its size is sent whole from then on.

config/config.yml models.context_cache turns it on or off.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.genai import Client, types

from app import settings
from app.harness import models, trace

# Gemini caches from about 4096 tokens; at about 4 characters a token, with margin.
MIN_CHARS = 20_000
TTL_SECONDS = 1200  # a turn takes minutes; a cache left behind simply expires
# A longer prefix is cached once the part sent whole is this big: twice the cached
# part, and at least GROW_MIN_CHARS.
GROW, GROW_MIN_CHARS = 2.0, 40_000
MAX_TURNS = 50  # turns remembered; older ones are forgotten

log = logging.getLogger(__name__)
_client: Client | None = None


@dataclass
class _Cache:
    name: str
    count: int  # how many leading contents it holds
    prefix: str  # their fingerprint
    chars: int  # the size of everything it holds
    expires: float


@dataclass
class _Turn:
    caches: list[_Cache] = field(default_factory=list)
    building: bool = False
    refused: bool = False
    too_small: int = 0  # the size Gemini last found too small to cache


_turns: dict[str, _Turn] = {}
# What a request held before its prefix went by reference: id(request) -> (turn,
# system instruction, tools, tool config, contents), so undo() can send it whole.
_sent: dict[int, tuple] = {}


def _gemini() -> Client:
    global _client
    if _client is None:
        _client = Client()
    return _client


def _head(
    config: types.GenerateContentConfig, **extra
) -> types.CreateCachedContentConfig:
    """The role prompt, the tool declarations and their config (ADK sends tools as
    declarations, never as Python callables), plus what else the cache holds."""
    tools = [t for t in config.tools or [] if isinstance(t, types.Tool)]
    return types.CreateCachedContentConfig(
        system_instruction=config.system_instruction,
        tools=tools or None,
        tool_config=config.tool_config,
        **extra,
    )


def _chars(item: types.Content | types.CreateCachedContentConfig) -> int:
    return len(item.model_dump_json(exclude_none=True))


def _fingerprint(contents: list[types.Content]) -> str:
    digest = hashlib.sha256()
    for content in contents:
        digest.update(content.model_dump_json(exclude_none=True).encode())
    return digest.hexdigest()[:16]


def _say(agent: str, summary: str) -> None:
    """One line in the run's activity log, so the human sees the cache at work."""
    try:
        trace.note(agent, summary, kind="cache")
    except Exception:
        log.info("%s %s", agent, summary)


async def _create(
    turn: _Turn,
    agent: str,
    model: str,
    config: types.GenerateContentConfig,
    contents: list[types.Content],
    chars: int,
) -> None:
    global _client
    started = time.time()
    try:
        cache = await _gemini().aio.caches.create(
            model=model,
            config=_head(config, contents=contents, ttl=f"{TTL_SECONDS}s"),
        )
    except Exception as error:
        text = str(error)
        if getattr(error, "code", None) == 401:  # a stale login: a fresh one next time
            _client = None
        elif "minimum token count" in text:  # too small yet: try once it has grown
            turn.too_small = chars
        else:
            turn.refused = True
            _say(agent, f"⚡ prompt cache not used this turn: {text[:200]}")
    else:
        usage = getattr(cache, "usage_metadata", None)
        tokens = getattr(usage, "total_token_count", None)
        turn.caches.append(
            _Cache(
                name=str(cache.name),
                count=len(contents),
                prefix=_fingerprint(contents),
                chars=chars,
                expires=time.time() + TTL_SECONDS,
            )
        )
        _say(
            agent,
            f"⚡ prompt cache: {tokens or '?'} tokens stored in the background "
            f"({time.time() - started:.0f}s); later calls send only the new part",
        )
    finally:
        turn.building = False


def _turn(callback_context: CallbackContext) -> _Turn:
    key = (
        f"{getattr(callback_context, 'invocation_id', '')}:"
        f"{getattr(callback_context, 'agent_name', '')}"
    )
    if key not in _turns and len(_turns) >= MAX_TURNS:
        _turns.pop(next(iter(_turns)))
    return _turns.setdefault(key, _Turn())


def use(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
    """before_model_callback (last): send the turn's cached prefix by reference when
    a cache matches it; start caching a longer prefix in the background when that
    pays off. Never waits."""
    config, contents = llm_request.config, list(llm_request.contents or [])
    if (
        not settings.load().context_cache
        or not models.on_gemini()
        or config is None
        or config.cached_content
        or not contents
    ):
        return None
    turn = _turn(callback_context)
    if turn.refused:
        return None
    now = time.time()
    usable = next(
        (
            cache
            for cache in reversed(turn.caches)  # the longest prefix first
            if cache.expires - now > 60
            and len(contents) > cache.count
            and _fingerprint(contents[: cache.count]) == cache.prefix
        ),
        None,
    )
    # A cache holds all but the newest message, so the part sent with it starts with
    # a user turn (a tool result or the request), as a conversation does.
    prefix = contents[:-1]
    if prefix and prefix[-1].role != "model":
        prefix = []  # the newest message should follow a model turn; wait for one
    total = _chars(_head(config)) + sum(_chars(c) for c in prefix)
    cached = usable.chars if usable else 0
    if (
        prefix
        and not turn.building
        and total >= max(MIN_CHARS, turn.too_small * 1.5)
        and (not usable or total - cached >= max(GROW_MIN_CHARS, GROW * cached))
    ):
        turn.building = True
        asyncio.get_running_loop().create_task(
            _create(
                turn,
                str(getattr(callback_context, "agent_name", "") or "agent"),
                str(llm_request.model or models.base()),
                config.model_copy(),
                prefix,
                total,
            )
        )
    if usable:
        while len(_sent) >= 20:  # only requests in flight can need undoing
            _sent.pop(next(iter(_sent)))
        _sent[id(llm_request)] = (
            turn,
            config.system_instruction,
            config.tools,
            config.tool_config,
            contents,
        )
        # The cache holds these; Gemini refuses a request that repeats them.
        config.cached_content = usable.name
        config.system_instruction = None
        config.tools = None
        config.tool_config = None
        llm_request.contents = contents[usable.count :]
    return None


def undo(llm_request: LlmRequest) -> bool:
    """Gemini rejected a request that used a cache: put back what it held, so the
    call can be sent whole, and stop caching this turn. False if it used none."""
    saved = _sent.pop(id(llm_request), None)
    if saved is None or llm_request.config is None:
        return False
    turn, system, tools, tool_config, contents = saved
    turn.refused = True
    llm_request.config.cached_content = None
    llm_request.config.system_instruction = system
    llm_request.config.tools = tools
    llm_request.config.tool_config = tool_config
    llm_request.contents = contents
    return True


def reset() -> None:
    """Forget every cache (tests)."""
    global _client
    _client = None
    _turns.clear()
    _sent.clear()
