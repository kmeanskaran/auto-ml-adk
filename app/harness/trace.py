"""Tracing: every agent step, tool call and code change, as clean logs.

TracePlugin is an ADK plugin: registered once on the App, it sees every agent,
model and tool call without the agents knowing about it. Per project folder
(a pipeline run, or the analyst's folder) it keeps:

  logs/activity.log        one plain line per step; the console's Activity panel
  logs/trace.jsonl         the same steps as records: agent, tool, arguments,
                           result, seconds, tokens
  logs/code/NNN-agent-path a copy of every file an agent writes, in order, so any
                           version of any script can be traced to who wrote it when

The same plain lines go to the server console through the "ml_team" logger.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import time
import warnings
from collections.abc import Awaitable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from google.adk.plugins.base_plugin import BasePlugin
from opentelemetry import context as otel_context

from app.harness.project import append_jsonl, home

T = TypeVar("T")

LOGGER = logging.getLogger("ml_team")
ACTIVITY = "logs/activity.log"
TRACE = "logs/trace.jsonl"
CODE = "logs/code"
MAX_TEXT = 2_000  # characters of any argument or reply kept in trace.jsonl
PIPELINE = "pipeline"  # the author of workflow steps that are not agents


def setup_logging(level: int = logging.INFO) -> None:
    """Plain, aligned lines on the console; the libraries underneath kept quiet."""
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        LOGGER.addHandler(handler)
        LOGGER.setLevel(level)
        LOGGER.propagate = False
    for noisy in ("LiteLLM", "litellm", "httpx", "google_adk", "google.adk"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # The console polls /api/view every 1.5 s (and the healthcheck calls it): keep
    # those out of the access log so the agents' lines stay readable.
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _SkipPolling) for f in access.filters):
        access.addFilter(_SkipPolling())
    warnings.filterwarnings("ignore", message=r"\[EXPERIMENTAL\]")
    try:
        import litellm

        litellm.suppress_debug_info = True
    except ImportError:
        pass


class _SkipPolling(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "GET /api/view" not in record.getMessage()


async def own_trace(job: Awaitable[T]) -> T:
    """Run a background job (a pipeline run, an analyst answer) in a trace of its own.

    Started inside a console request, it would inherit that request's trace, and with
    it the request's sampling decision: deployed, Google samples few requests, so most
    runs would send no spans to Cloud Trace. Detached, the job is a root trace, and
    ADK's spans (agents, model calls, tools) are always recorded.
    """
    token = otel_context.attach(otel_context.Context())
    try:
        return await job
    finally:
        otel_context.detach(token)


def note(agent: str, summary: str, /, **fields: Any) -> None:
    """Record one step in the folder the agent works in (the current run for the pipeline)."""
    root = home(agent)
    line = f"{agent:<18} {summary}"
    LOGGER.info(line)
    stamp = time.strftime("%H:%M:%S")
    with (root / ACTIVITY).open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp}  {line}\n")
    append_jsonl(root / TRACE, {"agent": agent, "summary": summary, **fields})


def activity(root: Path, limit: int = 80) -> list[str]:
    path = root / ACTIVITY
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]


def usage(root: Path) -> dict[str, Any]:
    """What a run cost, from its trace: time, model tokens and the share of them served
    from the prompt cache, and tool calls, in total and per agent turn.

    `minutes` is the wall clock from the first step to the last (waits for the human
    included); `agent_minutes` is the agents' own working time.
    """
    turns, first, last = [], None, None
    path = root / TRACE
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        at = record.get("at") or ""
        first, last = first or at, at or last
        if record.get("kind") == "agent" and record.get("event") == "end":
            turns.append(
                {
                    "agent": record.get("agent", ""),
                    "seconds": record.get("seconds", 0),
                    "tool_calls": record.get("tool_calls", 0),
                    "tokens": record.get("tokens", 0),
                    "cached_tokens": record.get("cached_tokens", 0),
                }
            )
    tokens = sum(t["tokens"] for t in turns)
    cached = sum(t["cached_tokens"] for t in turns)
    wall = 0.0
    if first and last:
        with contextlib.suppress(ValueError):
            wall = (
                datetime.fromisoformat(last) - datetime.fromisoformat(first)
            ).total_seconds()
    return {
        "minutes": round(wall / 60, 1),
        "agent_minutes": round(sum(t["seconds"] for t in turns) / 60, 1),
        "tokens": tokens,
        "cached_tokens": cached,
        "cached_pct": round(100 * cached / tokens) if tokens else 0,
        "tool_calls": sum(t["tool_calls"] for t in turns),
        "turns": turns,
    }


class TracePlugin(BasePlugin):
    """Logs agents, tool calls, model replies and every file written."""

    def __init__(self) -> None:
        super().__init__(name="trace")
        self._started: dict[str, float] = {}
        self._tokens: dict[str, int] = {}
        self._cached: dict[str, int] = {}  # of those, read from the prompt cache
        self._calls: dict[str, int] = {}

    # --- agents ------------------------------------------------------------------------

    async def before_agent_callback(self, *, agent, callback_context):
        if not _is_llm(agent):
            return None
        key = _key(callback_context)
        self._started[key] = time.monotonic()
        self._tokens[key] = self._cached[key] = self._calls[key] = 0
        note(agent.name, "▶ started", kind="agent", event="start")
        return None

    async def after_agent_callback(self, *, agent, callback_context):
        if not _is_llm(agent):
            return None
        key = _key(callback_context)
        seconds = round(time.monotonic() - self._started.pop(key, time.monotonic()), 1)
        tokens, calls = self._tokens.pop(key, 0), self._calls.pop(key, 0)
        cached = self._cached.pop(key, 0)
        share = f" ({100 * cached // tokens}% cached)" if tokens and cached else ""
        note(
            agent.name,
            f"■ finished in {seconds}s · {calls} tool calls · {tokens:,} tokens{share}",
            kind="agent",
            event="end",
            seconds=seconds,
            tool_calls=calls,
            tokens=tokens,
            cached_tokens=cached,
        )
        return None

    async def on_agent_error_callback(self, *, agent, callback_context, error):
        note(agent.name, f"✗ failed: {_one_line(error)}", kind="agent", event="error")
        return None

    # --- model ---------------------------------------------------------------------------

    async def after_model_callback(self, *, callback_context, llm_response):
        if llm_response.partial:
            return None
        key = _key(callback_context)
        usage = llm_response.usage_metadata
        if usage and usage.total_token_count:
            self._tokens[key] = self._tokens.get(key, 0) + usage.total_token_count
            self._cached[key] = self._cached.get(key, 0) + (
                usage.cached_content_token_count or 0
            )
        parts = (llm_response.content.parts if llm_response.content else None) or []
        text = " ".join(p.text.strip() for p in parts if p.text and not p.thought)
        if text:
            note(
                callback_context.agent_name,
                f"says: {_short(text, 160)}",
                kind="model",
                text=text[:MAX_TEXT],
            )
        return None

    async def on_model_error_callback(self, *, callback_context, llm_request, error):
        del llm_request
        note(
            callback_context.agent_name,
            f"✗ model error: {_one_line(error)}",
            kind="model",
            event="error",
        )
        return None

    # --- tools ---------------------------------------------------------------------------

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        del tool, tool_args
        self._started[_call(tool_context)] = time.monotonic()
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        agent = tool_context.agent_name
        started = self._started.pop(_call(tool_context), time.monotonic())
        seconds = round(time.monotonic() - started, 2)
        key = _key(tool_context)
        self._calls[key] = self._calls.get(key, 0) + 1
        result = result if isinstance(result, dict) else {"status": "ok"}
        extra = {}
        if tool.name in ("write_file", "write_and_run") and result.get("path"):
            extra["snapshot"] = _snapshot(home(agent), agent, result["path"])
        note(
            agent,
            _summary(tool.name, tool_args, result),
            kind="tool",
            tool=tool.name,
            args=_args(tool_args),
            status=result.get("status"),
            message=result.get("message"),
            seconds=seconds,
            **extra,
        )
        return None

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
        self._started.pop(_call(tool_context), None)
        note(
            tool_context.agent_name,
            f"✗ {tool.name} crashed: {_one_line(error)}",
            kind="tool",
            tool=tool.name,
            args=_args(tool_args),
            status="error",
        )
        return None  # the agent's own on_tool_error_callback turns it into a result


def _summary(tool: str, args: dict[str, Any], result: dict[str, Any]) -> str:
    """One line a person can scan: what the agent did and how it went."""
    if result.get("status") == "error":
        return f"✗ {tool} {_target(args)}: {_short(result.get('message', ''), 140)}"
    if tool == "run_python":
        line = f"ran {args.get('script')} → exit {result.get('exit_code')} in {result.get('seconds')}s"
        if result.get("exit_code") != 0:
            line += (
                f": {_last_line(result.get('stderr') or result.get('stdout') or '')}"
            )
        return line
    if tool == "write_file":
        lines = str(args.get("content", "")).count("\n") + 1
        return f"wrote {result.get('path')} ({lines} lines)"
    if tool == "write_and_run":
        lines = str(args.get("content", "")).count("\n") + 1
        line = (
            f"wrote and ran {result.get('path')} ({lines} lines) → exit "
            f"{result.get('exit_code')} in {result.get('seconds')}s"
        )
        if result.get("exit_code") != 0:
            line += (
                f": {_last_line(result.get('stderr') or result.get('stdout') or '')}"
            )
        return line
    if tool == "read_file":
        return f"read {args.get('path')}" + (
            " (unchanged, not resent)" if result.get("unchanged") else ""
        )
    if tool == "search":
        return f"searched {args.get('pattern')!r} ({len(result.get('matches', []))} matches)"
    if tool == "list_files":
        return (
            f"listed {args.get('folder_name')} ({len(result.get('files', []))} files)"
        )
    if tool == "check_stage":
        problems = result.get("problems") or []
        if not problems:
            return "checked hand-over → complete"
        return (
            f"checked hand-over → {len(problems)} problems: {_short(problems[0], 100)}"
        )
    if tool == "submit_summary":
        return f"summary: {_short(args.get('headline', ''), 140)}"
    if tool == "submit_receipt":
        return "receipt: " + " | ".join(_short(f, 70) for f in args.get("findings", []))
    if tool == "propose_plan":
        return f"proposed plan: {', '.join(args.get('models', []))} judged on {args.get('metric')}"
    if tool == "submit_review":
        return (
            f"review: {args.get('verdict')} · {len(args.get('findings', []))} findings · "
            f"{len(args.get('recommendations', []))} recommendations"
        )
    if tool == "show_chart":
        return f"chart: {result.get('chart', {}).get('title') or args.get('spec_path')}"
    return f"{tool} {_target(args)}".strip()


def _snapshot(root: Path, agent: str, relative: str) -> str:
    """Keep this version of the file under logs/code/, numbered in write order."""
    folder = root / CODE
    folder.mkdir(parents=True, exist_ok=True)
    number = sum(1 for _ in folder.iterdir()) + 1
    name = f"{number:03d}-{agent}-{relative.replace('/', '__')}"
    source = root / relative
    if source.is_file():
        (folder / name).write_bytes(source.read_bytes())
    return f"{CODE}/{name}"


def _args(args: dict[str, Any]) -> dict[str, Any]:
    """Arguments for the trace; file contents become a size and hash (the code is snapshotted)."""
    out = {}
    for key, value in args.items():
        if key == "content" and isinstance(value, str):
            out[key] = {
                "chars": len(value),
                "sha": hashlib.sha256(value.encode()).hexdigest()[:12],
            }
        elif isinstance(value, str):
            out[key] = value[:MAX_TEXT]
        else:
            out[key] = json.loads(json.dumps(value, default=str))
    return out


def _is_llm(agent: Any) -> bool:
    return hasattr(
        agent, "instruction"
    )  # LlmAgents; workflows and nodes log themselves


def _key(context: Any) -> str:
    return f"{context.invocation_id}:{context.agent_name}"


def _call(context: Any) -> str:
    return f"{context.invocation_id}:{context.function_call_id}"


def _target(args: dict[str, Any]) -> str:
    return str(args.get("path") or args.get("script") or args.get("folder_name") or "")


def _short(text: Any, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _one_line(error: Any) -> str:
    return _short(f"{type(error).__name__}: {error}", 200)


def _last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return _short(lines[-1], 140) if lines else ""
