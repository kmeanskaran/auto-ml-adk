"""Background runs the UI drives: the pipeline (with pauses) and the analyst chat.

Both use the process-wide session service: a SQLite file locally, Agent Platform
sessions when deployed. The pipeline's position (its session, run folder and the
review it is waiting on) is saved to runs/console.json, so a server restart
while a review is open resumes that review instead of losing the run. clear() is the
cold start: it removes every run, model, feature view, session and chat message.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from google.adk.apps import App
from google.adk.runners import Runner
from google.genai import types

from app.app_utils import services
from app.harness import feature_store, history, project, registry, trace
from app.harness.trace import TracePlugin

USER = "human"
AGENT_NAMES = (
    "analyst_profile",
    "engineer_features",
    "skeptic_features",
    "engineer_model",
    "skeptic_model",
)
MAX_CHAT_MESSAGES = 200
MAX_CHARTS = 3  # per answer: a report may need a few diagrams


def _text(message: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=message)])


def _save(name: str, payload: dict[str, Any]) -> None:
    path = project.RUNS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)


def _load(name: str) -> dict[str, Any]:
    path = project.RUNS / name
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        return {}


@dataclass
class Pause:
    interrupt_id: str
    invocation_id: str
    payload: dict[str, Any]


@dataclass
class PipelineRun:
    """One pipeline run at a time: start it, watch it, answer its reviews, pause,
    resume or restart it."""

    STATE = "console.json"

    status: str = "idle"  # idle | running | paused | waiting | done | error
    active: str = ""  # the agent working right now
    pause: Pause | None = None
    project: str = ""
    error: str = ""
    session_id: str = ""
    runner: Runner | None = None
    task: asyncio.Task | None = None
    # Cleared to pause: the run stops pulling events, so it holds after the model call
    # or script in flight finishes, and nothing more happens until it is set again.
    go: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.go.set()

    @classmethod
    def restore(cls) -> PipelineRun:
        """Pick up where the last server left off."""
        saved = _load(cls.STATE)
        run = cls(
            status=saved.get("status", "idle"),
            project=saved.get("project", ""),
            error=saved.get("error", ""),
            session_id=saved.get("session_id", ""),
            pause=Pause(**saved["pause"]) if saved.get("pause") else None,
        )
        if run.project:
            project.use(run.project)
        if run.status in ("running", "paused"):  # it died with the old process
            run.status, run.pause = "error", None
            run.error = (
                "The server restarted while an agent was working. The finished stages "
                f"are kept in runs/{run.project}; start a new run to continue."
            )
            run._save()
        elif run.status == "waiting" and not run.pause:
            run.status = "error"
        return run

    def _runner(self) -> Runner:
        if self.runner is None:
            from app.agent import app

            self.runner = Runner(
                app=app, session_service=services.get_session_service()
            )
        return self.runner

    def _save(self) -> None:
        _save(
            self.STATE,
            {
                "status": self.status,
                "project": self.project,
                "error": self.error,
                "session_id": self.session_id,
                "pause": asdict(self.pause) if self.pause else None,
            },
        )

    async def start(self, feedback: str = "") -> None:
        if self.status in ("running", "paused", "waiting"):
            raise RuntimeError("A pipeline run is already in progress.")
        runner = self._runner()
        self.status, self.active, self.pause, self.project, self.error = (
            "running",
            "",
            None,
            "",
            "",
        )
        session = await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=USER,
            state={project.FEEDBACK_KEY: feedback.strip()},
        )
        self.session_id = session.id
        self._launch(_text("Run the pipeline."), None)

    async def answer(self, answer: dict[str, Any]) -> None:
        if self.status != "waiting" or self.pause is None:
            raise RuntimeError("Nothing is waiting for a decision.")
        pause, self.pause = self.pause, None
        response = types.FunctionResponse(
            id=pause.interrupt_id,
            name="adk_request_input",
            response=answer,
        )
        self._launch(
            types.Content(role="user", parts=[types.Part(function_response=response)]),
            pause.invocation_id,
        )

    def pause_run(self) -> None:
        if self.status != "running":
            raise RuntimeError("Only a running pipeline can be paused.")
        self.go.clear()
        self.status = "paused"
        self._save()
        trace.note(trace.PIPELINE, "⏸ paused by the human", kind="pause")

    def resume_run(self) -> None:
        if self.status != "paused":
            raise RuntimeError("The pipeline is not paused.")
        self.status = "running"
        self._save()
        self.go.set()
        trace.note(trace.PIPELINE, "▶ resumed by the human", kind="pause")

    async def restart(self) -> None:
        """Stop this run wherever it is and start a new one with the same feedback."""
        feedback = ""
        if self.project:
            brief = project.read_json(project.RUNS / self.project / history.BRIEF) or {}
            feedback = brief.get("feedback") or ""
            trace.note(trace.PIPELINE, "↻ restarted by the human", kind="pause")
        await self.stop()
        self.status = "idle"
        await self.start(feedback)

    async def stop(self) -> None:
        """Cancel the run in flight (its running script too) and forget its pause."""
        task, self.task = self.task, None
        self.go.set()
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.active, self.pause = "", None

    def _launch(self, message: types.Content, invocation_id: str | None) -> None:
        self.status = "running"
        self.go.set()
        self._save()
        self.task = asyncio.create_task(self._consume(message, invocation_id))

    async def _consume(self, message: types.Content, invocation_id: str | None) -> None:
        try:
            async for event in self._runner().run_async(
                user_id=USER,
                session_id=self.session_id,
                new_message=message,
                invocation_id=invocation_id,
            ):
                if event.author in AGENT_NAMES:
                    self.active = event.author
                # The run folder is named in the session state by the start step.
                run = (event.actions.state_delta or {}).get(project.RUN_KEY)
                if run and run != self.project:
                    self.project = project.use(run).name
                    self._save()
                for call in event.get_function_calls():
                    if call.name == "adk_request_input":
                        args = call.args or {}
                        self.pause = Pause(
                            call.id or "",
                            event.invocation_id,
                            args.get("payload") or {},
                        )
                await self.go.wait()  # paused: hold here until resumed
            self.active = ""
            self.status = "waiting" if self.pause else "done"
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # shown in the UI; start a new run to retry
            self.status, self.error, self.active = (
                "error",
                f"{type(exc).__name__}: {exc}",
                "",
            )
        self._save()


@dataclass
class AnalystChat:
    """Questions about the data, answered by the analyst. It never trains."""

    STATE = f"{project.ANALYSIS}/chat.json"

    status: str = "idle"
    messages: list[dict[str, Any]] = field(default_factory=list)
    session_id: str = ""
    runner: Runner | None = None
    task: asyncio.Task | None = None

    @classmethod
    def restore(cls) -> AnalystChat:
        saved = _load(cls.STATE)
        messages = [  # answers saved before clean() existed render cleanly too
            {**m, "text": clean(m.get("text", ""))} if m.get("role") == "analyst" else m
            for m in saved.get("messages", [])
        ]
        return cls(messages=messages, session_id=saved.get("session_id", ""))

    def _save(self) -> None:
        self.messages = self.messages[-MAX_CHAT_MESSAGES:]
        _save(self.STATE, {"session_id": self.session_id, "messages": self.messages})

    async def ask(self, question: str) -> None:
        if self.status == "running":
            raise RuntimeError("The analyst is still answering.")
        if self.runner is None:
            from app.agents import analyst

            self.runner = Runner(
                app=App(name="analyst", root_agent=analyst(), plugins=[TracePlugin()]),
                session_service=services.get_session_service(),
            )
        # Keep the conversation's session across restarts; start one if it is gone.
        service = self.runner.session_service
        if not self.session_id or not await service.get_session(
            app_name="analyst", user_id=USER, session_id=self.session_id
        ):
            session = await service.create_session(app_name="analyst", user_id=USER)
            self.session_id = session.id
        self.messages.append({"role": "you", "text": question})
        self.status = "running"
        self._save()
        self.task = asyncio.create_task(self._answer(question))

    async def _answer(self, question: str) -> None:
        assert self.runner is not None
        reply, charts, started = "", [], time.time()
        try:
            async for event in self.runner.run_async(
                user_id=USER, session_id=self.session_id, new_message=_text(question)
            ):
                for response in event.get_function_responses():
                    result = response.response or {}
                    if response.name == "show_chart" and result.get("status") == "ok":
                        charts.append(result["chart"])
                if (
                    event.content
                    and event.content.parts
                    and not event.get_function_calls()
                    and not event.get_function_responses()
                ):
                    words = "".join(
                        p.text or "" for p in event.content.parts if not p.thought
                    ).strip()
                    reply = words or reply
            self.messages.append(
                {
                    "role": "analyst",
                    "text": clean(reply) or "(no answer)",
                    "charts": (charts or _drawn_since(started))[-MAX_CHARTS:],
                }
            )
        except Exception as exc:
            self.messages.append(
                {"role": "error", "text": f"{type(exc).__name__}: {exc}"}
            )
        self.status = "idle"
        self._save()


async def clear() -> None:
    """Cold start: delete every run, model version, feature view, session and chat
    message, so the next run starts from nothing. Refused while an agent works."""
    if PIPELINE.status == "running" or CHAT.status == "running":
        raise RuntimeError(
            "An agent is still working. Pause it (or wait until it finishes), then clear."
        )
    await PIPELINE.stop()  # a paused run is abandoned
    service = services.get_session_service()
    for app_name in (PIPELINE._runner().app_name, "analyst"):
        listed = await service.list_sessions(app_name=app_name, user_id=USER)
        for session in listed.sessions:
            await service.delete_session(
                app_name=app_name, user_id=USER, session_id=session.id
            )
    for folder in (project.RUNS, registry.REGISTRY, feature_store.STORE):
        project.empty(folder)
    registry.forget()
    project.forget()
    PIPELINE.__init__(runner=PIPELINE.runner)
    CHAT.__init__(runner=CHAT.runner)


_TYPOGRAPHY = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u2009": " ",
        "\u202f": " ",
        "\u200b": "",
    }
)
_SYMBOLS = {  # LaTeX commands the analyst writes, as the symbols they stand for
    r"\times": "\u00d7",
    r"\cdot": "\u00b7",
    r"\geq": "\u2265",
    r"\leq": "\u2264",
    r"\ge": "\u2265",
    r"\le": "\u2264",
    r"\approx": "\u2248",
    r"\neq": "\u2260",
}


def clean(text: str) -> str:
    """The analyst's answer as the console renders it: plain markdown. Math markup
    becomes plain text, escaped symbols lose the backslash, odd dashes and spaces
    become plain ones."""
    text = text.translate(_TYPOGRAPHY)
    text = re.sub(r"\\[\[\]()]", "", text)  # \[ \] \( \) delimiters
    text = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"_\{(\w+)\}", r"_\1", text)  # x_{FP} -> x_FP
    for _ in range(3):  # nested fractions, innermost first
        text = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1) / (\2)", text)
    text = re.sub(r"\(([\w.]+)\)", r"\1", text)  # "(2) / (3)" -> "2 / 3"
    for latex, plain in _SYMBOLS.items():
        text = re.sub(re.escape(latex) + r"(?![a-zA-Z])", plain, text)
    text = re.sub(r"\\([$%&_#])", r"\1", text)  # \$ -> $
    # inline math $x >= 3$ loses its dollars; amounts like $5 never start a pair
    text = re.sub(r"\$(?=[^\d\s$])([^$\n]{1,80}?)\$", r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _drawn_since(started: float) -> list[dict[str, Any]]:
    """Charts the analyst's scripts wrote during this answer, when it drew them but
    did not call show_chart: what it drew is shown, never a chart it did not draw."""
    from app.harness.tools import _chart

    folder = project.folder(project.ANALYSIS)
    drawn = sorted(
        (p for p in (folder / "charts").glob("*.json") if p.stat().st_mtime >= started),
        key=lambda p: p.stat().st_mtime,
    )
    charts = []
    for path in drawn:
        chart, issue = _chart(folder, str(path.relative_to(folder)))
        if not issue:
            charts.append(chart)
    return charts


PIPELINE = PipelineRun.restore()
CHAT = AnalystChat.restore()
