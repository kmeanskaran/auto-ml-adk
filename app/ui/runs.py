"""Background runs the UI drives: the pipeline (with pauses) and the analyst chat."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.harness import project

USER = "human"
AGENT_NAMES = (
    "analyst_profile",
    "engineer_features",
    "skeptic_features",
    "engineer_model",
    "skeptic_model",
)


def _text(message: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=message)])


@dataclass
class Pause:
    interrupt_id: str
    invocation_id: str
    payload: dict[str, Any]


@dataclass
class PipelineRun:
    """One pipeline run at a time: start it, watch it, answer its reviews."""

    status: str = "idle"  # idle | running | waiting | done | error
    active: str = ""  # the agent working right now
    pause: Pause | None = None
    project: str = ""
    error: str = ""
    runner: Runner | None = None
    session_id: str = ""
    task: asyncio.Task | None = None

    async def start(self) -> None:
        if self.status in ("running", "waiting"):
            raise RuntimeError("A pipeline run is already in progress.")
        from app.agent import app

        self.__init__()
        self.runner = Runner(app=app, session_service=InMemorySessionService())
        session = await self.runner.session_service.create_session(
            app_name=app.name, user_id=USER
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

    def _launch(self, message: types.Content, invocation_id: str | None) -> None:
        self.status = "running"
        self.task = asyncio.create_task(self._consume(message, invocation_id))

    async def _consume(self, message: types.Content, invocation_id: str | None) -> None:
        assert self.runner is not None
        try:
            async for event in self.runner.run_async(
                user_id=USER,
                session_id=self.session_id,
                new_message=message,
                invocation_id=invocation_id,
            ):
                if event.author in AGENT_NAMES:
                    self.active = event.author
                if not self.project and project.current_name():
                    self.project = project.current_name()
                for call in event.get_function_calls():
                    if call.name == "adk_request_input":
                        args = call.args or {}
                        self.pause = Pause(
                            call.id or "",
                            event.invocation_id,
                            args.get("payload") or {},
                        )
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


@dataclass
class AnalystChat:
    """Questions about the data, answered by the analyst. It never trains."""

    status: str = "idle"
    messages: list[dict[str, Any]] = field(default_factory=list)
    runner: Runner | None = None
    session_id: str = ""
    task: asyncio.Task | None = None

    async def ask(self, question: str) -> None:
        if self.status == "running":
            raise RuntimeError("The analyst is still answering.")
        if self.runner is None:
            from app.agents import analyst

            self.runner = Runner(
                app=App(name="analyst", root_agent=analyst()),
                session_service=InMemorySessionService(),
            )
            session = await self.runner.session_service.create_session(
                app_name="analyst", user_id=USER
            )
            self.session_id = session.id
        self.messages.append({"role": "you", "text": question})
        self.status = "running"
        self.task = asyncio.create_task(self._answer(question))

    async def _answer(self, question: str) -> None:
        assert self.runner is not None
        reply, charts = "", []
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
                    "text": reply or "(no answer)",
                    "charts": charts[-1:],
                }
            )
        except Exception as exc:
            self.messages.append(
                {"role": "error", "text": f"{type(exc).__name__}: {exc}"}
            )
        self.status = "idle"


PIPELINE = PipelineRun()
CHAT = AnalystChat()
