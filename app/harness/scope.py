"""Binds every agent step to its session's run folder (project.RUN_KEY in the state).

Registered first on the App, before the trace, so a tool call, a model reply and the
trace line about it all land in the folder of the run that made them, even with
several sessions in one process or a run resumed by another server.
"""

from __future__ import annotations

from google.adk.plugins.base_plugin import BasePlugin

from app.harness import project


class RunScope(BasePlugin):
    def __init__(self) -> None:
        super().__init__(name="run_scope")

    async def before_agent_callback(self, *, agent, callback_context):
        del agent
        project.bind(callback_context.state)

    async def before_model_callback(self, *, callback_context, llm_request):
        del llm_request
        project.bind(callback_context.state)

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        del tool, tool_args
        project.bind(tool_context.state)
