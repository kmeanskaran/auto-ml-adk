"""Deterministic model used when Ollama is not selected.

It still goes through ADK's tool loop. ``NEEDED:`` is written onto the request
by the guidance callback, and this model does only that.
"""

from __future__ import annotations

import re
import uuid

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from typing_extensions import override

_NEEDED = re.compile(r"NEEDED:\s*(call:([A-Za-z0-9_]+)|stop:[^\n]*)")


class ScriptedLlm(BaseLlm):
  """Follows the harness NEEDED line and never improvises a tool call."""

  @override
  async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False):
    del stream
    yield LlmResponse(content=_content_for(llm_request))


def _content_for(llm_request: LlmRequest) -> types.Content:
  action = _needed_action(llm_request)
  if action and action.startswith("call:"):
    tool_name = action.split(":", 1)[1]
    return types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(
                    name=tool_name,
                    args={},
                    id=f"call-{uuid.uuid4().hex[:12]}",
                )
            )
        ],
    )
  if action and "approval" in action:
    text = "Waiting for a human to approve training."
  elif action and "target met" in action:
    text = "The metric reached the target, so the loop stops."
  elif action and "planned" in action:
    text = "The trial missed the target. An improvement plan is recorded."
  else:
    text = "Step complete."
  return types.Content(role="model", parts=[types.Part(text=text)])


def _needed_action(llm_request: LlmRequest) -> str | None:
  blobs = [_stringify(llm_request.config.system_instruction)]
  for content in llm_request.contents or []:
    for part in content.parts or []:
      if part.text:
        blobs.append(part.text)
  matches = _NEEDED.findall("\n".join(blobs))
  if not matches:
    return None
  full, tool_name = matches[-1]
  if tool_name:
    return f"call:{tool_name}"
  return full


def _stringify(value: object) -> str:
  if value is None:
    return ""
  if isinstance(value, str):
    return value
  if isinstance(value, types.Content):
    return "\n".join(part.text or "" for part in value.parts or [])
  return str(value)
