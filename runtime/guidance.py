"""Tell each specialist the single next tool call for this visit.

The line is appended to the model request as ``NEEDED:``. The scripted model
follows it exactly. A local LLM sees the same instruction, so a small model
does not have to rediscover the workflow.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from team.data_engineer.role import PRIMARY_TOOLS as DATA_ENGINEER_TOOLS
from team.data_scientist.role import PRIMARY_TOOLS as DATA_SCIENTIST_TOOLS
from team.ml_engineer.role import PRIMARY_TOOLS as ML_ENGINEER_TOOLS
from team.product_manager.role import NAME as PRODUCT_MANAGER
from team.product_manager.role import TOOLS as PRODUCT_MANAGER_TOOLS
from team.researcher.role import PRIMARY_TOOLS as RESEARCHER_TOOLS

PRIMARY_TOOLS: dict[str, list[str]] = {
    PRODUCT_MANAGER: list(PRODUCT_MANAGER_TOOLS),
    **RESEARCHER_TOOLS,
    **DATA_ENGINEER_TOOLS,
    **DATA_SCIENTIST_TOOLS,
    **ML_ENGINEER_TOOLS,
}


def on_agent_start(callback_context: CallbackContext) -> types.Content | None:
  """Count how many times this agent has been entered in the invocation.

  A resumed approval does not count as a new visit, otherwise training would
  be requested twice.
  """
  if _awaiting_confirmation(callback_context):
    return None
  key = _entry_key(callback_context.agent_name, callback_context.invocation_id)
  current = int(callback_context.state.get(key) or 0)
  callback_context.state[key] = current + 1
  return None


def on_before_model(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> None:
  state = callback_context.state.to_dict()
  action = next_action(
      callback_context.agent_name,
      callback_context.invocation_id,
      list(callback_context.session.events),
      state,
  )
  line = (
      f"NEEDED: {action}\n"
      "Follow NEEDED exactly. call:TOOL means call that tool once with no"
      " arguments. stop means reply in one sentence and do not call a tool."
  )
  note = str(state.get("last_plan") or "").strip()
  if note:
    line += (
        f"\nSME_NOTE: {note}\n"
        "SME_NOTE is a requirement for this stage, not a suggestion. "
        "If it mentions categorical values, including a count of zero, "
        "name the categorical column and say whether it is the target. "
        "Do not invent categorical features. "
        "When every predictor is numeric, keep them numeric, skip one-hot "
        "encoding, and prefer logistic regression. "
        "When the tool result includes mean, median, mode, std, or outliers, "
        "cite those numbers in the reply."
    )
  _append_system(llm_request, line)
  return None


def next_action(
    agent_name: str,
    invocation_id: str,
    events: list[Any],
    state: dict[str, Any],
) -> str:
  tools = PRIMARY_TOOLS.get(agent_name)
  if not tools:
    return "stop:nothing to do"
  if _last_is_confirmation_error(events, invocation_id, tools):
    return "stop:awaiting human approval"

  entries = int(state.get(_entry_key(agent_name, invocation_id)) or 1)
  successes = {
      tool: _success_count(events, invocation_id, tool) for tool in tools
  }
  if agent_name == "evaluation_agent":
    return _evaluation_action(entries, successes, events, invocation_id)
  if agent_name in {"design_agent", "report_agent"}:
    return _sequence_action(entries, successes, tools)
  tool = tools[0]
  if successes[tool] < entries:
    return f"call:{tool}"
  if successes[tool] >= 6:
    return "stop:safety cap"
  return "stop:done"


def _evaluation_action(
    entries: int,
    successes: dict[str, int],
    events: list[Any],
    invocation_id: str,
) -> str:
  if successes["evaluate_model"] < entries:
    return "call:evaluate_model"
  last = _last_success(events, invocation_id, "evaluate_model")
  target_met = bool(last and _response_body(last).get("target_met"))
  if target_met:
    if successes["exit_loop"] < entries:
      return "call:exit_loop"
    return "stop:target met"
  if successes["record_improvement_plan"] < entries:
    return "call:record_improvement_plan"
  return "stop:improvement planned"


def _sequence_action(entries: int, successes: dict[str, int], tools: list[str]) -> str:
  for tool in tools:
    if successes[tool] < entries:
      return f"call:{tool}"
  return "stop:done"


def _entry_key(agent_name: str, invocation_id: str) -> str:
  return f"entries:{agent_name}:{invocation_id}"


def _success_count(events: list[Any], invocation_id: str, tool_name: str) -> int:
  return sum(
      1
      for event in events
      if getattr(event, "invocation_id", None) == invocation_id
      for response in event.get_function_responses()
      if response.name == tool_name and _is_success(response)
  )


def _last_success(events: list[Any], invocation_id: str, tool_name: str) -> Any | None:
  found = None
  for event in events:
    if getattr(event, "invocation_id", None) != invocation_id:
      continue
    for response in event.get_function_responses():
      if response.name == tool_name and _is_success(response):
        found = response
  return found


def _last_is_confirmation_error(
    events: list[Any],
    invocation_id: str,
    tools: list[str],
) -> bool:
  last = None
  for event in events:
    if getattr(event, "invocation_id", None) != invocation_id:
      continue
    for response in event.get_function_responses():
      if response.name in tools:
        last = response
  if last is None or _is_success(last):
    return False
  text = str(_response_body(last).get("error") or "").lower()
  return "confirmation" in text or "rejected" in text


def _awaiting_confirmation(callback_context: CallbackContext) -> bool:
  pending: set[str] = set()
  answered: set[str] = set()
  for event in callback_context.session.events:
    if event.invocation_id != callback_context.invocation_id:
      continue
    for call in event.get_function_calls():
      if call.name == "adk_request_confirmation" and call.id:
        pending.add(call.id)
    if event.author == "user":
      for response in event.get_function_responses():
        if response.name == "adk_request_confirmation" and response.id:
          answered.add(response.id)
  return bool(pending - answered)


def _is_success(response: Any) -> bool:
  body = _response_body(response)
  error = body.get("error")
  if not error:
    return True
  text = str(error).lower()
  return "confirmation" not in text and "rejected" not in text


def _response_body(response: Any) -> dict[str, Any]:
  body = getattr(response, "response", None) or {}
  if isinstance(body, dict) and set(body.keys()) == {"response"} and isinstance(
      body["response"], str
  ):
    try:
      import json

      parsed = json.loads(body["response"])
    except json.JSONDecodeError:
      return body
    return parsed if isinstance(parsed, dict) else body
  return body if isinstance(body, dict) else {}


def _append_system(llm_request: LlmRequest, line: str) -> None:
  current = llm_request.config.system_instruction
  if current is None or current == "":
    llm_request.config.system_instruction = line
    return
  if isinstance(current, str):
    llm_request.config.system_instruction = current + "\n" + line
    return
  if isinstance(current, types.Content):
    parts = list(current.parts or [])
    parts.append(types.Part(text=line))
    current.parts = parts
    return
  llm_request.config.system_instruction = str(current) + "\n" + line
