"""Allowlist pandas snippets the data engineer may run on a frame."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from team.config import load_config

_FORBIDDEN_SNIPPET = (
    "import ",
    "__",
    "open(",
    "exec(",
    "eval(",
    "globals(",
    "locals(",
    "getattr(",
    "setattr(",
    "subprocess",
    "socket",
    "os.",
    "sys.",
    "shutil",
    "pathlib",
    "compile(",
)


def execute_snippet(frame: pd.DataFrame, code: str) -> dict[str, Any]:
  """Run a short analysis snippet against a dataframe.

  This is an allowlist sandbox for the PoC, not a hardened isolate. Imports,
  file access, and dunder access are rejected before exec.
  """
  limit = load_config().sandbox.max_chars
  if len(code) > limit:
    raise ValueError(f"Analysis snippets are limited to {limit} characters.")
  lowered = code.lower()
  for token in _FORBIDDEN_SNIPPET:
    if token in lowered:
      raise ValueError(f"Snippet rejected by policy: contains {token.strip()!r}.")
  safe_builtins = {
      "len": len,
      "range": range,
      "min": min,
      "max": max,
      "sum": sum,
      "abs": abs,
      "round": round,
      "sorted": sorted,
      "list": list,
      "dict": dict,
      "float": float,
      "int": int,
      "str": str,
      "print": print,
      "enumerate": enumerate,
      "zip": zip,
      "bool": bool,
      "True": True,
      "False": False,
      "None": None,
  }
  local: dict[str, Any] = {}
  exec(  # noqa: S102 — the token filter and builtins allowlist are the PoC boundary.
      code,
      {"__builtins__": safe_builtins, "df": frame, "pd": pd, "np": np},
      local,
  )
  result = local.get("result")
  return {"status": "ok", "result": _snippet_result(result)}


def _snippet_result(result: Any) -> Any:
  if isinstance(result, pd.DataFrame):
    return result.head(20).to_dict(orient="records")
  if isinstance(result, pd.Series):
    return {str(key): value for key, value in result.head(20).to_dict().items()}
  if isinstance(result, np.generic):
    return result.item()
  if isinstance(result, (str, int, float, bool)) or result is None:
    return result
  if isinstance(result, dict):
    return {str(key): _snippet_result(value) for key, value in result.items()}
  if isinstance(result, (list, tuple)):
    return [_snippet_result(value) for value in list(result)[:20]]
  return str(result)
