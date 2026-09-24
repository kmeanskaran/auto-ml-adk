"""Load instruction text from a role's ``prompts/`` folder."""

from __future__ import annotations

from importlib.resources import files


def load_prompt(package: str, filename: str) -> str:
  """Read ``prompts/<filename>`` from an installed team role package."""
  return files(package).joinpath("prompts", filename).read_text(encoding="utf-8").strip()
