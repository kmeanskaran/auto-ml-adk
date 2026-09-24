"""The ML team: one package per role, plus shared config and storage.

Roles live under ``team.<role>``. Each role keeps its tools, ops, and prompts
together. ADK orchestration stays in the top-level ``runtime`` package.
"""

from __future__ import annotations

import warnings

from team.config import load_config

load_config()
warnings.filterwarnings("ignore", message=r"\[EXPERIMENTAL\]")

__version__ = "0.1.0"
