"""Process settings for the HTTP service.

Values come from config/config.yml. Environment variables override them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from team.config import load_config


@dataclass
class Settings:
  host: str = "127.0.0.1"
  port: int = 8000
  log_level: str = "INFO"
  cors_origins: list[str] = field(default_factory=list)
  root: Path | None = None
  model: str | None = None
  max_upload_bytes: int = 20_000_000


def load_settings() -> Settings:
  cfg = load_config()
  raw_cors = os.environ.get("ML_API_CORS")
  if raw_cors is None:
    origins = list(cfg.api.cors_origins)
  else:
    origins = [item.strip() for item in raw_cors.split(",") if item.strip()]
  root_env = os.environ.get("ML_DATA_ROOT") or os.environ.get("ML_RUNTIME_ROOT")
  model = os.environ.get("ML_RUNTIME_MODEL") or cfg.litellm_model
  return Settings(
      host=os.environ.get("ML_API_HOST", cfg.api.host),
      port=int(os.environ.get("ML_API_PORT", str(cfg.api.port))),
      log_level=os.environ.get("ML_LOG_LEVEL", cfg.api.log_level),
      cors_origins=origins,
      root=Path(root_env).expanduser() if root_env else None,
      model=model,
      max_upload_bytes=int(
          os.environ.get("ML_API_MAX_UPLOAD_BYTES", str(cfg.api.max_upload_bytes))
      ),
  )
