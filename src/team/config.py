"""Load config/config.yml and apply process overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yml"

_CACHE: dict[str, "ProjectConfig"] = {}


@dataclass(frozen=True)
class ProviderConfig:
  name: str
  model: str
  api_base: str
  litellm_prefix: str
  temperature: float


@dataclass(frozen=True)
class SessionConfig:
  backend: str
  app_name: str
  db_path: str


@dataclass(frozen=True)
class RuntimeConfig:
  root: str
  identity: str
  user: str
  max_improvement_loops: int
  approve_training: bool
  blocked_tools: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentDefaults:
  objective: str
  target_column: str
  metric: str
  target_value: float


@dataclass(frozen=True)
class SampleDataConfig:
  rows: int
  seed: int
  missing_rate: float
  missing_seed: int
  missing_columns: tuple[str, ...]


@dataclass(frozen=True)
class EstimatorConfig:
  max_iter: int = 400
  class_weight: str | None = None
  n_estimators: int = 40
  max_depth: int = 8
  min_samples_leaf: int = 2
  n_jobs: int = 1


@dataclass(frozen=True)
class TrainingConfig:
  min_rows: int
  test_size: float
  random_state: int
  threshold_start: float
  threshold_stop: float
  threshold_step: float
  logistic_regression: EstimatorConfig
  random_forest: EstimatorConfig
  gradient_boosting: EstimatorConfig


@dataclass(frozen=True)
class SandboxConfig:
  max_chars: int


@dataclass(frozen=True)
class ApiConfig:
  host: str
  port: int
  log_level: str
  cors_origins: tuple[str, ...]
  max_upload_bytes: int


@dataclass(frozen=True)
class TelemetryConfig:
  disable_otel: bool


@dataclass(frozen=True)
class ProjectConfig:
  provider: ProviderConfig
  session: SessionConfig
  runtime: RuntimeConfig
  experiment: ExperimentDefaults
  sample_data: SampleDataConfig
  training: TrainingConfig
  sandbox: SandboxConfig
  api: ApiConfig
  telemetry: TelemetryConfig

  @property
  def litellm_model(self) -> str:
    """Model string LiteLLM expects, built from the provider and model name."""
    name = self.provider.name.strip().lower()
    model = self.provider.model.strip()
    if name == "scripted" or model == "scripted":
      return "scripted"
    prefix = self.provider.litellm_prefix.strip()
    if model.startswith(f"{prefix}/") or model.startswith(f"{name}/"):
      return model
    if prefix:
      return f"{prefix}/{model}"
    return f"{name}/{model}"


def load_config(path: Path | None = None) -> ProjectConfig:
  """Read the YAML config. The file is cached for the process."""
  config_path = Path(path or os.environ.get("ML_CONFIG") or DEFAULT_CONFIG_PATH).expanduser()
  key = str(config_path.resolve())
  cached = _CACHE.get(key)
  if cached is not None:
    return cached
  if not config_path.is_file():
    raise FileNotFoundError(f"Config file not found: {config_path}")
  raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
  if not isinstance(raw, dict):
    raise ValueError(f"Config file must be a mapping: {config_path}")
  loaded = _parse(raw)
  _CACHE[key] = loaded
  apply_process_env(loaded)
  return loaded


def apply_process_env(config: ProjectConfig) -> None:
  """Publish provider and telemetry settings that libraries read from the environment."""
  if "OLLAMA_API_BASE" not in os.environ:
    os.environ["OLLAMA_API_BASE"] = config.provider.api_base
  if "OTEL_SDK_DISABLED" not in os.environ:
    os.environ["OTEL_SDK_DISABLED"] = "true" if config.telemetry.disable_otel else "false"


def detect_model_name() -> str:
  """Return the LiteLLM model id.

  ``ML_RUNTIME_MODEL`` replaces the config value for this process. Use
  ``scripted`` there to run the in-process tool model instead of Ollama.
  """
  override = os.environ.get("ML_RUNTIME_MODEL")
  if override:
    return override.strip()
  return load_config().litellm_model


def data_root() -> Path:
  """Directory that holds experiments, uploads, and long-term memory."""
  override = os.environ.get("ML_DATA_ROOT") or os.environ.get("ML_RUNTIME_ROOT")
  if override:
    return Path(override).expanduser()
  raw = Path(load_config().runtime.root).expanduser()
  if not raw.is_absolute():
    raw = PROJECT_ROOT / raw
  return raw


def config_path() -> Path:
  return Path(os.environ.get("ML_CONFIG") or DEFAULT_CONFIG_PATH).expanduser()


def _parse(raw: dict[str, Any]) -> ProjectConfig:
  provider = _section(raw, "provider")
  session = _section(raw, "session")
  runtime = _section(raw, "runtime")
  experiment = _section(raw, "experiment")
  sample = _section(raw, "sample_data")
  training = _section(raw, "training")
  sandbox = _section(raw, "sandbox")
  api = _section(raw, "api")
  telemetry = _section(raw, "telemetry")
  return ProjectConfig(
      provider=ProviderConfig(
          name=_text(provider, "name"),
          model=_text(provider, "model"),
          api_base=_text(provider, "api_base"),
          litellm_prefix=_text(provider, "litellm_prefix"),
          temperature=float(provider["temperature"]),
      ),
      session=SessionConfig(
          backend=_text(session, "backend"),
          app_name=_text(session, "app_name"),
          db_path=str(session.get("db_path") or "sessions.db"),
      ),
      runtime=RuntimeConfig(
          root=_text(runtime, "root"),
          identity=_text(runtime, "identity"),
          user=_text(runtime, "user"),
          max_improvement_loops=int(runtime["max_improvement_loops"]),
          approve_training=bool(runtime["approve_training"]),
          blocked_tools=tuple(_string_list(runtime, "blocked_tools")),
      ),
      experiment=ExperimentDefaults(
          objective=_text(experiment, "objective"),
          target_column=_text(experiment, "target_column"),
          metric=_text(experiment, "metric"),
          target_value=float(experiment["target_value"]),
      ),
      sample_data=SampleDataConfig(
          rows=int(sample["rows"]),
          seed=int(sample["seed"]),
          missing_rate=float(sample["missing_rate"]),
          missing_seed=int(sample["missing_seed"]),
          missing_columns=tuple(_string_list(sample, "missing_columns")),
      ),
      training=TrainingConfig(
          min_rows=int(training["min_rows"]),
          test_size=float(training["test_size"]),
          random_state=int(training["random_state"]),
          threshold_start=float(training["threshold_start"]),
          threshold_stop=float(training["threshold_stop"]),
          threshold_step=float(training["threshold_step"]),
          logistic_regression=_estimator(training, "logistic_regression"),
          random_forest=_estimator(training, "random_forest"),
          gradient_boosting=_estimator(training, "gradient_boosting"),
      ),
      sandbox=SandboxConfig(max_chars=int(sandbox["max_chars"])),
      api=ApiConfig(
          host=_text(api, "host"),
          port=int(api["port"]),
          log_level=_text(api, "log_level"),
          cors_origins=tuple(_string_list(api, "cors_origins")),
          max_upload_bytes=int(api["max_upload_bytes"]),
      ),
      telemetry=TelemetryConfig(disable_otel=bool(telemetry["disable_otel"])),
  )


def _estimator(training: dict[str, Any], name: str) -> EstimatorConfig:
  section = _section(training, name)
  weight = section.get("class_weight")
  return EstimatorConfig(
      max_iter=int(section.get("max_iter", 400)),
      class_weight=None if weight in (None, "", "null") else str(weight),
      n_estimators=int(section.get("n_estimators", 40)),
      max_depth=int(section.get("max_depth", 8)),
      min_samples_leaf=int(section.get("min_samples_leaf", 2)),
      n_jobs=int(section.get("n_jobs", 1)),
  )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
  value = raw.get(name)
  if not isinstance(value, dict):
    raise ValueError(f"Config section {name!r} must be a mapping.")
  return value


def _text(section: dict[str, Any], key: str) -> str:
  if key not in section or section[key] is None:
    raise ValueError(f"Config is missing {key!r}.")
  text = str(section[key]).strip()
  if not text:
    raise ValueError(f"Config value {key!r} is empty.")
  return text


def _string_list(section: dict[str, Any], key: str) -> list[str]:
  value = section.get(key, [])
  if value is None:
    return []
  if not isinstance(value, list):
    raise ValueError(f"Config value {key!r} must be a list.")
  return [str(item) for item in value]
