"""FastAPI service in front of the agent orchestrator.

Run it from the repo root:

    runtime/scripts/serve.sh
    python -m runtime.cli serve --port 8000
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic import Field

from api.middleware import RequestContextMiddleware
from api.middleware import request_id_from
from api.settings import Settings
from api.settings import load_settings
from team.config import data_root
from team.config import detect_model_name
from team.config import load_config
from team.data_engineer.catalog import list_samples
from runtime.orchestrator import ExperimentError
from runtime.orchestrator import ExperimentRequest
from runtime.orchestrator import MlRuntime

logger = logging.getLogger("api")


_DEFAULTS = load_config()


_METRIC_NAMES = ("f1", "precision", "recall", "accuracy", "mae", "rmse", "r2")


def _metric_selection(primary: str, selected: list[str] | None) -> tuple[str, list[str]]:
  """Keep an explicit empty selection empty so data engineering can choose later."""
  names = [str(name or "").strip().lower() for name in (selected or [])]
  names.append(str(primary or "").strip().lower())
  if not any(names):
    return "", []
  picked: list[str] = []
  for key in names:
    if key in _METRIC_NAMES and key not in picked:
      picked.append(key)
  if not picked:
    picked = ["f1"]
  gate = str(primary or "").strip().lower()
  if gate not in picked:
    gate = "f1" if "f1" in picked else picked[0]
  return gate, picked


class CreateExperiment(BaseModel):
  objective: str = _DEFAULTS.experiment.objective
  target_column: str = _DEFAULTS.experiment.target_column
  metric: str = _DEFAULTS.experiment.metric
  metrics: list[str] = Field(default_factory=list)
  target_value: float = _DEFAULTS.experiment.target_value
  dataset_path: str | None = None
  approve_training: bool = True
  fault_tool: str | None = None
  user_id: str = "local"
  staged: bool = False
  checkpoints: list[str] = Field(default_factory=list)
  sample_id: str | None = None


class Approval(BaseModel):
  confirmed: bool = True
  allow_retries: bool = True


class Iteration(BaseModel):
  note: str
  target_value: float | None = None


class Rework(BaseModel):
  prompt: str


class ModelChoice(BaseModel):
  model: str


class PredictionRequest(BaseModel):
  rows: list[dict[str, Any]] = Field(min_length=1)


def create_app(
    root: Path | None = None,
    model: str | None = None,
    settings: Settings | None = None,
) -> FastAPI:
  """Build an app bound to one experiment directory and one model."""
  settings = settings or load_settings()
  changes: dict[str, Any] = {}
  if root is not None:
    changes["root"] = root
  if model is not None:
    changes["model"] = model
  if changes:
    settings = replace(settings, **changes)
  _configure_logging(settings.log_level)
  model_name = settings.model or detect_model_name()
  engine = MlRuntime(root=settings.root or data_root(), model_name=model_name)

  @asynccontextmanager
  async def lifespan(app: FastAPI):
    logger.info("api ready model=%s root=%s", engine.model_name, engine.root)
    yield
    logger.info("api stopped")

  app = FastAPI(
      title="Autonomous ML Engineer",
      version="0.1.0",
      summary="ADK runtime that designs, trains, and serves a classifier.",
      lifespan=lifespan,
  )
  app.state.runtime = engine
  app.state.settings = settings
  app.add_middleware(RequestContextMiddleware)
  origins = list(settings.cors_origins) or ["http://localhost:3000"]
  app.add_middleware(
      CORSMiddleware,
      allow_origins=origins,
      allow_methods=["*"],
      allow_headers=["*"],
  )

  @app.get("/health")
  def health() -> dict[str, str]:
    return {"status": "ok"}

  @app.get("/ready")
  def ready() -> dict[str, str]:
    directory = engine.root
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / ".ready"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    return {"status": "ready"}

  @app.get("/")
  def home() -> dict[str, Any]:
    return {
        "service": "autonomous-ml-engineer",
        "model": engine.model_name,
        "provider": load_config().provider.name,
        "identity": load_config().runtime.identity,
        "routes": {
            "health": "GET /health",
            "ready": "GET /ready",
            "create": "POST /experiments",
            "clear": "DELETE /experiments",
            "upload": "POST /experiments/upload",
            "status": "GET /experiments/{id}",
            "approve": "POST /experiments/{id}/approve",
            "resume": "POST /experiments/{id}/resume",
            "iterate": "POST /experiments/{id}/iterate",
            "design": "GET /experiments/{id}/design",
            "predict": "POST /experiments/{id}/predict",
            "board": "GET /experiments/{id}/board",
            "next": "POST /experiments/{id}/next",
            "rework": "POST /experiments/{id}/rework",
            "datasets": "GET /datasets",
            "choose_model": "POST /experiments/{id}/choose-model",
        },
    }

  @app.get("/datasets")
  def datasets() -> list[dict[str, Any]]:
    return list_samples()

  @app.get("/experiments")
  def list_experiments() -> list[dict[str, Any]]:
    return engine.list_experiments()

  @app.delete("/experiments")
  def clear_experiments() -> dict[str, int]:
    return engine.clear_experiments()

  @app.post("/experiments")
  async def create_experiment(body: CreateExperiment) -> dict[str, Any]:
    gate, selected = _metric_selection(body.metric, body.metrics)
    return await engine.start(
        ExperimentRequest(
            objective=body.objective,
            target_column=body.target_column,
            metric=gate,
            metrics=selected,
            target_value=body.target_value,
            dataset_path=body.dataset_path,
            approve_training=body.approve_training,
            fault_tool=body.fault_tool,
            user_id=body.user_id,
            staged=body.staged,
            checkpoints=body.checkpoints,
            sample_id=body.sample_id,
        )
    )

  @app.post("/experiments/upload")
  async def upload_experiment(
      file: UploadFile = File(...),
      objective: str = Form(_DEFAULTS.experiment.objective),
      target_column: str = Form(_DEFAULTS.experiment.target_column),
      metric: str = Form(_DEFAULTS.experiment.metric),
      metrics: str = Form(""),
      target_value: float = Form(_DEFAULTS.experiment.target_value),
      approve_training: bool = Form(True),
      staged: bool = Form(False),
      checkpoints: str = Form(""),
  ) -> dict[str, Any]:
    payload = await file.read()
    if len(payload) > settings.max_upload_bytes:
      raise HTTPException(status_code=413, detail="Upload exceeds the size limit.")
    safe_name = Path(file.filename or "dataset.csv").name or "dataset.csv"
    incoming = engine.root / "uploads"
    incoming.mkdir(parents=True, exist_ok=True)
    destination = incoming / f"{uuid.uuid4().hex[:8]}-{safe_name}"
    destination.write_bytes(payload)
    gate, selected = _metric_selection(
        metric, [item.strip() for item in metrics.split(",") if item.strip()]
    )
    return await engine.start(
        ExperimentRequest(
            objective=objective,
            target_column=target_column,
            metric=gate,
            metrics=selected,
            target_value=target_value,
            dataset_path=str(destination),
            approve_training=approve_training,
            staged=staged,
            checkpoints=[item.strip() for item in checkpoints.split(",") if item.strip()],
        )
    )

  @app.get("/experiments/{experiment_id}")
  def get_experiment(experiment_id: str) -> dict[str, Any]:
    return engine.get(experiment_id)

  @app.get("/experiments/{experiment_id}/board")
  def get_board(experiment_id: str) -> dict[str, Any]:
    return engine.board(experiment_id)

  @app.post("/experiments/{experiment_id}/next")
  async def next_stage(experiment_id: str) -> dict[str, Any]:
    return await engine.next_stage(experiment_id)

  @app.post("/experiments/{experiment_id}/rework")
  async def rework_stage(experiment_id: str, body: Rework) -> dict[str, Any]:
    return await engine.rework_stage(experiment_id, prompt=body.prompt)

  @app.post("/experiments/{experiment_id}/choose-model")
  def choose_model(experiment_id: str, body: ModelChoice) -> dict[str, Any]:
    return engine.choose_model(experiment_id, model=body.model)

  @app.post("/experiments/{experiment_id}/approve")
  async def approve_experiment(experiment_id: str, body: Approval) -> dict[str, Any]:
    return await engine.approve(
        experiment_id,
        confirmed=body.confirmed,
        allow_retries=body.allow_retries,
    )

  @app.post("/experiments/{experiment_id}/resume")
  async def resume_experiment(experiment_id: str) -> dict[str, Any]:
    return await engine.resume(experiment_id)

  @app.post("/experiments/{experiment_id}/iterate")
  async def iterate_experiment(experiment_id: str, body: Iteration) -> dict[str, Any]:
    return await engine.iterate(
        experiment_id,
        note=body.note,
        target_value=body.target_value,
    )

  @app.get("/experiments/{experiment_id}/design", response_class=PlainTextResponse)
  def get_design(experiment_id: str) -> str:
    return engine.design_text(experiment_id)

  @app.post("/experiments/{experiment_id}/predict")
  def predict(experiment_id: str, body: PredictionRequest) -> dict[str, Any]:
    return engine.predict(experiment_id, body.rows)

  @app.exception_handler(ExperimentError)
  async def handle_experiment_error(request, exc: ExperimentError) -> JSONResponse:
    return _error_response(request, exc.status_code, str(exc))

  @app.exception_handler(FileNotFoundError)
  async def handle_missing(request, exc: FileNotFoundError) -> JSONResponse:
    return _error_response(request, 404, str(exc))

  @app.exception_handler(HTTPException)
  async def handle_http(request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return _error_response(request, exc.status_code, detail)

  @app.exception_handler(Exception)
  async def handle_unexpected(request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error request_id=%s", request_id_from(request))
    return _error_response(request, 500, "Internal server error.")

  return app


def _error_response(request, status_code: int, message: str) -> JSONResponse:
  response = JSONResponse(
      status_code=status_code,
      content={"error": message, "request_id": request_id_from(request)},
  )
  request_id = request_id_from(request)
  if request_id:
    response.headers["X-Request-ID"] = request_id
  return response


def _configure_logging(level: str) -> None:
  resolved = getattr(logging, level.upper(), logging.INFO)
  api_logger = logging.getLogger("api")
  api_logger.setLevel(resolved)
  if not api_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    api_logger.addHandler(handler)
  api_logger.propagate = False


_APP: FastAPI | None = None


def get_app() -> FastAPI:
  """Create the process-wide app on first use."""
  global _APP
  if _APP is None:
    _APP = create_app()
  return _APP


def __getattr__(name: str) -> Any:
  if name == "app":
    return get_app()
  raise AttributeError(name)
