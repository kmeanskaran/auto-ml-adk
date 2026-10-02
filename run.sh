#!/usr/bin/env bash
# Start the ML Team: the backend (agents, API, /predict) and the console (Next.js),
# each in its own container, connected by docker-compose.yml.
#
#   ./run.sh              build and start both, wait until healthy, follow the agents' log
#   ./run.sh logs         follow the logs again (Ctrl+C stops following, not the team)
#   ./run.sh status       what is running
#   ./run.sh down         stop both
#   ./run.sh local        no Docker: backend with uv and console with npm, on this machine
#
#   BACKEND_PORT=9000 FRONTEND_PORT=4000 ./run.sh   other ports
#   ML_MODEL=ollama_chat/gpt-oss:120b-cloud ./run.sh   another LLM for the agents (Ollama)
set -euo pipefail
cd "$(dirname "$0")"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
# The model: config/config.yml (models.team), unless ML_MODEL is set in the shell or .env.
[ -z "${ML_MODEL:-}" ] && [ -f .env ] && ML_MODEL="$(sed -n 's/^ML_MODEL=//p' .env | tail -n 1)"
ML_MODEL="${ML_MODEL:-}"   # empty: the backend reads config/config.yml
MODEL_SHOWN="${ML_MODEL:-$(sed -n 's/^  team: *\([^ #]*\).*/\1/p' config/config.yml)}"
OLLAMA_HOST_URL="${OLLAMA_HOST_URL:-http://localhost:11434}"   # Ollama as seen from this machine
export BACKEND_PORT FRONTEND_PORT ML_MODEL

fail() { echo "✗ $*" >&2; exit 1; }
warn() { echo "! $*" >&2; }
step() { echo "● $*"; }

checks() {
  # The team learns from the dataset; the production records feed the serving checks.
  local dataset traffic
  dataset="$(sed -n 's/^dataset: *\([^ #]*\).*/\1/p' config/config.yml)"
  traffic="$(sed -n 's/^traffic: *\([^ #]*\).*/\1/p' config/config.yml)"
  [ -f "data/$dataset" ] || fail "No data/$dataset. Put the loan train.csv and test.csv in data/lending-loan/
  (or point dataset and traffic in config/config.yml at your files)."
  [ -f "data/$traffic" ] || warn "No data/$traffic: stage checks and /predict tests will fail."
  [ -f .env ] || { cp .env.example .env; step "Created .env from .env.example"; }
  # Bind-mounted folders: created here so they belong to you, not to the container.
  mkdir -p runs registry feature_store .adk
  # The agents need their model; the console and /predict work without it.
  if [[ "$MODEL_SHOWN" == gemini* ]]; then
    if grep -qE '^GOOGLE_GENAI_USE_VERTEXAI=(true|1|TRUE|True)' .env; then
      grep -qE '^GOOGLE_CLOUD_PROJECT=.+' .env || warn "Vertex AI needs GOOGLE_CLOUD_PROJECT in .env."
      [ -f "$HOME/.config/gcloud/application_default_credentials.json" ] \
        || warn "No gcloud credentials for Vertex AI. Run: gcloud auth application-default login"
    else
      grep -qE '^(GEMINI_API_KEY|GOOGLE_API_KEY)=.+' .env \
        || warn "No GEMINI_API_KEY in .env (nor Vertex AI settings): the pipeline and the analyst will fail until one is set."
    fi
    if grep -qE '^GOOGLE_GENAI_USE_VERTEXAI=(true|1|TRUE|True)' .env \
      && [ ! -f "$HOME/.config/gcloud/application_default_credentials.json" ]; then
      warn "Vertex AI is on but there are no Google Cloud credentials. Run: gcloud auth application-default login"
    fi
  elif [[ "$MODEL_SHOWN" == ollama* ]]; then
    local model="${MODEL_SHOWN#*/}" tags
    if ! tags="$(curl -sf --max-time 3 "$OLLAMA_HOST_URL/api/tags")"; then
      warn "Ollama is not reachable at $OLLAMA_HOST_URL: the pipeline and the analyst will fail until it is."
    elif [[ "$tags" != *"\"$model\""* ]]; then
      warn "Ollama has no model $model. Pull it with: ollama pull $model"
    fi
  fi
}

port_free() {
  ! { command -v lsof >/dev/null && lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }
}

urls() {
  echo
  echo "  Console   http://localhost:$FRONTEND_PORT"
  echo "  Backend   http://localhost:$BACKEND_PORT   (POST /predict, ADK dev UI at /dev-ui)"
  echo "  Agents    $MODEL_SHOWN (analyst: $(sed -n 's/^  analyst: *\([^ #]*\).*/\1/p' config/config.yml))"
  echo "  Traces    runs/<run>/logs/"
  echo
}

docker_ready() {
  command -v docker >/dev/null || fail "Docker is not installed: https://docs.docker.com/get-docker/"
  docker info >/dev/null 2>&1 || fail "Docker is not running. Start Docker Desktop, or use: ./run.sh local"
}

up() {
  docker_ready
  checks
  step "Building and starting backend and frontend…"
  docker compose up --build --detach --wait --wait-timeout 300 \
    || { docker compose logs --tail 40 backend >&2; fail "The containers did not become healthy (logs above)."; }
  urls
  step "Following the agents' log. Ctrl+C stops following; the team keeps running (./run.sh down stops it)."
  docker compose logs --follow --no-log-prefix --since 0s backend
}

local_run() {
  command -v uv >/dev/null || fail "uv is not installed: https://docs.astral.sh/uv/"
  command -v npm >/dev/null || fail "npm is not installed: https://nodejs.org/"
  checks
  port_free "$BACKEND_PORT" || fail "Port $BACKEND_PORT is in use. Try: BACKEND_PORT=<port> ./run.sh local"
  port_free "$FRONTEND_PORT" || fail "Port $FRONTEND_PORT is in use. Try: FRONTEND_PORT=<port> ./run.sh local"
  if [ ! -d .venv ] || [ uv.lock -nt .venv ]; then step "Installing backend dependencies…"; uv sync; touch .venv; fi
  if [ ! -d frontend/node_modules ] || [ frontend/package-lock.json -nt frontend/node_modules ]; then
    step "Installing console dependencies…"; (cd frontend && npm ci --no-audit --no-fund); fi
  export OLLAMA_API_BASE="${OLLAMA_API_BASE:-$OLLAMA_HOST_URL}"
  trap 'kill 0' EXIT INT TERM   # Ctrl+C stops both
  uv run uvicorn app.fast_api_app:app --host 127.0.0.1 --port "$BACKEND_PORT" &
  (cd frontend && BACKEND_URL="http://localhost:$BACKEND_PORT" npm run dev -- --port "$FRONTEND_PORT") &
  urls
  wait
}

case "${1:-up}" in
  up) up ;;
  logs) docker compose logs --follow --no-log-prefix backend ;;
  status) docker compose ps ;;
  down) docker compose down ;;
  local) local_run ;;
  *) fail "Unknown command ${1}. Use: up, logs, status, down or local." ;;
esac
