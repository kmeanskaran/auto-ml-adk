#!/usr/bin/env bash
# Start the FastAPI service. Extra arguments are passed to `runtime.cli serve`.
# Example: runtime/scripts/serve.sh --port 8000
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec "$PYTHON" -m runtime.cli serve "$@"
