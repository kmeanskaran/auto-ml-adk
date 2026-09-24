#!/usr/bin/env bash
# Start the ADK orchestrator. Extra arguments are passed to `runtime.cli run`.
# Example: runtime/scripts/run_agents.sh --no-approve --target 0.75
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec "$PYTHON" -m runtime.cli run "$@"
