#!/usr/bin/env bash
# Retrain from a note: runtime/scripts/iterate.sh <experiment-id> --note "..."
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec "$PYTHON" -m runtime.cli iterate "$@"
