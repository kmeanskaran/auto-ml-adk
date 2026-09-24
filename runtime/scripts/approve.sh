#!/usr/bin/env bash
# Approve a paused training step: runtime/scripts/approve.sh <experiment-id>
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec "$PYTHON" -m runtime.cli approve "$@"
