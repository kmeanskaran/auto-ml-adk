#!/usr/bin/env bash
# Resume an interrupted experiment: runtime/scripts/resume.sh <experiment-id>
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec "$PYTHON" -m runtime.cli resume "$@"
