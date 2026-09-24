# Shared path setup for the agent scripts. Source this; do not execute it.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHON="${PYTHON:-python}"
cd "$ROOT"
