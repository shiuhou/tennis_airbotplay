#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="$root/.agents/runs/foundation/camera_calibration/venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo 'Run: bash scripts/camera_calibration/setup.sh' >&2
  exit 2
fi
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$root"
exec "$python_bin" "$root/scripts/camera_calibration/cli.py" "$@"
