#!/usr/bin/env bash
# Run one stage under the pinned Isaac image, and record its actual process exit.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# != 2 || ( "$1" != collect && "$1" != fit && "$1" != probe && "$1" != remaining ) ]]; then
  echo "Usage: bash scripts/run_recovery_bc.sh {collect|fit|probe|remaining} OUTPUT_DIRECTORY" >&2
  exit 2
fi
stage=$1
output=$2
original=.agents/runs/foundation/m7/deepmind_five/bc
entry=tests/pace_recovery.py
report="$output/recovery.json"
if [[ $stage == fit ]]; then report="$output/fit/results.json"; fi
if [[ $stage == probe || $stage == remaining ]]; then
  report="$output/probe.json"
  entry=tests/pace_recovery_probe.py
fi
mkdir -p "$(dirname "$output")"
exec 9>"$(dirname "$output")/.$(basename "$output").lock"
flock -n 9 || { echo "This study is already running" >&2; exit 1; }
if [[ -e $report || ( $stage != fit && -e $output ) ]]; then
  echo "This experiment already has evidence at $output" >&2
  exit 2
fi
root=$PWD
cache="${XDG_CACHE_HOME:-$HOME/.cache}/airbot-isaac"
mkdir -p "$cache/home" "$cache/kit" "$cache/logs" .agents/runs/foundation/tmp
image=sha256:9c5b00fb775c65913f78f22eadb56912853c1f9f1c38828e53fb99369e29c49e
log=$(mktemp "$(dirname "$output")/recovery_${stage}_XXXX.log")
echo "Stage $stage log: $log"
docker run --rm --gpus all --shm-size=2g --user "$(id -u):$(id -g)" -v "$root:$root" -w "$root" \
  -e "PYTHONPATH=$root/src:$root/tests:$root/scripts" -e PYTHONUNBUFFERED=1 \
  -e "TMPDIR=$root/.agents/runs/foundation/tmp" -e PRIVACY_CONSENT=N -e HOME=/tmp/airbot-home \
  -e NVIDIA_DRIVER_CAPABILITIES=all -v "$cache/home:/tmp/airbot-home" -v "$cache/kit:/isaac-sim/kit/cache" \
  -v "$cache/logs:/isaac-sim/kit/logs" --tmpfs "/isaac-sim/kit/data:uid=$(id -u),gid=$(id -g),mode=0700" \
  --entrypoint /bin/bash "$image" /workspace/isaaclab/isaaclab.sh -p "$entry" \
  --original-bc "$original" --output "$output" --stage "$stage" --device cuda:0 --headless >"$log" 2>&1
rg -q "^RECOVERY_STAGE_COMPLETE $stage PASS$" "$log"
python3 - "$report" <<'PY'
import json
from pathlib import Path
import sys
path = Path(sys.argv[1])
result = json.loads(path.read_text())
assert result['status'] == 'PASS'
result['process_exit_code'] = 0
updated = path.with_suffix('.tmp')
updated.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
updated.replace(path)
PY
mv "$log" "$output/$stage.log"
echo "Completed $stage: $report"
