#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
run_dir=.agents/runs/foundation/ball
mkdir -p "$run_dir"
rm -f "$run_dir/convergence.json"
./run.sh --unit
# Each frequency starts a fresh scene/process; do not mutate a running timeline.
for hz in 240 480 960; do
  ./run.sh --headless --device cpu --mode ball --ball-hz "$hz" --report "$run_dir/dt_$hz.json"
done
python3 tests/check_ball_convergence.py --directory "$run_dir"
