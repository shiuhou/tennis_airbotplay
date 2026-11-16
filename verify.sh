#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
./run.sh --unit
run_dir=.agents/runs/foundation
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != --tool-config ]]; then
    echo 'Usage: ./verify.sh [--tool-config FILE]' >&2
    exit 2
  fi
  run_dir+=/racket
fi
mkdir -p "$run_dir"
for mode in verify contact demo; do
  ./run.sh --headless --device cpu --mode "$mode" "$@" \
    --report "$run_dir/$mode.json" --video "$run_dir/airbot_play.mp4"
done
if [[ $# -gt 0 ]]; then
  ./run.sh --headless --device cpu --mode contact "$@" --contact-body fixer \
    --report "$run_dir/contact_fixer.json"
fi
python3 tests/check_video.py --video "$run_dir/airbot_play.mp4" --report "$run_dir/demo.json"
