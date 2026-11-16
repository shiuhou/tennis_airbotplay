#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
./run.sh --unit
for mode in verify contact demo; do
  ./run.sh --headless --device cpu --mode "$mode"
done
python3 tests/check_video.py
