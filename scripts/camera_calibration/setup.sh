#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
local_dir="$root/.agents/runs/foundation/camera_calibration"
mkdir -p "$local_dir/tmp"
export TMPDIR="$local_dir/tmp"
python3 -m venv "$local_dir/venv"
"$local_dir/venv/bin/python" -m pip install -r "$root/scripts/camera_calibration/requirements.txt"
# The SDK declares opencv-python; installing it alongside contrib overwrites cv2.
# These tools use only the SDK extension, not its AV/pygame/pynput examples.
"$local_dir/venv/bin/python" -m pip install --no-deps pyorbbecsdk2==2.1.3
"$local_dir/venv/bin/python" -c 'import cv2, pyorbbecsdk; assert hasattr(cv2, "aruco"); print("Camera environment ready:", cv2.__version__)'
