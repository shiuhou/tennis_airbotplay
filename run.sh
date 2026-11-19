#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
image=sha256:9c5b00fb775c65913f78f22eadb56912853c1f9f1c38828e53fb99369e29c49e
python_path="$project_root/src:$project_root/tests"
mkdir -p "$project_root/.agents/runs/foundation/tmp"
mount_args=(--rm -i --user "$(id -u):$(id -g)" -v "$project_root:$project_root" -w "$project_root"
  -e "PYTHONPATH=$python_path" -e PYTHONUNBUFFERED=1
  -e "TMPDIR=$project_root/.agents/runs/foundation/tmp")
if [[ "${1:-}" == --unit ]]; then
  exec bash scripts/verify_groups.sh "${2:-all}"
fi

cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/airbot-isaac"
mkdir -p "$cache_dir/home" "$cache_dir/kit" "$cache_dir/logs"
docker_args=(--gpus all --shm-size=2g -e PRIVACY_CONSENT=N
  -e HOME=/tmp/airbot-home -e NVIDIA_DRIVER_CAPABILITIES=all
  -v "$cache_dir/home:/tmp/airbot-home" -v "$cache_dir/kit:/isaac-sim/kit/cache"
  -v "$cache_dir/logs:/isaac-sim/kit/logs"
  --tmpfs "/isaac-sim/kit/data:uid=$(id -u),gid=$(id -g),mode=0700")
if [[ -t 0 && -t 1 ]]; then docker_args+=(-t); fi
if [[ -n "${DISPLAY:-}" ]]; then
  docker_args+=(-e "DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix:ro)
  if [[ -f "${XAUTHORITY:-}" ]]; then
    docker_args+=(-e XAUTHORITY=/tmp/airbot-xauth -v "$XAUTHORITY:/tmp/airbot-xauth:ro")
  fi
fi

# Kit may exit zero during startup. A run-specific completion token prevents
# accepting an incomplete launch or a report left by a previous invocation.
token="$(cat /proc/sys/kernel/random/uuid)"
process_exit=0
docker run "${mount_args[@]}" "${docker_args[@]}" -e "AIRBOT_RUN_TOKEN=$token" \
  --entrypoint /bin/bash "$image" /workspace/isaaclab/isaaclab.sh -p "$project_root/scripts/run_airbot.py" "$@" || process_exit=$?
for argument in "$@"; do
  case "$argument" in
    -h|--help) exit "$process_exit" ;;
  esac
done
python3 "$project_root/scripts/run_airbot.py" --check-completion "$token" "$process_exit" "$@"
