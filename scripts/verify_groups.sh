#!/usr/bin/env bash
# Environment-aware dispatcher. These are unit checks, not the Foundation simulation.
set -uo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
group="${1:-all}"
case "$group" in
  all) groups=(research core pace camera);;
  research|core|pace|frozen|mean|camera) groups=("$group");;
  *) echo 'Usage: bash scripts/verify_groups.sh [all|research|core|pace|frozen|mean|camera]' >&2; exit 2;;
esac
image=sha256:9c5b00fb775c65913f78f22eadb56912853c1f9f1c38828e53fb99369e29c49e
mkdir -p .agents/runs/foundation/verification/tmp
status=0
for group in "${groups[@]}"; do
  report=".agents/runs/foundation/verification/$group.json"
  rm -f "$report"
  case "$group" in
    research) command=(python3 scripts/verify.py "$group" --report "$report");;
    camera) command=(.agents/runs/foundation/camera_calibration/venv/bin/python scripts/verify.py "$group" --report "$report");;
    *) command=(docker run --rm -i --user "$(id -u):$(id -g)" -v "$root:$root" -w "$root"
       -e "TMPDIR=$root/.agents/runs/foundation/verification/tmp"
       --entrypoint /isaac-sim/python.sh "$image" scripts/verify.py "$group" --report "$report");;
  esac
  "${command[@]}"
  code=$?
  if [[ ! -f "$report" ]]; then
    echo "$group: BLOCKED (environment/launcher failed, exit $code)" >&2
    code=2
  fi
  if [[ $code == 1 ]]; then status=1
  elif [[ $code != 0 && $status == 0 ]]; then status=2; fi
done
exit "$status"
