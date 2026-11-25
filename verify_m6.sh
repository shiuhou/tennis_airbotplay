#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
data_dir="${3:-.agents/runs/foundation/m6}"
output="${2:?Usage: ./verify_m6.sh --reuse-data NEW_OUTPUT_DIRECTORY (or --collect NEW_OUTPUT_DIRECTORY)}"
case "${1:-}" in
  --reuse-data|--collect) ;;
  *) echo 'Choose --reuse-data or --collect explicitly' >&2; exit 2;;
esac
if [[ -e "$output" ]]; then echo "Output already exists: $output" >&2; exit 2; fi
python3 scripts/prepare_pace.py
mkdir -p .agents/runs/foundation/m6/tmp
if [[ "${1:-}" != --reuse-data ]]; then
  ./run.sh --mode pace-data --pace-data-dir "$data_dir" --headless --report "$data_dir/collection.json"
fi
image=sha256:9c5b00fb775c65913f78f22eadb56912853c1f9f1c38828e53fb99369e29c49e
args=(--rm -i --user "$(id -u):$(id -g)" -v "$project_root:$project_root" -w "$project_root"
  -e "PYTHONPATH=$project_root/src:$project_root/tests"
  -e "TMPDIR=$project_root/.agents/runs/foundation/m6/tmp"
  --entrypoint /isaac-sim/python.sh "$image")
docker run "${args[@]}" -m unittest discover -s tests -p test_pace_m6.py -v
# Explicit new output directory protects the original M6 result.
docker run "${args[@]}" scripts/verify_m6.py --data-dir "$data_dir" --output-dir "$output"
