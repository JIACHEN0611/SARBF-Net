#!/usr/bin/env bash
set -euo pipefail
repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repository_root"
if [[ "${1:-}" == "--list-tasks" ]]; then
  python -m tools.sa_lcm.matrix_profiles --list-tasks
  exit 0
fi
run_root=${1:-"$repository_root/runs/sarbf_net"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Activate the user's own Python environment before calling this script.
python -m tools.sarbf_net.inputs --repository-root "$repository_root"
mapfile -t tasks < <(python -m tools.sa_lcm.matrix_profiles --list-tasks)
for task in "${tasks[@]}"; do
  for seed in 2021 2022 2023; do
    result="$run_root/seed${seed}/$task/sarbf_net/result.json"
    if [[ -f "$result" ]]; then
      status=$(python -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' "$result")
      if [[ "$status" == "COMPLETE" ]]; then
        echo "[skip COMPLETE] $task seed=$seed"
        continue
      fi
      echo "Existing incomplete result: $result. Choose a new run-root or move the failed task directory before retrying." >&2
      exit 1
    fi
    echo "[start] $task seed=$seed"
    python -u -m tools.sarbf_net.run --task "$task" --seed "$seed" --run-root "$run_root"
  done
done
python -m tools.sarbf_net.summarize --run-root "$run_root" --output "$run_root/summary.csv"
