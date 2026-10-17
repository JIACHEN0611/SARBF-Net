#!/usr/bin/env bash
set -eo pipefail

TASKS=(
  ETTh1-96 ETTh1-192 ETTh1-336 ETTh1-720
  ETTh2-96 ETTh2-192 ETTh2-336 ETTh2-720
  ETTm1-96 ETTm1-192 ETTm1-336 ETTm1-720
  ETTm2-96 ETTm2-192 ETTm2-336 ETTm2-720
  Weather-96 Weather-192 Weather-336 Weather-720
  Solar-96 Solar-192 Solar-336 Solar-720
  ECL-96 ECL-192 ECL-336 ECL-720
  Traffic-96 Traffic-192 Traffic-336 Traffic-720
)

if [ "${1:-}" = "--list-tasks" ]; then
  printf '%s\n' "${TASKS[@]}"
  exit 0
fi

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_root=${1:-"$repository_root/runs/output_fold_training_v1"}

cd "$repository_root"
# Activate your own Python environment before launching.

for task in "${TASKS[@]}"; do
  for seed in 2021 2022 2023; do
    result="$run_root/seed${seed}/${task}/output_fold/result.json"
    if [ -f "$result" ]; then
      status=$(python - "$result" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle).get("status", ""))
PY
)
      if [ "$status" = "COMPLETE" ]; then
        echo "[跳过] $task seed=$seed：已有完整 test 结果"
        continue
      fi
      echo "[停止] 已有非 COMPLETE 记录，请检查：$result" >&2
      exit 1
    fi

    echo "[开始] $task seed=$seed"
    python -u -m tools.sa_lcm_transport.run_pilot \
      --task "$task" --arm output_fold --seed "$seed" \
      --repository-root "$repository_root" --run-root "$run_root"
  done
done
