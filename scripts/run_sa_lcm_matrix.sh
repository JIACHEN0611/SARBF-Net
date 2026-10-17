#!/usr/bin/env bash
set -eo pipefail

TASKS=(
  ETTh1-96 ETTh1-192 ETTh1-336 ETTh1-720
  ETTh2-96 ETTh2-192 ETTh2-336 ETTh2-720
  ETTm1-96 ETTm1-192 ETTm1-336 ETTm1-720
  ETTm2-96 ETTm2-192 ETTm2-336 ETTm2-720
  ECL-96 ECL-192 ECL-336 ECL-720
  Exchange-96 Exchange-192 Exchange-336 Exchange-720
  Traffic-96 Traffic-192 Traffic-336 Traffic-720
  Weather-96 Weather-192 Weather-336 Weather-720
  Solar-96 Solar-192 Solar-336 Solar-720
  PEMS03-12 PEMS03-24 PEMS03-48 PEMS03-96
  PEMS04-12 PEMS04-24 PEMS04-48 PEMS04-96
  PEMS07-12 PEMS07-24 PEMS07-48 PEMS07-96
  PEMS08-12 PEMS08-24 PEMS08-48 PEMS08-96
)

if [ "${1:-}" = "--list-tasks" ]; then
  printf '%s\n' "${TASKS[@]}"
  exit 0
fi

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_root=${1:-"$repository_root/runs/sa_lcm_compact_final_v1"}

cd "$repository_root"
# Legacy runner. Activate your own Python environment before launching.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python -m tools.sa_lcm.matrix_profiles \
  --check-inputs --repository-root "$repository_root"

for task in "${TASKS[@]}"; do
  for seed in 2021 2022 2023; do
    result="$run_root/seed${seed}/${task}/sa_lcm/result.json"
    if [ -f "$result" ]; then
      status=$(python -c \
        'import json,sys; print(json.load(open(sys.argv[1])).get("status", ""))' \
        "$result")
      if [ "$status" = "COMPLETE" ]; then
        echo "[跳过] $task seed=$seed：已有完整 test 结果"
        continue
      fi
    fi

    echo "[开始] $task seed=$seed"
    python -u -m tools.sa_lcm_transport.run_pilot \
      --task "$task" --arm sa_lcm --seed "$seed" \
      --repository-root "$repository_root" --run-root "$run_root"
  done
done
