# SA-LCM Compact Final：服务器运行说明

## 1. 解压并建立数据链接

```bash
cd /home/ljc/pythonProject
unzip -q SA_LCM_FULL_MATRIX_52TASK_FINAL_V1.zip
cd SA_LCM_FULL_MATRIX_52TASK_FINAL_V1

source /home/ljc/miniconda3/etc/profile.d/conda.sh
conda activate olinear

test -e dataset || ln -s ../SA_LCM_OUTPUT_FOLD_RELEASE/dataset dataset
```

如果数据位于其他目录，请修改软链接目标，不要复制数据到发布包中。

## 2. 完整验收

```bash
python -m unittest discover -s tests -v
python -m tools.sa_lcm.matrix_profiles \
  --check-inputs --repository-root "$PWD"
```

服务器必须显示最终 `OK`。其中数值测试会验证：

- `SA_LCM` 与已经选定的 `pair_free_fold` 计算图逐元素一致；
- 构造前后的 CPU/CUDA RNG 状态一致；
- 旧 Pair-Free state dict 可以 `strict=True` 加载；
- 正式模型不存在 `pair_scorer` 参数；
- SA-LCM 前端和 BCIF 均能获得梯度。

## 3. 单任务正式运行

```bash
ROOT="$PWD"
RUNS="$PWD/runs/sa_lcm_compact_final_v1"

python -u -m tools.sa_lcm_transport.run_pilot \
  --task Solar-96 \
  --arm sa_lcm \
  --seed 2021 \
  --repository-root "$ROOT" \
  --run-root "$RUNS"
```

## 4. 四任务三随机种子串行运行

```bash
ROOT="$PWD"
RUNS="$PWD/runs/sa_lcm_compact_final_v1"
PY="/home/ljc/miniconda3/envs/olinear/bin/python"
mkdir -p "$RUNS"

nohup env ROOT="$ROOT" RUNS="$RUNS" PY="$PY" CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  bash -c '
set -euo pipefail
cd "$ROOT"
for seed in 2021 2022 2023; do
  for task in Solar-96 Weather-192 ETTh1-720 ECL-720; do
    result="$RUNS/seed${seed}/${task}/sa_lcm/result.json"
    if [ -f "$result" ] && [ "$("$PY" -c \
      "import json,sys; print(json.load(open(sys.argv[1])).get(\"status\", \"\"))" \
      "$result")" = COMPLETE ]; then
      echo "[跳过] $task seed=$seed"
      continue
    fi
    echo "[开始] $task seed=$seed"
    "$PY" -u -m tools.sa_lcm_transport.run_pilot \
      --task "$task" --arm sa_lcm --seed "$seed" \
      --repository-root "$ROOT" --run-root "$RUNS"
  done
done
echo "[完成] SA-LCM Compact Final"
' > "$RUNS/launcher_all.log" 2>&1 < /dev/null &

echo "启动 PID：$!"
tail -n 40 -f "$RUNS/launcher_all.log"
```

旧命令中的 `--arm pair_free_fold` 仍可使用，它指向同一个正式模型；新实验统一使用 `--arm sa_lcm`。

完整 52 任务、三随机种子实验请使用 `SERVER_RUN_MATRIX.md` 中的串行命令。
