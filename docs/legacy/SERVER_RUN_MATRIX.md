# SA-LCM Compact Final：完整任务矩阵

矩阵中的每个任务均使用正式 `sa_lcm` arm。模型结构与四任务三种子确认版一致。

固定实验协议：主干输入长度为 96，总连续上下文为 768（前 672 点用于历史
检索，最近 96 点进入预测主干）。ETTh1、ETTh2、ETTm1、ETTm2、ECL、
Exchange、Traffic、Weather、Solar 的预测长度为 96/192/336/720；PEMS03、
PEMS04、PEMS07、PEMS08 的预测长度为 12/24/48/96。共 52 个任务，三种子
共 156 次正式训练。

## Setup and tests

```bash
cd /home/ljc/pythonProject
unzip -q SA_LCM_FULL_MATRIX_52TASK_FINAL_V1.zip
cd SA_LCM_FULL_MATRIX_52TASK_FINAL_V1
source /home/ljc/miniconda3/etc/profile.d/conda.sh
conda activate olinear
test -e dataset || ln -s ../SA_LCM_OUTPUT_FOLD_RELEASE/dataset dataset
python -m unittest discover -s tests -v
python -m tools.sa_lcm.matrix_profiles \
  --check-inputs --repository-root "$PWD"
```

All tests must pass. PyTorch numerical tests must not be skipped in the server
environment. 输入预检也必须通过；它会一次性列出缺少的原始数据或 Q 矩阵。

## Three-seed full test runs

每个任务独立训练，通过 validation 选择 checkpoint，最后只评估一次 test。

```bash
RUNS="$PWD/runs/sa_lcm_compact_final_v1"
LOGS="$PWD/runs/matrix_launcher_v1"
mkdir -p "$LOGS"

nohup bash scripts/run_sa_lcm_matrix.sh "$RUNS" \
  > "$LOGS/launcher_matrix_three_seeds.log" 2>&1 < /dev/null &

echo "PID: $!"
tail -n 40 -f "$LOGS/launcher_matrix_three_seeds.log"
```

The process is serial on one visible GPU. CPU/BLAS and DataLoader settings are
left at the established training-protocol defaults. Existing `COMPLETE`
records are skipped. Disconnecting SSH does not stop it because it is run
under `nohup`.
