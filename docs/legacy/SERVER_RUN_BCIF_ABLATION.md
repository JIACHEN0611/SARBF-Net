# BCIF ablation: validation-first server procedure

The two arms isolate one mechanism at a time. `no_mass_bcif` removes the
external SA-LCM confidence only inside BCIF. `unbalanced_bcif` removes only the
balanced self-cancellation term. Neither arm changes SA-LCM retrieval, the
output-fold head, data/Q inputs, or training hyperparameters.

## 1. Extract, link data, and run all tests

```bash
cd /home/ljc/pythonProject/SA_LCM_BCIF_ABLATION_RELEASE
source /home/ljc/miniconda3/etc/profile.d/conda.sh
conda activate olinear

test -e dataset || ln -s ../SA_LCM_OUTPUT_FOLD_RELEASE/dataset dataset
python -m unittest discover -s tests -v || exit 1
```

The server suite must run the PyTorch numerical tests in
`tests/test_bcif_ablation.py`; they must not be skipped in the `olinear`
environment.

## 2. First screen: two arms, four tasks, seed 2021, validation only

```bash
RUNS="$PWD/runs/bcif_ablation_validation_v1"
mkdir -p "$RUNS"

nohup bash -c '
set -e
for arm in no_mass_bcif unbalanced_bcif; do
  for task in Solar-96 Weather-192 ETTh1-720 ECL-720; do
    echo "[开始] arm=$arm task=$task seed=2021"
    python -u -m tools.sa_lcm_transport.run_pilot \
      --task "$task" --arm "$arm" --seed 2021 --validation-only \
      --repository-root "$PWD" --run-root "$PWD/runs/bcif_ablation_validation_v1"
  done
done
' > "$RUNS/launcher_validation.log" 2>&1 < /dev/null &

echo "启动 PID：$!"
tail -n 40 -f "$RUNS/launcher_validation.log"
```

This launches eight fresh training runs. It does not train `output_fold` and
does not open the held-out test split.

## 3. Show validation results

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('runs/bcif_ablation_validation_v1/seed2021')
for task in ('Solar-96', 'Weather-192', 'ETTh1-720', 'ECL-720'):
    print(f'\n{task}')
    for arm in ('no_mass_bcif', 'unbalanced_bcif'):
        path = root / task / arm / 'validation/result.json'
        if not path.is_file():
            print(f'  {arm:18} 尚无结果')
            continue
        record = json.loads(path.read_text())
        value = record.get('validation') or {}
        print(f"  {arm:18} {record.get('status','—'):10} "
              f"MSE={value.get('mse','—')} MAE={value.get('mae','—')}")
PY
```

Compare these with the existing `output_fold` validation records only after
checking that seed, raw-data SHA256, Q SHA256, target manifest, and training
profile match. Do not select an arm from held-out test performance.

## 4. Confirmation only after validation selection

Replace `CHOSEN_ARM` with the selected ablation. If neither ablation improves
or preserves the validation result across representative tasks, keep full
BCIF and do not run this stage.

```bash
CHOSEN_ARM=no_mass_bcif
RUNS="$PWD/runs/bcif_ablation_final_v1"
mkdir -p "$RUNS"

nohup bash -c '
set -e
arm='"$CHOSEN_ARM"'
for seed in 2021 2022 2023; do
  for task in Solar-96 Weather-192 ETTh1-720 ECL-720; do
    echo "[开始] arm=$arm task=$task seed=$seed"
    python -u -m tools.sa_lcm_transport.run_pilot \
      --task "$task" --arm "$arm" --seed "$seed" \
      --repository-root "$PWD" --run-root "$PWD/runs/bcif_ablation_final_v1"
  done
done
' > "$RUNS/launcher_final.log" 2>&1 < /dev/null &

echo "启动 PID：$!"
tail -n 40 -f "$RUNS/launcher_final.log"
```

The final stage evaluates test MSE/MAE only for the validation-selected arm.
