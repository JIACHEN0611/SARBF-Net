# Server procedure: folded output arm against verified historical checkpoints

Run from the extracted `SA_LCM_OUTPUT_FOLD_RELEASE` directory with PyTorch,
the original forecasting data and selected old BCIF checkpoints available.
This procedure trains only the new `output_fold` arm on four tasks at seed
2021. Use the existing `no_diag` records for historical comparison; rerun
`no_diag` only if a genuinely matched new seed or source comparison is needed.

## 1. Environment and source checks

```bash
cd /home/ljc/pythonProject/SA_LCM_OUTPUT_FOLD_RELEASE
source /home/ljc/miniconda3/etc/profile.d/conda.sh
conda activate olinear
# In a fresh environment: python -m pip install -r requirements.txt
if [ ! -d dataset ]; then
  if [ -d ../SA_LCM_PRUNE_RELEASE/dataset ]; then
    ln -s ../SA_LCM_PRUNE_RELEASE/dataset dataset
  elif [ -d ../SA_LCM_NET_RELEASE/dataset ]; then
    ln -s ../SA_LCM_NET_RELEASE/dataset dataset
  else
    echo 'Missing dataset: set a working dataset symlink first' >&2
    exit 1
  fi
fi
python -m unittest discover -s tests -v || exit 1
python -c 'import torch, patoolib, tqdm' || exit 1

PRUNE_SOURCE=/home/ljc/pythonProject/SA_LCM_PRUNE_RELEASE
PRUNE_RUNS="$PRUNE_SOURCE/runs/sa_lcm_prune_v1"
SAFE_SOURCE=/home/ljc/pythonProject/SA_LCM_SAFE_PRUNE_RELEASE
SAFE_RUNS="$SAFE_SOURCE/runs/safe_prune_missing_bcif_v1"
# Optional: export OLD_NODIAG_RUNS=/actual/path/to/existing/no_diag/runs
REPORTS="$PWD/runs/output_fold_replay_v1"
RUNS="$PWD/runs/output_fold_training_v1"
```

Set `OLD_NODIAG_RUNS` only if you have prior validation-only `no_diag` records.
The replay itself always provides a same-checkpoint no_diag comparison.
The reference source root must be the unchanged full directory that
produced each original record; its old `SOURCE_FILES` tuple determines its
source hash. Solar/Weather use old PRUNE; ETTh1/ECL use old SAFE_PRUNE.
Existing result JSON, checkpoint, source, and Q files are read-only.

## 2. Replay verified old BCIF checkpoints on the full validation split

```bash
for task in Solar-96 Weather-192 ETTh1-720 ECL-720; do
  case "$task" in
    Solar-96|Weather-192) old_source="$PRUNE_SOURCE"; old_runs="$PRUNE_RUNS" ;;
    ETTh1-720|ECL-720) old_source="$SAFE_SOURCE"; old_runs="$SAFE_RUNS" ;;
  esac
  old="$old_runs/seed2021/$task/bcif/validation/result.json"
  test -f "$old" || { echo "Missing validated BCIF record: $old" >&2; exit 1; }
  python -u -m tools.sa_lcm_safe_prune.replay \
    --old-result "$old" --reference-source-root "$old_source" \
    --output "$REPORTS/seed2021/$task/comparison.json" || exit 1
done
```

If an old checkpoint moved, add `--checkpoint /actual/path/checkpoint.pth` for
that task. The replay validates the old validation-only record and verifies
source, selected checkpoint, entire CSV, both Q files, train/valid target
manifest, and seed. It constructs **only valid**, uses one ordered loader for
all three arms, loads BCIF and no_diag old weights strictly, and strictly
converts the same old weights for output_fold. The report includes each arm's
full-valid MSE/MAE, parameter count, maximum absolute prediction difference,
RMS difference, worst target, and folded parameter savings. The replay checks
the exact reduction formula `E×H×(D+F+1)−F×(D+H)` (1,263,104 for H=96 and
11,177,216 for H=720 when E=16 and D=F=512). The old BCIF metrics must
reproduce within 0.02%, and each graph-preserving comparison must pass max
absolute `0.001` and relative MSE/MAE `0.0002`. A failed report stops the
screen.

## 3. Train only output_fold on validation, four tasks, seed 2021

```bash
for task in Solar-96 Weather-192 ETTh1-720 ECL-720; do
  python -u -m tools.sa_lcm_transport.run_pilot \
    --task "$task" --arm output_fold --seed 2021 --validation-only \
    --repository-root "$PWD" --run-root "$RUNS" || exit 1
done

python - <<'PY'
import json
import os
from pathlib import Path

reports = Path('runs/output_fold_replay_v1/seed2021')
fresh = Path('runs/output_fold_training_v1/seed2021')
old_no_diag = os.environ.get('OLD_NODIAG_RUNS')
for task in ('Solar-96', 'Weather-192', 'ETTh1-720', 'ECL-720'):
    replay = json.loads((reports/task/'comparison.json').read_text())
    new = json.loads((fresh/task/'output_fold/validation/result.json').read_text())
    assert replay['status'] == 'VALIDATED_EQUIVALENT', (task, replay['status'])
    assert replay['test'] is None and replay['evaluation_scope'] == 'validation_only', task
    assert new['status'] == 'VALIDATED' and new['test'] is None, task
    assert new['architecture_revision'] == 'SA_LCM_OUTPUT_FOLD_V1', task
    old = replay['recorded_bcif_validation']
    candidate = new['validation']
    same_weights = replay['validation']['no_diag']
    print(f"{task:13} old BCIF MSE={old['mse']:.9f} MAE={old['mae']:.9f}"
          f" | replay no_diag MSE={same_weights['mse']:.9f} MAE={same_weights['mae']:.9f}"
          f" | fresh fold MSE={candidate['mse']:.9f} MAE={candidate['mae']:.9f}")
    if old_no_diag:
        path = Path(old_no_diag)/'seed2021'/task/'no_diag/validation/result.json'
        if not path.is_file():
            print(f"  prior no_diag record not found: {path}")
            continue
        baseline = json.loads(path.read_text())
        same_inputs = (baseline.get('status') == 'VALIDATED'
                       and baseline.get('seed') == 2021
                       and baseline.get('arm') == 'no_diag'
                       and baseline.get('evaluation_scope') == 'validation_only'
                       and baseline.get('test') is None
                       and baseline.get('data_sha256') == new.get('data_sha256')
                       and baseline.get('q_identity', {}).get('sha256')
                           == new.get('q_identity', {}).get('sha256')
                       and baseline.get('sample_manifest') == new.get('sample_manifest'))
        if same_inputs:
            prior = baseline['validation']
            print(f"  prior fresh no_diag MSE={prior['mse']:.9f}"
                  f" MAE={prior['mae']:.9f} (matched inputs and targets)")
        else:
            print(f"  prior no_diag provenance does not match: {path}")
PY
```

The selection screen never constructs held-out test targets or computes test
metrics. The upstream training “Test Loss” during validation-only training is
redirected to validation by the blind-run guard. The raw CSV SHA256 includes
the whole file; that hash is a provenance check, not test evaluation. Fresh
retraining can vary from checkpoint replay, so assess all four new validation
metrics before selecting an architecture. For a matched comparison if prior
`no_diag` records have a different source/seed, run a separate validation-only
`--arm no_diag` with the same task, source and seed; this optional run is not
part of the default four commands.

## 4. Only after validation selection, run actual held-out test

The full command independently trains the **chosen** arm again, selects its
checkpoint by validation, and then evaluates both full validation and actual
held-out test. It writes beside earlier `validation/` records without
overwriting them. If output_fold is selected:

```bash
for task in Solar-96 Weather-192 ETTh1-720 ECL-720; do
  python -u -m tools.sa_lcm_transport.run_pilot \
    --task "$task" --arm output_fold --seed 2021 \
    --repository-root "$PWD" --run-root "$RUNS" || exit 1
done

python - <<'PY'
import json
from pathlib import Path

for task in ('Solar-96', 'Weather-192', 'ETTh1-720', 'ECL-720'):
    path = Path('runs/output_fold_training_v1/seed2021')/task/'output_fold/result.json'
    record = json.loads(path.read_text())
    assert record['status'] == 'COMPLETE', (task, record.get('artifact_error'))
    assert record['evaluation_scope'] == 'complete', task
    print(f"{task:13} actual test MSE={record['test']['mse']:.9f}"
          f" MAE={record['test']['mae']:.9f}")
PY
```

Keep checkpoints, JSON records, reports, and datasets outside the ZIP.
