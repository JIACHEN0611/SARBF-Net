# SAER transfer experiments

We compare DLinear, PatchTST, iTransformer, CycleNet and TSMixer with and without SAER. Both groups receive the same 768-point data window. The original group uses the last 96 points; the SAER group refines those 96 points with information from the preceding 672. BCIF is not used. Backbone normalization and the archived training recipes are preserved.

## Data

Download the benchmark files from [DATA_SOURCES.md](DATA_SOURCES.md). The transfer launcher expects a flat data directory, separate from the main-model layout:

| CLI dataset | File under `--data-root` |
| --- | --- |
| ETTm1 | `ETTm1.csv` |
| Electricity | `electricity.csv` |
| PEMS03 | `PEMS03.npz` |
| PEMS07 | `PEMS07.npz` |
| Solar | `solar_AL.txt` |
| Weather | `weather.csv` |

The default directory is `dataset_pilot`. These backbones do not need the main model's external Q arrays.

## Single task and paired comparison

Run from the source root. Paths with spaces must be quoted. `--dry-run` prints the command without training or creating a run directory:

```bash
python -m tools.saer_transfer.run --backbone DLinear --dataset ETTm1 --horizon 96 --seed 2021 --arm sa_frontend --data-root dataset_pilot --dry-run
python -m tools.saer_transfer.run --backbone DLinear --dataset ETTm1 --horizon 96 --seed 2021 --arm original --data-root dataset_pilot --run-root runs/saer_transfer
python -m tools.saer_transfer.run --backbone DLinear --dataset ETTm1 --horizon 96 --seed 2021 --arm sa_frontend --data-root dataset_pilot --run-root runs/saer_transfer
```

On Linux, an external `CUDA_VISIBLE_DEVICES` selection is retained. `python -m experiments.saer_transfer.train --help` lists lower-level training flags, including `--use_gpu false` for CPU execution. The fixed public matrix uses 96 recent points and 768 total observations.

## Matrix and results

```bash
python -m tools.saer_transfer.run --list-tasks
bash scripts/run_saer_transfer_matrix.sh --data-root dataset_pilot --run-root runs/saer_transfer
python -m tools.saer_transfer.summarize --run-root runs/saer_transfer --output transfer_detail.csv --paired-output transfer_paired.csv --aggregate-output transfer_summary.csv
```

The matrix contains 5 backbones × 6 datasets × 4 horizons (96/192/336/720) × 3 seeds (2021/2022/2023) × 2 groups = **720 presets**. Omitting a filter selects all its values. The launcher runs tasks serially, skips completed tasks and refuses to overwrite an incomplete nonempty task directory. Use a new run root or preserve and relocate an incomplete directory before retrying. `--keep-going` continues after a training process fails.

Each task saves `task.json`, `stdout.log`, selected checkpoints and `metrics.json`. Training uses train and validation splits for early stopping; the selected checkpoint is then tested. Summary tables reject duplicate task/seed/group records and exclude incomplete or failed records. Paired aggregates use only seeds with both groups; standard deviations use ddof=1 and remain blank for one seed. A zero baseline leaves its undefined percentage improvement blank; raw metrics and direct win counts remain available.

These are runnable presets. The v1.0.0 asset release covers main-model studies rather than this entire transfer matrix. No transfer accuracy is inferred from smoke checks. The integrated code follows the archived five-model package; later server `protocol_v2` corrections require separate comparison before claiming the same final experimental protocol.
