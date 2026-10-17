# SARBF-Net

**Selective Anchored Retrieval & Balanced Feedback Network** for multivariate
time series forecasting. [中文说明](README_zh-CN.md)

This repository packages the supplied final model under its current paper name.
It preserves the selected model computation, initialization order, training
profiles and historical checkpoint keys.

- **SAER** aligns historical observations to recent statistics, retrieves
  relevant historical evidence, and refines the recent input with a bounded correction.
- **BCIF** uses evidence support to condition feedback between intra-variable
  updates and cross-variable interaction.
- A folded affine output head removes a redundant expanded projection.

## Release status

The preparation archive includes model code, loaders, a locked 52-task profile matrix,
legacy ablation/replay utilities, tests and public experiment tools. It does
**not** contain datasets, experiment Q arrays, trained weights or paper result
files. A `With_Assets` archive created by the collector additionally contains
verified Q/checkpoint/result files listed in `artifacts/manifest.json`.
A public code repository can be created from this directory now; exact
paper reproduction still requires the original Q assets and matching data.
No paper metrics are claimed by the synthetic checks.

The inherited matrix has **13 datasets × 4 horizons = 52 configurations**, with
three recommended seeds (156 runs). It includes Exchange even if a later paper
version omits that dataset. The model consumes **768 total observations**:
672 historical points plus a 96-point recent window. Only the refined 96-point
window enters the compact predictor. This is not a total-input-96 protocol.

## Installation

Use Python 3.10–3.12. Create an environment, then install a PyTorch build suitable
for the server GPU/driver using the [official selector](https://pytorch.org/get-started/locally/).
Do not replace an already working server PyTorch environment just to rename code.
For a CPU-only code check:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate using `.venv\Scripts\Activate.ps1`.
Dependency ranges are compatibility targets, not the historical training lock.
`docs/validation-environment.txt` records the environment actually checked for
this packaging release. Record your original GPU environment separately before
claiming matching experimental conditions.

## Quick code check (no real data needed)

```bash
python -m unittest discover -s tests -v
python -m tools.sarbf_net.smoke
python -m tools.sarbf_net.run --task ETTm1-96 --dry-run
```

The smoke test uses random series and temporary identity Q matrices. Those are
fixtures only and must not replace experiment Q matrices.

## Publish original experiment assets

Use [scripts/collect_release.py](scripts/collect_release.py) on the experiment
server to create one source + Q + selected-checkpoint + result archive. It verifies
recorded hashes, rejects conflicting duplicate seeds and excludes raw datasets.
See [docs/ASSETS_zh-CN.md](docs/ASSETS_zh-CN.md). The preparation archive itself
contains no server weights. [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) lists
maintainer-provided dataset download entries. After collecting assets, replay a
published checkpoint without training:

```bash
python -m tools.sarbf_net.evaluate --task ETTm1-96 --seed 2021 --device cpu
```

Published checkpoints under `artifacts/checkpoints/` are configured for Git LFS;
raw datasets and temporary run directories remain ignored.

## Reproduce a task

Prepare `dataset/` exactly as [docs/DATA.md](docs/DATA.md) describes. Use the
original dataset versions and Q matrices from the experiment server.

```bash
python -m tools.sarbf_net.inputs --tasks ETTm1-96
CUDA_VISIBLE_DEVICES=0 python -u -m tools.sarbf_net.run \
  --task ETTm1-96 --seed 2021 --run-root runs/sarbf_net
```

The runner records data/Q/source/checkpoint hashes, selects the checkpoint using
validation loss, then independently replays validation and test predictions.
Existing run folders are not overwritten. Choose a new run root after a failed
attempt, or move that failed task folder before retrying. `--validation-only`
records `VALIDATED` without evaluating the held-out test split.

## Matrix and aggregation

```bash
bash scripts/run_sarbf_matrix.sh --list-tasks
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sarbf_matrix.sh
python -m tools.sarbf_net.summarize --run-root runs/sarbf_net --output summary.csv
```

The sequential matrix script uses seeds 2021/2022/2023 and skips existing COMPLETE
results. It does not resume a partially trained checkpoint. Aggregation includes
only COMPLETE records, rejects duplicate task/arm/seed records, reports seed
counts and uses sample standard deviation (`ddof=1`); singleton std is left blank.

## Repository map

| Path | Purpose |
| --- | --- |
| `model/SARBF_Net.py` | Current full-model entry (`Model`) |
| `layers/SAER.py` | Paper-name frontend API |
| `layers/BalancedCrossIntraFeedback.py` | BCIF layers and coordinators |
| `tools/sarbf_net/` | Public run, preflight, export, summary and smoke commands |
| `scripts/run_sarbf_matrix.sh` | Sequential 52-task / three-seed launcher |
| `tools/sa_lcm*/`, `model/OLinear*.py` | Retained implementation and legacy replay paths |
| `tests/` | Numerical, checkpoint, protocol and packaging contracts |
| `docs/` | Data, reproduction, naming, audit and GitHub instructions |

See [docs/NAMING.md](docs/NAMING.md) for why old names remain in compatibility
code, [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the protocol, and
[docs/GITHUB_zh-CN.md](docs/GITHUB_zh-CN.md) for upload instructions.

## Attribution and citation

The inherited backbone and utilities retain their existing provenance comments.
Renaming does not imply that inherited OLinear/RevIN/attention/data utilities
were newly authored. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[LICENSE.md](LICENSE.md). The uploaded source did not contain a verified
repository-wide license or complete upstream license bundle; this package does
not invent one. Paper authors, publication details and DOI were not supplied;
see [CITATION.md](CITATION.md) for the current citation status.
