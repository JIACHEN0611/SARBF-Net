# SARBF-Net

**Selective Anchored Retrieval & Balanced Feedback Network** for multivariate time series forecasting. [中文说明](README_zh-CN.md)

SARBF-Net combines selective historical retrieval (SAER) with balanced cross-variable and intra-variable feedback (BCIF). It uses 768 observations: seven historical blocks of 96 and a recent window of 96. SAER refines the recent window before prediction; the folded affine head avoids a redundant expanded projection.

We also evaluate SAER as an input frontend for **DLinear, PatchTST, iTransformer, CycleNet and TSMixer**. These transfer experiments use SAER without BCIF. The original and SAER groups share data windows and backbone training recipes.

## Installation

Install PyTorch for the target hardware, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python -m tools.sarbf_net.smoke
python -m tools.saer_transfer.smoke
```

The tests and smoke checks run on CPU without benchmark data or trained checkpoints. The recorded environment and verification scope are in [the validation report](docs/VALIDATION_REPORT.json).

## Data and trained assets

Raw datasets are downloaded separately; see [data sources](docs/DATA_SOURCES.md) and [file layout](docs/DATA.md). Training reads local files.

The [v1.0.0 asset release](https://github.com/JIACHEN0611/SARBF-Net/releases/tag/v1.0.0) contains matched Q matrices, checkpoints and experiment records for the main-model studies. Download all four `SARBF-Net_Assets_v1.7z.001`–`.004` parts into one directory and extract `.001` with 7-Zip. Place the extracted `artifacts/` beside `run.py`. The source ZIP contains no trained assets. See [asset layout and replay](docs/ASSETS_zh-CN.md).

The five-backbone code contains **720 task presets**, not 720 collected results. The v1.0.0 assets do not supply the trained checkpoints/results for this transfer matrix.

## Main-model experiments

Prepare the local dataset and matched Q files according to [DATA.md](docs/DATA.md), then run:

```bash
python -m tools.sarbf_net.inputs --tasks ETTm1-96
python -m tools.sarbf_net.run --task ETTm1-96 --seed 2021 --dry-run
python -m tools.sarbf_net.run --task ETTm1-96 --seed 2021 --run-root runs/sarbf_net
python -m tools.sarbf_net.summarize --run-root runs/sarbf_net --output summary.csv
```

`--validation-only` selects a checkpoint using training/validation data without test evaluation. The main matrix retains 52 task configurations and seeds 2021/2022/2023. Additional released traffic records are described by the asset manifest; their presence does not add new recipes to the 52-task launcher.

## SAER transfer experiments

```bash
python -m tools.saer_transfer.run --backbone DLinear --dataset ETTm1 --horizon 96 --seed 2021 --arm sa_frontend --dry-run
```

See [SAER_TRANSFER.md](docs/SAER_TRANSFER.md) for data paths, paired runs, batch execution and summaries.

## Implementation and reproducibility

`model/SARBF_Net.py` is the public main-model entry. Internal implementations are in `model/components/`; the short `OLinear*.py` files preserve earlier imports and source lineage. Five transfer backbones are in `model/backbones/`, with SAER entries in `model/saer_adapters/`. [Naming and checkpoint compatibility](docs/NAMING.md) explains the retained state keys.

Independent CPU comparisons cover seeded initialization, outputs, gradients, one optimizer step and strict checkpoint cross-loading against the archived implementations. They do not replace evaluation of published checkpoints on the original data or establish equivalence to later server `protocol_v2` changes. See [reproduction protocol](docs/REPRODUCIBILITY.md).

Source credits and component licenses are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Repository and paper citation information is in [CITATION.md](CITATION.md).
