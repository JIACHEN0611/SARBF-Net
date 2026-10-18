# Dataset sources and acquisition instructions

Raw datasets are intentionally excluded from the SARBF-Net release. Download
prepared benchmark files from the sources below and place them under `dataset/`
using the paths in [DATA.md](DATA.md). The code reads local files; it does not
require a remote service during training or inference.

| Data | Maintainer/source | Local path used by this code |
| --- | --- | --- |
| ETTh1, ETTh2, ETTm1, ETTm2 | [ETDataset original repository](https://github.com/zhouhaoyi/ETDataset), directory `ETT-small` | `dataset/ETT-small/<name>.csv` |
| ECL, Exchange, Traffic, Weather | Prepared bundles linked by [OLinear](https://github.com/jackyue1994/OLinear) or [Time-Series-Library](https://github.com/thuml/Time-Series-Library#prepare-data) | `electricity/electricity.csv`, `exchange_rate/exchange_rate.csv`, `traffic/traffic.csv`, `weather/weather.csv` beneath `dataset/` |
| Solar | Prepared benchmark bundle linked by [OLinear](https://github.com/jackyue1994/OLinear) / [iTransformer](https://github.com/thuml/iTransformer#usage) | `dataset/Solar/solar_AL.txt` (137 variables) |
| PEMS03, PEMS04, PEMS07, PEMS08 | Prepared traffic files in the benchmark sources linked by [OLinear](https://github.com/jackyue1994/OLinear) / [iTransformer](https://github.com/thuml/iTransformer#usage) | `dataset/PEMS/<name>.npz` |

## Download entry points

- OLinear's prepared-data [Google Drive archive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view)
  and [Tsinghua Cloud mirror](https://cloud.tsinghua.edu.cn/f/2ea5ca3d621e4e5ba36a/),
  both linked from its official README.
- iTransformer's official README also links a [Baidu Cloud mirror](https://pan.baidu.com/s/11AWXg1Z6UwjHzmto4hesAA?pwd=9qjr)
  (access code shown in the link: `9qjr`).
- Time-Series-Library's README links [Baidu Cloud](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy)
  (`i9iy`) and the maintained [Hugging Face dataset collection](https://huggingface.co/datasets/thuml/Time-Series-Library).
- ETT CSVs can be downloaded directly from the original repository, for example:

```bash
mkdir -p dataset/ETT-small
curl -L --fail https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv \
  -o dataset/ETT-small/ETTm1.csv
```

The links above identify upstream acquisition channels. For checkpoint replay,
use the exact dataset version identified by the SHA256 in the experiment record.
Reformatted or differently processed benchmark files may have different hashes.

Download the original CSV/text/NPZ form expected by the loaders. A parquet export
or another PEMS/Solar variant is not automatically interchangeable. Check channel
counts and file names against `docs/task_profiles.json`. For exact checkpoint
replay, the new evaluator also checks the raw dataset SHA256 against the original
experiment record and stops if bytes differ; it does not silently accept a
reformatted CSV as the original dataset.

Do not download an upstream Q file and assume it is the one used by a trained
SARBF-Net checkpoint. Export the exact matched arrays from the experiment server
with the collection tool; their hashes are recorded alongside the weights.

## PEMSD7 and PEMS-BAY

We obtained these two datasets from the benchmark archive linked in the
[official FACT repository](https://github.com/wanghq21/FACT), using its
[Google Drive data link](https://drive.google.com/file/d/1MKugRwUKN2u9tIBgES-n3-QOT5l6unLl/view).
The local download was named `FACT_datasets.zip`. The Drive display name may
change; the file identifier above records the acquisition link.

- PEMSD7: the original `PEMSD7.npy` has shape 228 x 12,672 (variables x time).
  We orient it as 12,672 x 228 before preparing the NPZ input used in the final
  traffic experiments. This 228-variable speed dataset is different from
  the 883-variable traffic-flow dataset PEMS07.
- PEMS-BAY: the original `PEMS-BAY.csv` contains 52,116 time steps and
  325 traffic-speed variables. We retain the variable ordering when preparing
  the final model input.

The final experiments retain the FACT chronological 7:1:2 loader conventions,
including integer rounding. Target intervals are half-open:

| Dataset | Training | Validation targets | Test targets |
| --- | --- | --- | --- |
| PEMSD7 | [0, 8870) | [8870, 10137) | [10138, 12672) |
| PEMS-BAY | [0, 36481) | [36481, 41693) | [41693, 52116) |

The one-step separation in PEMSD7 follows the recorded loader and is retained
for consistency. Validation and test inputs may use preceding observations;
prediction targets remain in their respective partitions. Dataset-level
standardization is fitted on training data, and reported MSE/MAE are measured
in that standardized space. Inverse RevIN alone does not restore raw dataset units.

Our additional CGTFra, TimeCNN, and OLinear experiments cover PEMSD7 and
PEMS-BAY only: 3 models x 2 datasets x 4 horizons x 3 seeds = 72 runs.
The baseline runs marked with an asterisk do not include PEMS08.
Their source status is recorded in `research_data/csv/horizon_results_as_reported.csv`.

An earlier experimental branch converted FACT files into a TFB directory.
Those directory names do not identify an alternative source for the final
paper's two added datasets. The final experiments use the FACT protocol above.

## Scope and data rights

`research_data/csv/dataset_sources.csv` lists all fourteen datasets used in the
paper, their dimensions, splits, raw filenames, and acquisition links.
Exchange appears in older source-code presets and asset records but is not
one of the fourteen paper datasets and is excluded from the shared main-result CSVs.
ETT splits apply to the first twenty months, as specified in Supplementary Appendix E.1.
Solar uses 7:1:2; the different published 6:2:2 results are retained as cross-protocol
references under the ranking rules in Supplementary Appendix E.2.

We do not redistribute third-party raw benchmark files in this research-data
archive. Obtain them from the upstream sources and follow their terms.
