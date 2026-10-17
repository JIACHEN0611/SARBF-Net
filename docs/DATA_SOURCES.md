# Dataset download sources (checked 2026-10-15)

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

The cloud archives themselves were not downloaded/inspected during this
packaging task. These are maintainer-provided acquisition entry points, not
proof that every mirror contains the exact historical bytes used in your runs.
Baidu links were verified as links in the official READMEs; their download pages
could not be opened by the checking environment.

Download the original CSV/text/NPZ form expected by the loaders. A parquet export
or another PEMS/Solar variant is not automatically interchangeable. Check channel
counts and file names against `docs/task_profiles.json`. For exact checkpoint
replay, the new evaluator also checks the raw dataset SHA256 against the original
experiment record and stops if bytes differ; it does not silently accept a
reformatted CSV as the original dataset.

Do not download an upstream Q file and assume it is the one used by a trained
SARBF-Net checkpoint. Export the exact matched arrays from the experiment server
with the collection tool; their hashes are recorded alongside the weights.

## Added traffic assets

The collector also accepts PEMSD7 and PEMS-BAY task records. Their original data
are not bundled; retain the precise data bytes identified by each result's hash.
The new checkpoint evaluator uses the recorded arguments, rather than inventing
an additional training recipe. For any changed preprocessing in the traffic
extension source, verify replay against the recorded metrics before publishing.
