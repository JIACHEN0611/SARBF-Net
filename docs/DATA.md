# Data and original Q assets

Download links and mirrors: [DATA_SOURCES.md](DATA_SOURCES.md).
To bundle existing Q, weights and results with source: [ASSETS_zh-CN.md](ASSETS_zh-CN.md).

The source archive contains no dataset or Q-array bytes and no Q generation
script. Consequently the historical Q-generation algorithm, training boundary,
eigenvector order/sign conventions and original asset hashes cannot be recovered
from this archive alone. Do not claim that newly generated matrices reproduce
the published results. The exporter below copies existing experiment arrays.

## Expected layout

Place files under the repository's `dataset/` directory (or point that directory
to the dataset directory from the experiment server).

| Dataset | Relative raw-data path | Channels | Q name prefix | Q suffix |
| --- | --- | ---: | --- | --- |
| ETTh1/ETTh2/ETTm1/ETTm2 | `ETT-small/ETTh1.csv` (and corresponding names) | 7 | matching dataset name | `_ratio0.6.npy` |
| ECL | `electricity/electricity.csv` | 321 | `electricity` | `_ratio0.7.npy` |
| Exchange | `exchange_rate/exchange_rate.csv` | 8 | `exchange_rate` | `_ratio0.7.npy` |
| Traffic | `traffic/traffic.csv` | 862 | `traffic` | `_ratio0.7.npy` |
| Weather | `weather/weather.csv` | 21 | `weather` | `_ratio0.7.npy` |
| Solar | `Solar/solar_AL.txt` | 137 | `solar_AL` | `_ratio0.7.npy` |
| PEMS03/04/08 | `PEMS/PEMS03.npz` (and corresponding names) | 358/307/170 | matching dataset name | `_ratio0.6.npy` |
| PEMS07 | `PEMS/PEMS07.npz` | 883 | `PEMS07` | `.npy` (no ratio suffix) |

Q arrays live **beside the corresponding raw dataset**, with names
`<prefix>_<length><suffix>`:

- ETTm1-96: `dataset/ETT-small/ETTm1_96_ratio0.6.npy` (shared input/output file).
- ECL-720: `dataset/electricity/electricity_96_ratio0.7.npy` and
  `electricity_720_ratio0.7.npy`.
- PEMS07-12: `dataset/PEMS/PEMS07_96.npy` and `PEMS07_12.npy`.

Each input Q has shape `[96,96]`; output Q has shape `[horizon,horizon]`.
CSV datasets use a `date` column followed by variables; ETT includes `OT`.
Solar uses the inherited comma-separated numeric text loader. PEMS uses a
NumPy NPZ containing `data`, typically `[time,nodes,features]`; the inherited
loader forecasts the first feature. See `data_provider/data_loader.py` for the
exact preprocessing. Do not substitute another dataset merely because a name matches.

## Export existing Q arrays on the experiment server

Upload/extract this repository to the server and activate the existing working
environment. Replace only the path below with the directory containing your
original `ETT-small`, `weather`, `PEMS`, etc. subdirectories:

```bash
python -m tools.sarbf_net.export_q \
  --dataset-root /absolute/path/to/original/dataset \
  --output SARBF-Net_Q_assets.zip
```

If the server has only some tasks, export exactly those tasks:

```bash
python -m tools.sarbf_net.export_q \
  --dataset-root /absolute/path/to/original/dataset \
  --tasks ETTm1-96 ECL-720 PEMS04-96 \
  --output SARBF-Net_Q_assets_subset.zip
```

The command validates requested Q shapes/finiteness before writing, preserves
original bytes, includes SHA256 hashes, excludes raw datasets, and refuses to
overwrite an existing zip. A missing file causes an error. It does not search
unrelated directories or generate replacement matrices.

Extract the Q archive to the repository root; its entries begin with `dataset/`.
For example, `unzip SARBF-Net_Q_assets.zip -d .`. Record the hashes from
`Q_ASSETS_MANIFEST.json` with the corresponding experiment results.

## Link data and check inputs (Linux)

The shipped `dataset` contains only a README. To use an existing server directory,
rename that empty template and create a symbolic link:

```bash
mv dataset dataset_template
ln -s /absolute/path/to/original/dataset dataset
python -m tools.sarbf_net.inputs --tasks ETTm1-96
```

Alternatively copy datasets/Q arrays into the shipped directory. Check all 52
configurations with `python -m tools.sarbf_net.inputs`. Preflight verifies data
existence and Q shape/finiteness; it is not proof of the raw dataset's provenance
or of a leakage-free historical Q estimation procedure. Run records store hashes
so actual assets can be compared.

The newly added DATA_SOURCES.md lists maintainer-provided download entries.
Exact historical versions were not supplied in the original archive; use the
recorded data hashes when replaying a published checkpoint.
