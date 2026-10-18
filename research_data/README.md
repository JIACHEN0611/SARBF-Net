# SARBF-Net research data

This archive accompanies the SARBF-Net manuscript. We provide numerical results,
benchmark acquisition instructions, and data used in the additional analyses.
The source-code repository is https://github.com/JIACHEN0611/SARBF-Net.

## Contents

| File in research_data/csv | Coverage | Record level |
| --- | --- | --- |
| main_per_seed.csv | 14 datasets x 4 horizons x 3 seeds = 168 records | Original recorded test metrics, full available precision |
| main_summary.csv | 56 forecasting configurations | Mean and sample SD computed from those three seeds |
| horizon_results_as_reported.csv | General-data comparisons and four-model traffic comparison | Paper summaries; author runs and reported baselines are explicitly distinguished |
| saer_transfer_summary.csv | 5 backbones x 6 datasets x 4 horizons x 2 variants = 240 records | Three-seed means, as displayed in the paper |
| ablation_summary.csv | 6 tasks x 7 model variants = 42 records | Three-seed means, SDs and reported percentage changes |
| saer_mechanism_summary.csv | 8 backbone/task pairs x 6 variants x 2 metrics = 96 records | Three-seed means and SDs |
| depth_per_seed.csv | 2 tasks x 4 depths x 3 seeds x 2 variants = 48 records | Original per-seed metrics |
| history_blocks_per_seed.csv | 2 tasks x 5 history budgets x 3 seeds = 30 records | Original per-seed metrics |
| efficiency_summary.csv | 16 models on ECL-720 | Hardware measurement summaries, with parameters, MACs, time and memory |
| dataset_sources.csv | All 14 paper datasets | Upstream acquisition links and protocol metadata |

`research_data/paper_tables/` contains CSV exports of all 35 native scientific
tables or table parts in the latest main manuscript and supplement. `index.csv`
maps each export to its source caption. Merged cells are repeated across their
grid positions. Multi-paragraph cell contents remain inside quoted CSV fields.
These exports retain displayed values and are not raw run logs.

`research_data/plot_data/` contains G1 prediction curves, selected-window metadata,
the plotting script, and history/depth summaries. G1 values describe the displayed
prediction windows; they do not replace full-test-set metrics.
`research_data/provenance/` contains sanitized main-run metadata, data/checkpoint
hashes, efficiency records, and validation results. Private server paths and
unrelated experiment records are omitted.

## Statistics and precision

Seeds are 2021, 2022 and 2023. Means are arithmetic means across seeds;
SD is sample standard deviation (ddof=1). MSE/MAE use the training-standardized
metric space described in Supplementary Appendix E.2. The main model accesses
768 observed steps while its forecasting backbone receives 96 recent steps.

Legacy arm names in `depth_per_seed.csv` are preserved: `sa_lcm` denotes the
full SARBF-Net model, and `saer_only` denotes SARBF-Net without BCIF.
`K` is backbone depth; `history_blocks` counts the historical candidates,
excluding the recent block. Efficiency SD is over measured epochs, not random seeds.

Main-run means were checked against all 56 horizon-specific paper entries.
Both metrics agree at the paper's display precision. Original seed values are
retained without rounding in `main_per_seed.csv`.

The SAER transfer CSV is transcribed from three-decimal paper means.
Do not recompute the paper's percentage improvements or win counts from these
rounded values. Reported gains and aggregate win counts remain available in
the corresponding native table exports. Ablation percentages retain the paper's
definition, `(variant/full - 1) * 100`; they were not optimized by choosing a
different denominator. Missing SD cells are blank, not zero.

## Availability of individual runs

This archive includes per-seed main-model, history-budget and depth records.
SAER transfer experiments, main ablation variants, and the 72 additional traffic
baseline runs are shared here at the paper-summary level. Their individual seed
logs have not been included in the files recovered for this archive. We do not
reconstruct missing seed results from means or SDs.

`tools/export_existing_results.py` can export existing server records without
training. Its output can be added in a later release without replacing the
paper-summary files. Pretrained weights and Q arrays are separate repository
release assets and are not included in this small numerical-data archive.

## Benchmark sources

See `docs/DATA_SOURCES.md` and `research_data/csv/dataset_sources.csv`.
PEMSD7 and PEMS-BAY were obtained from FACT's official data link, not inferred
from a similarly named PEMS07 dataset. Only these two added traffic datasets
carry our reproduced-baseline designation for CGTFra, TimeCNN and OLinear.

## File integrity

`SHA256SUMS.txt` lists file checksums. CSV files use UTF-8 with a BOM for Excel/WPS.
Numeric measurements remain numerical fields; metric units appear in headers.
The accompanying Excel workbook provides the principal result tables in nine sheets.
