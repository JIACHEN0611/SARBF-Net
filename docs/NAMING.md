# Naming and checkpoint compatibility

| Purpose | Current name | Preserved compatibility name |
| --- | --- | --- |
| Paper/model | SARBF-Net | SA-LCM / SA_LCM |
| Python full-model entry | `model.SARBF_Net.Model` | `model.SA_LCM.Model` (alias) |
| Experiment arm | `sarbf_net` | `sa_lcm`, `pair_free_fold` |
| Frontend | SAER (`layers/SAER.py`) | `SelectiveAnchoredLongContext` implementation |
| Frontend access in the full model | `model.saer` property | `model.sa_lcm` module/state prefix |
| Backend | BCIF | `BalancedCrossIntraFeedback` |
| Public launcher | `tools.sarbf_net.run` | `tools.sa_lcm_transport.run_pilot` |
| Batch script | `scripts/run_sarbf_matrix.sh` | `scripts/run_sa_lcm_matrix.sh` |

The actual current forward implementation lives in `model/SARBF_Net.py`.
`model/SA_LCM.py` is only a backwards-compatible import. New experiment records
use `model=SARBF_Net`, `arm=sarbf_net`, and named output directories.
The inherited runner's `PRIMARY_ARM` stays `sa_lcm` for its historical API; the
new public runner always explicitly selects `sarbf_net`.

The saved parameter prefix `sa_lcm.*` intentionally remains unchanged. Renaming
that prefix or registering a second `self.saer` module could duplicate state
keys or break strict loading of old checkpoints. The `saer` property exposes the
paper name without changing registered parameters. The standalone `SAER` class
uses the selected cosine selector; the full model preserves its original
construction order before installing that selector.

Legacy result architecture IDs and environment flags retain their old strings
so old JSON/log readers remain usable. This is a packaging release of the same
selected architecture, not a new experimental revision with new scores.
The source-tree hash changes, as expected for modified source code.

`OLinear*.py` names identify inherited implementation paths and earlier
ablation variants. Their retention records the backbone lineage and permits
old checkpoint replay. A global string replacement would obscure provenance
and change imports/state keys without improving the algorithm.
