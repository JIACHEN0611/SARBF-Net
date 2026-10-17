# Naming and checkpoint compatibility

| Purpose | Current entry | Compatibility entry |
| --- | --- | --- |
| Full model | `model.SARBF_Net.Model` | `model.SA_LCM.Model` |
| Runtime components | `model/components/` | five short `model/OLinear*.py` import bridges |
| Frontend access | `model.saer` property | registered module/state prefix `sa_lcm.*` |
| Experiment arm | `sarbf_net` | `sa_lcm`, `pair_free_fold` |
| Launcher | `tools.sarbf_net.run` | `tools.sa_lcm_transport.run_pilot` |
| Transfer backbones | `model/backbones/` | original `Model(configs)` interfaces |
| Transfer frontend | `model/saer_adapters/` | `backbone.*`, `frontend.*` state keys |

The main entry directly inherits `model.components.folded_head_model.Model`; its runtime does not depend on the OLinear import bridges. Components retain the inherited orthogonal backbone, history refinement and feedback implementation. The legacy registry keeps earlier CLI names callable. OLinear remains part of the source lineage and is credited in THIRD_PARTY_NOTICES.md.

We retain `sa_lcm.*` in saved checkpoints. The `saer` property gives access under the current method name without registering another module or duplicating state keys. Q matrices are non-persistent buffers: they follow the model device but remain external to historical checkpoints.

Transfer backbones and their versions of RevIN, attention and embedding layers have separate package paths. Importing them alongside the main model does not replace the main-model dependencies. The public transfer entries default to SAER; `arm=original` uses the recent 96 points from the same 768-point window and does not construct a frontend.

Legacy architecture IDs and result records retain their original strings and hashes. The new source digest changes after relocation; historical records are not rewritten to conceal that change.
