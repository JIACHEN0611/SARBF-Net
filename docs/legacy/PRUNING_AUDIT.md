# SA-LCM Compact Final pruning audit

## Removed from the selected model

| Item | Effect | Decision |
|---|---:|---|
| Learned pair scorer | -16,513 parameters and no four-way pair tensor | Removed after three-seed confirmation |
| `output_fold_clean` arm | Duplicate graph with no independent mechanism | Removed |
| Clean → Pair-Free wrapper chain | Two candidate files replaced by one public model | Removed |
| Candidate planning documents | No runtime or reproduction value | Removed |
| BCIF mechanism variants in the main model package | Main registry clutter | Moved to `ablations/models` |

The folded output head retained from the previous release already removes
1,263,104 parameters for H=96 and 11,177,216 for H=720 under the audited
E=16, D=F=512 configuration.

## Components that should remain

| Component | Parameter/computation note | Reason to retain |
|---|---|---|
| Seven-value age bias | 7 parameters | Negligible saving; encodes temporal order |
| NULL head | 65 parameters | Defines history confidence and rejection |
| History confidence `m` | No separate large block | Formal ablations show dataset/seed-dependent degradation when removed |
| Trend/residual shared encoder | Material frontend capacity | Trend-only and raw-block screens were not uniformly safe |
| Bounded correction MLP | Main SA-LCM refinement | Removing it collapses the long-context mechanism |
| Balanced BCIF coordinator | Low-rank and zero-initialized | Main CSL/ISL collaboration contribution |
| Validation diagnostics | Opt-in, no forecast parameters | Required for evidence and reproducibility records |

## Source files that look generic but are still dependencies

`Transformer_EncDec.py`, `SelfAttention_Family.py`, `FANLayer.py`, `utils/CKA.py`
and the general data-loader modules contain inherited OLinear branches.  Some
are disabled by the selected configuration, but they are imported by the base
runtime.  Deleting them would require flattening and rewriting the inherited
backbone.  That would reduce repository size, not the selected model's live
parameters, and could change initialization or checkpoint compatibility.

## Final recommendation

Stop architecture pruning at this release.  Further deletion should be treated
as a new ablation study, not a safe cleanup.  The current primary graph has one
public entry, no live Pair Scorer, preserved checkpoint compatibility and
separate reproducibility-only ablations.
