# Reproduction protocol

This release preserves the supplied `SA_LCM_FULL_MATRIX_52TASK_FINAL_V1` profile
values. It does not infer later experiments from conversation notes or substitute
new hyperparameters. `tools/sa_lcm/matrix_profiles.py` remains the single
profile source; `docs/task_profiles.json` is an exported reference snapshot.

## Inputs and tasks

- Total contiguous input: 768; seven historical blocks of 96 plus recent 96.
- Long-term datasets: ETTh1, ETTh2, ETTm1, ETTm2, ECL, Exchange, Traffic,
  Weather, Solar; horizons 96/192/336/720.
- PEMS03/04/07/08: horizons 12/24/48/96.
- Seeds: 2021/2022/2023 in the matrix script.
- 52 configurations; three seeds give 156 runs.
- All profile details (width, layer count, learning rate, dropout, batch sizes,
  epoch cap, patience, schedule, Q names) are in `docs/task_profiles.json`.

Exchange retains the original one-epoch cap. ETTm2 retains two layers. Do not
silently change these recipes when comparing against existing records. A later
paper's selected datasets/configurations must be cross-checked separately.

## Selection and replay

The locked config uses the inherited weighted L1 training/validation criterion
(`lossfun_alpha=0.5`, `loss_mode=L1`) and early stopping. Validation MSE/MAE are
reported after replay, not used retrospectively to choose a different test score.
The runner redirects the training loop's nominal test-loader requests to the
validation split until checkpoint selection, then independently evaluates the
selected checkpoint on all validation/test targets. `--validation-only` never
evaluates the test split.

Each record includes the resolved arguments, seed, source digest, data hash,
input/output Q hashes, checkpoint hash, target-index digests, diagnostics and
replayed metrics. Status `COMPLETE` requires those checks. Earlier legacy arms
use the inherited record schema. The naming update does not transform old
result files or make their source hashes match the new source.

The supplied PEMS loader preserves evaluation target positions from its
96-input protocol while supplying 768-point context. Split boundaries and
source row counting are tested in `tests/test_matrix_expansion.py`.

## Compatibility fixes in this release

- Replace a `timm` import used only for tuple conversion with PyTorch's equivalent
  `_pair`, avoiding unrelated torchvision operator failures.
- Replace removed NumPy alias `np.Inf` with `np.inf` in early stopping.
- Store Q tensors as non-persistent buffers so `.to(device)` follows the model
  while keeping historical checkpoint keys unchanged.
- Allow explicit CPU execution without an interactive prompt.
- Remove fixed user-specific Conda activation from shell launchers.
- Preserve externally selected CUDA visibility in both training and replay.
- Read the selected epoch from the saved checkpoint marker, rather than rounded
  printed losses (ties can otherwise report the wrong epoch).

No forecasting equation, training profile value or state-dict key is changed.
Bitwise CPU comparisons cover seeded weights, RNG state, outputs and strict
checkpoint loading. This does not establish identical GPU training trajectories
across software versions or hardware.

## Before claiming exact paper reproduction

The original archive excludes Q/data bytes, original environment export, trained
checkpoints and final result JSONs. Collect these from the experiment server and
cross-check the final manuscript's model/profile version. The separate SAER
cross-backbone plugin study and later visualization packages are not part of this
archive and have not been reconstructed here.

A working CPU smoke test proves code execution on synthetic fixtures only.
Record the actual server environment with:

```bash
python -m pip freeze > experiment-requirements.txt
nvidia-smi > experiment-gpu.txt
```

Retain those exports with the experiment assets. If a checkpoint's Q matrices
are missing, the weights alone are insufficient: Q is deliberately absent from
historical state dictionaries and must be supplied externally.
