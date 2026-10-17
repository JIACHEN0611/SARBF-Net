"""Train a registered SA-LCM/BCIF arm and replay its selected checkpoint."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import re

from tools.sa_lcm.matrix_profiles import MATRIX_TASKS


REVISION = "SA_LCM_TRANSPORT_V1"
BCIF_REVISION = "SA_LCM_BCIF_V1"
SAFE_REVISION = "SA_LCM_SAFE_PRUNE_V1"
OUTPUT_FOLD_REVISION = "SA_LCM_OUTPUT_FOLD_V1"
NO_MASS_REVISION = "SA_LCM_BCIF_NO_MASS_V1"
UNBALANCED_REVISION = "SA_LCM_BCIF_UNBALANCED_V1"
FINAL_REVISION = "SA_LCM_COMPACT_FINAL_V1"
PAIR_FREE_REVISION = FINAL_REVISION
PILOT = MATRIX_TASKS
PRIMARY_ARM = "sa_lcm"
SAFE_MODELS = {
    PRIMARY_ARM: "SA_LCM",
    "no_diag": "OLinear_SALCM_BCIF_NoDiag",
    "output_fold": "OLinear_SALCM_BCIF_OutputFold",
    "no_mass_bcif": "OLinear_SALCM_BCIF_OutputFold_NoMass",
    "unbalanced_bcif": "OLinear_SALCM_BCIF_OutputFold_Unbalanced",
    # Command compatibility for the experiments used to select the final graph.
    "pair_free_fold": "SA_LCM",
    "sarbf_net": "SARBF_Net",
}
ARCHITECTURES = {"bcif": BCIF_REVISION, "no_diag": SAFE_REVISION,
                 "output_fold": OUTPUT_FOLD_REVISION,
                 "no_mass_bcif": NO_MASS_REVISION,
                 "unbalanced_bcif": UNBALANCED_REVISION,
                 PRIMARY_ARM: FINAL_REVISION,
                 "pair_free_fold": FINAL_REVISION,
                 "sarbf_net": FINAL_REVISION}
ARMS = tuple(SAFE_MODELS)
SOURCE_FILES = (
    'ablations/__init__.py',
    'ablations/models/OLinear_SALCM_BCIF_OutputFold_NoMass.py',
    'ablations/models/OLinear_SALCM_BCIF_OutputFold_Unbalanced.py',
    'ablations/models/__init__.py',
    'data_provider/__init__.py',
    'data_provider/data_factory.py',
    'data_provider/data_loader.py',
    'data_provider/dataset_physio.py',
    'data_provider/dataset_pm25.py',
    'data_provider/long_context_factory.py',
    'data_provider/m4.py',
    'data_provider/split_geometry.py',
    'experiments/__init__.py',
    'experiments/exp_basic.py',
    'experiments/exp_forecast.py',
    'experiments/exp_forecast_long_context.py',
    'layers/BalancedCrossIntraFeedback.py',
    'layers/CosineHistoricalSelector.py',
    'layers/Embed.py',
    'layers/FANLayer.py',
    'layers/RevIN.py',
    'layers/SAER.py',
    'layers/SafePruneFrontend.py',
    'layers/SelectiveAnchoredLongContext.py',
    'layers/SelfAttention_Family.py',
    'layers/Transformer_EncDec.py',
    'layers/__init__.py',
    'layers/newLinear.py',
    'model/OLinear.py',
    'model/OLinear_SALCM.py',
    'model/OLinear_SALCM_BCIF.py',
    'model/OLinear_SALCM_BCIF_NoDiag.py',
    'model/OLinear_SALCM_BCIF_OutputFold.py',
    'model/SARBF_Net.py',
    'model/SA_LCM.py',
    'model/__init__.py',
    'run.py',
    'scripts/collect_release.py',
    'tests/test_asset_release.py',
    'tests/test_bcif_ablation.py',
    'tests/test_compact_pairfree.py',
    'tests/test_final_release.py',
    'tests/test_matrix_expansion.py',
    'tests/test_output_fold.py',
    'tests/test_output_fold_replay.py',
    'tests/test_sarbf_release.py',
    'tests/test_slim_release.py',
    'tools/__init__.py',
    'tools/sa_lcm/__init__.py',
    'tools/sa_lcm/matrix_profiles.py',
    'tools/sa_lcm/run_controlled.py',
    'tools/sa_lcm_safe_prune/__init__.py',
    'tools/sa_lcm_safe_prune/replay.py',
    'tools/sa_lcm_transport/__init__.py',
    'tools/sa_lcm_transport/run_pilot.py',
    'tools/sarbf_net/__init__.py',
    'tools/sarbf_net/evaluate.py',
    'tools/sarbf_net/export_q.py',
    'tools/sarbf_net/inputs.py',
    'tools/sarbf_net/run.py',
    'tools/sarbf_net/smoke.py',
    'tools/sarbf_net/summarize.py',
    'utils/CKA.py',
    'utils/__init__.py',
    'utils/masking.py',
    'utils/metrics.py',
    'utils/timefeatures.py',
    'utils/tools.py',
)


def source_tree_digest(repository_root):
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        file = Path(repository_root) / name
        digest.update(name.encode("utf-8") + b"\0")
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def scope_target_manifest(config, validation_only=False):
    """Record exactly the target splits used in a run, without test in a screen."""
    if not validation_only:
        from tools.sa_lcm.run_controlled import _target_manifest
        return _target_manifest(config)
    from data_provider.long_context_factory import (
        long_context_data_provider, target_indices_for_dataset,
    )

    args = SimpleNamespace(
        task_name="forecasting", data=config["data"], root_path=config["root_path"],
        data_path=config["data_path"], features="M", target="OT", freq="h",
        embed="timeF", seq_len=96, long_len=768, label_len=0,
        pred_len=config["pred_len"], batch_size=config["batch_size"],
        test_batch_size=config["test_batch_size"], num_workers=0,
        train_ratio=1.0, train_part_first=0.0,
    )
    manifest = {}
    for split in ("train", "valid"):
        dataset, _ = long_context_data_provider(args, split)
        indices = [int(value) for value in target_indices_for_dataset(dataset)]
        manifest[split] = {
            "count": len(indices),
            "first_target": indices[0] if indices else None,
            "last_target": indices[-1] if indices else None,
            "digest": hashlib.sha256(
                json.dumps(indices, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    return manifest


def run_directory_occupied(run_dir, validation_only):
    """A final run may share its parent folder with an earlier validation run."""
    run_dir = Path(run_dir)
    return run_dir.exists() and any(
        validation_only or item.name != "validation" or not item.is_dir()
        for item in run_dir.iterdir()
    )


def assess_result(record):
    """Validate a newly trained registered-arm record."""
    return _assess_record(record, ARMS)


def assess_legacy_bcif_reference(record):
    """Validate an old BCIF result without adding BCIF to the training registry."""
    return _assess_record(record, ("bcif",))


def _assess_record(record, allowed_arms):
    if record.get("return_code") != 0:
        return "FAILED" if record.get("return_code") is not None else "MALFORMED"
    arm, args = record.get("arm"), record.get("args") or {}
    expected_model = {"bcif": "OLinear_SALCM_BCIF", **SAFE_MODELS}.get(arm)
    expected_architecture = ARCHITECTURES.get(arm)
    scope = record.get("evaluation_scope", "complete")
    if (arm not in allowed_arms or record.get("implementation_revision") != REVISION
            or scope not in ("complete", "validation_only")
            or args.get("model") != expected_model
            or args.get("seq_len") != 96 or args.get("long_len") != 768
            or args.get("run_seed") != record.get("seed")
            or record.get("architecture_revision") != expected_architecture):
        return "MALFORMED"
    if not isinstance(record.get("best_epoch"), int):
        return "MALFORMED"
    if scope == "validation_only" and record.get("test") is not None:
        return "MALFORMED"
    if scope == "validation_only" and "test" in (record.get("sample_manifest") or {}):
        return "MALFORMED"
    for split in (("validation",) if scope == "validation_only" else
                  ("validation", "test")):
        metrics = record.get(split) or {}
        try:
            if (not all(math.isfinite(float(metrics[field])) for field in ("mse", "mae"))
                    or int(metrics.get("samples", 0)) < 1):
                return "MALFORMED"
        except (KeyError, TypeError, ValueError, OverflowError):
            return "MALFORMED"
    checkpoint = record.get("checkpoint") or {}
    if not checkpoint.get("path") or not _digest(checkpoint.get("sha256")):
        return "MALFORMED"
    if (not _digest(record.get("data_sha256"))
            or not _digest(record.get("source_tree_sha256"))
            or record.get("provenance_verified") is not True
            or record.get("artifact_error")):
        return "MALFORMED"
    for name in (("validation_inference_seconds",) if scope == "validation_only" else
                 ("validation_inference_seconds", "test_inference_seconds")):
        try:
            if not math.isfinite(float(record[name])) or float(record[name]) < 0:
                return "MALFORMED"
        except (KeyError, TypeError, ValueError, OverflowError):
            return "MALFORMED"
    q = record.get("q_identity") or {}
    if q.get("paths_equal") is not True or not all(
        _digest((q.get("sha256") or {}).get(field))
        for field in ("q_mat_file", "q_out_mat_file")
    ):
        return "MALFORMED"
    for split in (("train", "valid") if scope == "validation_only" else
                  ("train", "valid", "test")):
        targets = (record.get("sample_manifest") or {}).get(split) or {}
        if targets.get("count", 0) < 1 or not _digest(targets.get("digest")):
            return "MALFORMED"
    diagnostics = record.get("diagnostics") or {}
    if (diagnostics.get("diagnostics_scope") != "full_validation"
            or diagnostics.get("samples") != record["validation"]["samples"]):
        return "MALFORMED"
    if (record["sample_manifest"]["valid"]["count"] != record["validation"]["samples"]
            or (scope == "complete" and record["sample_manifest"]["test"]["count"]
                != record["test"]["samples"])):
        return "MALFORMED"
    return "VALIDATED" if scope == "validation_only" else "COMPLETE"


def selected_checkpoint_epoch(checkpoint):
    """Read the trainer's checkpoint marker, avoiding rounded-loss log guesses."""
    markers = list(Path(checkpoint).parent.glob("Epoch_*.txt"))
    if len(markers) != 1:
        raise ValueError("expected exactly one selected checkpoint epoch marker")
    match = re.fullmatch(r"Epoch_([1-9][0-9]*)\.txt", markers[0].name)
    if match is None:
        raise ValueError("invalid selected checkpoint epoch marker")
    return int(match.group(1))


def evaluate_split(args_snapshot, checkpoint, split):
    """Independently replay the selected checkpoint on every target in the split."""
    import torch
    from experiments.exp_forecast_long_context import Exp_Forecast_LongContext

    args = SimpleNamespace(**args_snapshot)
    experiment = Exp_Forecast_LongContext(args)
    experiment.model.load_state_dict(torch.load(checkpoint, map_location=experiment.device, weights_only=True),
                                     strict=True)
    if experiment.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(experiment.device)
    _, loader = experiment._get_data(flag=split)
    module = experiment.model.module if hasattr(experiment.model, "module") else experiment.model
    if split == "valid":
        module.sa_lcm.reset_diagnostics()
    squared = absolute = 0.0
    elements = samples = 0
    inference_seconds = 0.0
    try:
        experiment.model.eval()
        with torch.no_grad():
            for x, y, x_mark, y_mark in loader:
                x = x.float().to(experiment.device)
                y = y.float().to(experiment.device)
                if experiment.device.type == "cuda":
                    torch.cuda.synchronize(experiment.device)
                started = time.perf_counter()
                prediction = experiment.model(
                    x, x_mark.float().to(experiment.device),
                    y, y_mark.float().to(experiment.device),
                )
                if experiment.device.type == "cuda":
                    torch.cuda.synchronize(experiment.device)
                inference_seconds += time.perf_counter() - started
                if isinstance(prediction, tuple):
                    prediction = prediction[0]
                first = -1 if args.features == "MS" else 0
                error = (prediction[:, -args.pred_len:, first:]
                         - y[:, -args.pred_len:, first:]).double()
                if not torch.isfinite(error).all().item():
                    raise ValueError(f"nonfinite {split} predictions")
                squared += error.square().sum().item()
                absolute += error.abs().sum().item()
                elements += error.numel()
                samples += x.shape[0]
        if samples != len(loader.dataset) or elements == 0:
            raise ValueError(f"incomplete {split} replay: {samples}/{len(loader.dataset)}")
        result = {"mse": squared / elements, "mae": absolute / elements, "samples": samples}
        timing = {f"{split if split == 'test' else 'validation'}_inference_seconds": inference_seconds}
        if split == "valid":
            return {"validation": result,
                    **timing,
                    "diagnostics": module.sa_lcm.finalize_diagnostics(samples),
                    "total_params": sum(p.numel() for p in module.parameters()),
                    "evaluation_peak_gpu_bytes": (torch.cuda.max_memory_allocated()
                                                   if experiment.device.type == "cuda" else None)}
        return {"test": result, **timing}
    finally:
        if split == "valid":
            module.sa_lcm.cancel_diagnostics()


def run_task(task, repository_root, run_root, seed=2021, arm="no_diag",
             validation_only=False):
    if task not in PILOT or arm not in ARMS:
        raise ValueError(f"task or arm not registered: {task}, {arm}")
    from tools.sa_lcm.run_controlled import (
        _command, _q_identity, _sha256,
        build_configs, parse_full_args,
    )
    repository_root, run_root = Path(repository_root).resolve(), Path(run_root).resolve()
    run_dir = run_root / f"seed{seed}" / task / arm
    if validation_only:
        run_dir = run_dir / "validation"
    if run_directory_occupied(run_dir, validation_only):
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    _, config = build_configs(task, run_root, repository_root)
    selected_model = SAFE_MODELS[arm]
    config.update(model=selected_model,
                  run_seed=seed, model_id=f"{task}_{selected_model}_{arm}_seed{seed}",
                  des=f"{selected_model}_{arm}_seed{seed}",
                  checkpoints=str(run_dir / "checkpoints"))
    # Missing data or Q files should not reserve the run directory: users can
    # repair a dataset symlink and retry without removing any run artifacts.
    input_dir = Path(config["root_path"])
    for name in (config["data_path"], config["q_mat_file"], config["q_out_mat_file"]):
        source = input_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"pilot input missing: {source}")
    run_dir.mkdir(parents=True, exist_ok=True)
    result = {"implementation_revision": REVISION,
              "architecture_revision": ARCHITECTURES[arm],
              "evaluation_scope": "validation_only" if validation_only else "complete",
              "task": task, "arm": arm, "seed": seed,
              "return_code": None, "checkpoint": None, "validation": None, "test": None,
              "args": None, "checkpoint_selection_metric": "validation_weighted_l1",
              "architecture_selection_metric": "validation_only",
              "provenance_verified": False}
    try:
        result["q_identity"] = _q_identity(config, config)
        if not all(result["q_identity"]["sha256"].values()):
            raise FileNotFoundError("required Q matrices missing")
        data_path = Path(config["root_path"]) / config["data_path"]
        result["data_sha256"] = _sha256(data_path)
        result["source_tree_sha256"] = source_tree_digest(repository_root)
        if (repository_root / ".git").exists():
            revision = subprocess.run(["git", "rev-parse", "HEAD"],
                                      cwd=repository_root, capture_output=True, text=True,
                                      check=False)
            result["git_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
        # Do not instantiate the held-out test dataset before checkpoint selection.
        result["sample_manifest"] = scope_target_manifest(config, validation_only=True)
        result["command"] = _command(config, repository_root)
        env = {**os.environ, "PYTHONPATH": str(repository_root), "PYTHONUNBUFFERED": "1",
               "SA_LCM_TRANSPORT_MONITOR_MEMORY": "1",
               "SA_LCM_PRUNE_TEST_BLIND": "1"}
        if validation_only:
            env["SA_LCM_PRUNE_VALIDATION_ONLY"] = "1"
        start = time.monotonic()
        with (run_dir / "train.log").open("x", encoding="utf-8") as log:
            completed = subprocess.run(result["command"], cwd=run_dir, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, check=False)
        result["runtime_seconds"] = time.monotonic() - start
        result["return_code"] = completed.returncode
        if completed.returncode == 0:
            log_text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
            result["args"] = parse_full_args(log_text)
            peak = re.findall(r"SA_LCM_TRANSPORT_TRAIN_PEAK_GPU_BYTES:\s*(\d+)", log_text)
            result["training_peak_gpu_bytes"] = int(peak[-1]) if peak else None
            candidates = list((run_dir / "checkpoints").rglob("checkpoint.pth"))
            if len(candidates) != 1:
                raise ValueError(f"expected one selected checkpoint, found {len(candidates)}")
            selected = candidates[0].resolve()
            result["best_epoch"] = selected_checkpoint_epoch(selected)
            result["checkpoint"] = {"path": str(selected), "sha256": _sha256(selected)}
            result.update(evaluate_split(result["args"], selected, "valid"))
            if not validation_only:
                result["sample_manifest"] = scope_target_manifest(config, validation_only=False)
                result.update(evaluate_split(result["args"], selected, "test"))
            if (_sha256(data_path) != result["data_sha256"]
                    or _q_identity(config, config)["sha256"] != result["q_identity"]["sha256"]
                    or scope_target_manifest(config, validation_only) != result["sample_manifest"]
                    or _sha256(selected) != result["checkpoint"]["sha256"]
                    or source_tree_digest(repository_root) != result["source_tree_sha256"]):
                raise ValueError("data, Q, targets, checkpoint, or source changed during run")
            result["provenance_verified"] = True
    except Exception as error:
        result["artifact_error"] = f"{type(error).__name__}: {error}"
    result["status"] = assess_result(result)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=PILOT, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--arm", choices=ARMS, default="no_diag")
    parser.add_argument("--validation-only", action="store_true",
                        help="train and replay only validation; record status VALIDATED")
    args = parser.parse_args(argv)
    result = run_task(args.task, args.repository_root, args.run_root, args.seed, args.arm,
                      validation_only=args.validation_only)
    print(json.dumps({name: result.get(name) for name in
                      ("task", "arm", "status", "validation", "test", "artifact_error")},
                     indent=2, ensure_ascii=False))
    return 0 if result["status"] in ("COMPLETE", "VALIDATED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
