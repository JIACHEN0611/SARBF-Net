"""Read-only comparison of an original BCIF checkpoint with safe variants.

Only the validation split is constructed; no held-out test targets or metrics
are evaluated. Existing BCIF records and checkpoints are never modified.
"""

import argparse
import ast
from collections.abc import Mapping
import hashlib
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from tools.sa_lcm_transport.run_pilot import (
    PILOT, SAFE_MODELS, assess_legacy_bcif_reference,
    scope_target_manifest, source_tree_digest,
)


# Historical checkpoint replay is intentionally limited to graph-compatible
# controls. Fresh ablation coordinators are trained by run_pilot instead.
MODELS = {
    "bcif": "OLinear_SALCM_BCIF",
    "no_diag": SAFE_MODELS["no_diag"],
    "output_fold": SAFE_MODELS["output_fold"],
}
MSE_MAE_MAX_RELATIVE_DIFFERENCE = 0.0002  # 0.02% on both validation metrics
PREDICTION_MAX_ABSOLUTE_DIFFERENCE = 0.001


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def original_source_digest(original_root):
    """Use the original release's locked file list, not this release's list."""
    root = Path(original_root).resolve()
    runner = root / "tools/sa_lcm_transport/run_pilot.py"
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    files = next((ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "SOURCE_FILES"
                          for target in node.targets)), None)
    if not isinstance(files, tuple) or not all(isinstance(name, str) for name in files):
        raise ValueError("original SOURCE_FILES list is not a fixed tuple of paths")
    digest = hashlib.sha256()
    for name in files:
        path = root / name
        digest.update(name.encode("utf-8") + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def verify_reference(record, old_result, checkpoint, original_root):
    """Reject stale source, changed data, Q, checkpoint, or target geometry."""
    if (record.get("status") != "VALIDATED"
            or assess_legacy_bcif_reference(record) != "VALIDATED"
            or record.get("task") not in PILOT
            or record.get("arm") != "bcif"
            or record.get("architecture_revision") != "SA_LCM_BCIF_V1"
            or record.get("evaluation_scope") != "validation_only"
            or record.get("test") is not None):
        raise ValueError("reference must be a verified, validation-only BCIF record")
    args = record.get("args") or {}
    if args.get("model") != MODELS["bcif"] or args.get("run_seed") != record.get("seed"):
        raise ValueError("reference model or seed mismatch")
    old_result = Path(old_result).resolve()
    if (old_result.name != "result.json" or old_result.parent.name != "validation"
            or old_result.parent.parent.name != "bcif"
            or old_result.parent.parent.parent.name != record.get("task")
            or old_result.parent.parent.parent.parent.name != f"seed{record['seed']}"):
        raise ValueError("reference result path does not match task, arm, and seed")
    if original_source_digest(original_root) != record.get("source_tree_sha256"):
        raise ValueError("original source SHA256 differs from validated record")
    if sha256(checkpoint) != (record.get("checkpoint") or {}).get("sha256"):
        raise ValueError("selected checkpoint SHA256 differs from validated record")
    data_path = Path(args["root_path"]) / args["data_path"]
    if sha256(data_path) != record.get("data_sha256"):
        raise ValueError("data SHA256 differs from validated record")
    from tools.sa_lcm.run_controlled import _q_identity

    q = _q_identity(args, args)
    if (q["paths_equal"] is not True
            or q["sha256"] != record["q_identity"]["sha256"]):
        raise ValueError("input/output Q SHA256 differs from validated record")
    expected = scope_target_manifest(args, validation_only=True)
    if expected != record.get("sample_manifest"):
        raise ValueError("training/validation target manifest differs from validated record")
    return {"source": record["source_tree_sha256"],
            "checkpoint": record["checkpoint"]["sha256"],
            "data": record["data_sha256"], "q": q["sha256"],
            "valid_targets": expected["valid"]}


def _relative_change(current, reference):
    return abs(current - reference) / max(abs(reference), 1e-12)


def _load_replay_models(args, device, weights):
    """Load the original state strictly, converting only the folded output arm."""
    modules = {}
    counts = {}
    for arm, module_name in MODELS.items():
        model_module = importlib.import_module("model." + module_name)
        module = model_module.Model(args).to(device)
        state = (model_module.convert_old_bcif_state_dict(module, weights)
                 if arm == "output_fold" else weights)
        if not isinstance(state, Mapping) or set(module.state_dict()) != set(state):
            raise ValueError(f"{arm}: checkpoint parameter/buffer keys changed")
        module.load_state_dict(state, strict=True)
        module.eval()
        modules[arm] = module
        counts[arm] = sum(parameter.numel() for parameter in module.parameters())
    # no_diag is a graph-preserving control. The folded output has fewer
    # parameters by design, so it is checked by conversion and predictions.
    if any(count != counts["bcif"] for arm, count in counts.items()
           if arm != "output_fold"):
        raise ValueError("checkpoint-compatible variants must retain identical parameters")
    if "output_fold" in counts:
        horizon, embedding, width, feedforward = (
            args.pred_len, args.embed_size, args.d_model, args.d_ff
        )
        saved = (embedding * horizon * (width + feedforward + 1)
                 - feedforward * (width + horizon))
        if saved <= 0 or counts["bcif"] - counts["output_fold"] != saved:
            raise ValueError("output_fold parameter reduction differs from affine-fold formula")
    return modules, counts


def replay(old_result, original_root, output, checkpoint=None):
    import torch
    from torch.utils.data import DataLoader
    from data_provider.long_context_factory import (
        long_context_data_provider, target_indices_for_dataset,
    )

    old_result = Path(old_result).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite replay: {output}")
    record = json.loads(old_result.read_text(encoding="utf-8"))
    checkpoint = Path(checkpoint or record["checkpoint"]["path"]).resolve()
    hashes = verify_reference(record, old_result, checkpoint, original_root)
    new_root = Path(__file__).resolve().parents[2]
    comparison_source_sha256 = source_tree_digest(new_root)
    args = SimpleNamespace(**record["args"])
    if args.use_gpu and args.gpu != 0:
        raise ValueError("original OLinear Q tensors require a cuda:0 replay")
    device = torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")
    if args.use_gpu and device.type != "cuda":
        raise RuntimeError("reference used GPU; CUDA is required for this comparison")

    try:
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # older PyTorch releases do not support weights_only
        weights = torch.load(checkpoint, map_location="cpu")
    if not isinstance(weights, dict):
        raise ValueError("reference checkpoint must be a model state_dict")
    modules, counts = _load_replay_models(args, device, weights)

    dataset, _ = long_context_data_provider(args, "valid")
    target_indices = target_indices_for_dataset(dataset)
    if (len(dataset) != hashes["valid_targets"]["count"]
            or len(target_indices) != len(dataset)):
        raise ValueError("validation dataset differs from locked manifest")
    # One ordered loader and exactly the same batch for all model variants.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, drop_last=False)
    sums = {arm: {"squared": 0.0, "absolute": 0.0, "elements": 0,
                  "samples": 0} for arm in MODELS}
    comparison_arms = tuple(arm for arm in MODELS if arm != "bcif")
    diffs = {arm: {"sum_squares": 0.0, "elements": 0,
                   "max_abs": 0.0, "worst_target": None}
             for arm in comparison_arms}
    pointer = 0
    with torch.no_grad():
        for x, y, x_mark, y_mark in loader:
            x = x.float().to(device)
            y = y.float().to(device)
            x_mark = x_mark.float().to(device)
            y_mark = y_mark.float().to(device)
            predictions = {}
            for arm, module in modules.items():
                prediction = module(x, x_mark, y, y_mark)
                if isinstance(prediction, tuple):
                    prediction = prediction[0]
                first = -1 if args.features == "MS" else 0
                prediction = prediction[:, -args.pred_len:, first:]
                truth = y[:, -args.pred_len:, first:]
                if (prediction.shape != truth.shape or not bool(torch.isfinite(prediction).all())
                        or not bool(torch.isfinite(truth).all())):
                    raise ValueError(f"{arm}: incomplete or nonfinite prediction")
                error = (prediction - truth).double()
                sums[arm]["squared"] += error.square().sum().item()
                sums[arm]["absolute"] += error.abs().sum().item()
                sums[arm]["elements"] += error.numel()
                sums[arm]["samples"] += x.shape[0]
                predictions[arm] = prediction
            reference = predictions["bcif"]
            for arm in comparison_arms:
                difference = (predictions[arm] - reference).double()
                diff = diffs[arm]
                diff["sum_squares"] += difference.square().sum().item()
                diff["elements"] += difference.numel()
                per_sample = difference.abs().flatten(start_dim=1).amax(dim=1)
                peak, index = per_sample.max(dim=0)
                if peak.item() > diff["max_abs"]:
                    diff["max_abs"] = peak.item()
                    diff["worst_target"] = target_indices[pointer + index.item()]
            pointer += x.shape[0]
    if pointer != len(dataset) or any(value["samples"] != pointer or not value["elements"]
                                      for value in sums.values()):
        raise ValueError("incomplete full validation replay")
    validation = {arm: {"mse": value["squared"] / value["elements"],
                        "mae": value["absolute"] / value["elements"],
                        "samples": value["samples"], "parameters": counts[arm]}
                  for arm, value in sums.items()}
    original = record["validation"]
    baseline_reproduction = {
        metric: _relative_change(validation["bcif"][metric], original[metric])
        for metric in ("mse", "mae")
    }
    comparisons = {}
    for arm, value in diffs.items():
        metric_changes = {metric: _relative_change(validation[arm][metric],
                                                   validation["bcif"][metric])
                          for metric in ("mse", "mae")}
        comparisons[arm] = {
            "prediction_max_abs": value["max_abs"],
            "prediction_rms": math.sqrt(value["sum_squares"] / value["elements"]),
            "worst_target": value["worst_target"],
            "metric_relative_change": metric_changes,
            "passed": (value["max_abs"] <= PREDICTION_MAX_ABSOLUTE_DIFFERENCE
                       and all(v <= MSE_MAE_MAX_RELATIVE_DIFFERENCE
                               for v in metric_changes.values())),
        }
    report = {
        "status": ("VALIDATED_EQUIVALENT" if all(comparison["passed"] for comparison
                   in comparisons.values()) and all(value <= MSE_MAE_MAX_RELATIVE_DIFFERENCE
                   for value in baseline_reproduction.values()) else "FAILED_EQUIVALENCE"),
        "evaluation_scope": "validation_only", "test": None,
        "task": record["task"], "seed": record["seed"],
        "reference_record": str(old_result), "checkpoint": str(checkpoint),
        "verified_sha256": hashes,
        "comparison_source_tree_sha256": comparison_source_sha256,
        "recorded_bcif_validation": original, "validation": validation,
        "output_fold_parameters_saved": counts["bcif"] - counts["output_fold"],
        "baseline_reproduction_relative_change": baseline_reproduction,
        "comparison_to_bcif": comparisons,
        "thresholds": {"prediction_max_abs": PREDICTION_MAX_ABSOLUTE_DIFFERENCE,
                       "metric_relative": MSE_MAE_MAX_RELATIVE_DIFFERENCE},
    }
    # Catch a modification of the locked inputs during the replay.
    if (sha256(checkpoint) != hashes["checkpoint"]
            or sha256(Path(args.root_path) / args.data_path) != hashes["data"]
            or original_source_digest(original_root) != hashes["source"]
            or source_tree_digest(new_root) != comparison_source_sha256):
        raise ValueError("reference checkpoint, data, or source changed during replay")
    from tools.sa_lcm.run_controlled import _q_identity

    if _q_identity(vars(args), vars(args))["sha256"] != hashes["q"]:
        raise ValueError("Q matrices changed during replay")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"{record['task']} seed={record['seed']} {report['status']}")
    for arm, metrics in validation.items():
        comparison = comparisons.get(arm)
        suffix = ("" if comparison is None else
                  f" max|Δ预测|={comparison['prediction_max_abs']:.8g}"
                  f" RMS={comparison['prediction_rms']:.8g}"
                  f" 通过={comparison['passed']}")
        print(f"  {arm:13} val MSE={metrics['mse']:.9f}"
              f" MAE={metrics['mae']:.9f} samples={metrics['samples']}{suffix}")
    print(f"验证报告：{output}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-result", type=Path, required=True,
                        help="original bcif/validation/result.json")
    parser.add_argument("--checkpoint", type=Path,
                        help="override checkpoint path after moving the original runs")
    parser.add_argument("--reference-source-root", type=Path, required=True,
                        help="the unchanged original source tree that produced the old record")
    parser.add_argument("--output", type=Path, required=True,
                        help="new, previously nonexistent JSON report")
    args = parser.parse_args(argv)
    report = replay(args.old_result, args.reference_source_root, args.output, args.checkpoint)
    return 0 if report["status"] == "VALIDATED_EQUIVALENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
