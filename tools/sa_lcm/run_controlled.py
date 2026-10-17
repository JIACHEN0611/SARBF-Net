"""Run one fail-closed SA-LCM V1 probe task without retraining references."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from data_provider.long_context_factory import (
    long_context_data_provider,
    target_indices_for_dataset,
)
from tools.sa_lcm.matrix_profiles import MATRIX_TASKS, all_profiles


PROBE_TASKS = MATRIX_TASKS
ALLOWED_DIFFERENCES = frozenset(
    {"model", "model_id", "des", "checkpoints", "run_dir", "result_path"}
)
REQUIRED_LOG_ARGUMENTS = (
    "model",
    "model_id",
    "data",
    "seq_len",
    "long_len",
    "pred_len",
    "enc_in",
    "learning_rate",
    "batch_size",
    "train_epochs",
    "patience",
    "q_mat_file",
    "q_out_mat_file",
    "run_seed",
)
REQUIRED_DIAGNOSTICS = (
    "diagnostics_scope",
    "samples",
    "batches",
    "null_weight_mean",
    "null_weight_p50",
    "null_weight_p90",
    "history_mass_mean",
    "history_mass_p10",
    "history_mass_p50",
    "history_mass_p90",
    "history_block_weight_mean",
    "attention_entropy_mean",
    "cosine_max_mean",
    "cosine_mean",
    "bounded_correction_rms",
    "bounded_correction_max",
    "effective_refinement_ratio",
    "fraction_history_mass_lt_0_1",
    "fraction_history_mass_gt_0_5",
)


_PROFILES = all_profiles()


def build_configs(task: str, run_root: Path | str, repository_root: Path | str):
    if task not in _PROFILES:
        raise ValueError(f"unsupported SA-LCM probe task: {task}")
    profile = dict(_PROFILES[task])
    repository_root = Path(repository_root).resolve()
    run_root = Path(run_root).resolve()
    pred_len = profile["pred_len"]
    ratio = profile["ratio"]
    common = {
        "task": task,
        "is_training": 1,
        "root_path": str((repository_root / profile["root"]).resolve()),
        "data_path": profile["data_path"],
        "data": profile["data"],
        "features": "M",
        "seq_len": 96,
        "long_len": 768,
        "pred_len": pred_len,
        "enc_in": profile["enc_in"],
        "dec_in": profile["enc_in"],
        "c_out": profile["enc_in"],
        "q_mat_file": profile["q_mat_file"],
        "q_out_mat_file": profile["q_out_mat_file"],
        "q_channel_file":
            f"{profile['dataset']}_COV_channel_ratio{float(ratio):.2f}.npy",
        "embed_size": 16,
        "d_model": profile["d_model"],
        "d_ff": profile["d_ff"],
        "batch_size": profile["batch_size"],
        "test_batch_size": profile["test_batch_size"],
        "learning_rate": profile["learning_rate"],
        "train_epochs": profile["train_epochs"],
        "patience": profile["patience"],
        "dropout": profile["dropout"],
        "e_layers": profile["e_layers"],
        "lradj": profile["lradj"],
        "itr": 1,
        "lossfun_alpha": 0.5,
        "loss_mode": "L1",
        "run_seed": 2021,
        "fix_seed": 1,
        "test_mode": 0,
        "CKA_flag": 0,
        "resume_training": 0,
        "resume_epoch": 0,
        "save_every_epoch": 0,
        "use_revin": 1,
        "use_norm": 1,
        "send_mail": 0,
        "save_pdf": 0,
        "train_ratio": 1.0,
        "plot_mat_flag": 0,
        "use_gpu": 1,
    }

    def arm(model, label):
        run_dir = run_root / task / label
        config = dict(common)
        config.update(
            model=model,
            model_id=f"{task}_{model}_SA_LCM_probe8_seed2021",
            des=f"SA_LCM_probe8_{label}_seed2021",
            checkpoints=str((run_dir / "checkpoints").resolve()),
            run_dir=str(run_dir.resolve()),
            result_path=str((run_dir / "result.json").resolve()),
        )
        return config

    return arm("OLinear_LongContextControl", "reference"), arm("OLinear_SALCM", "sa_lcm")


def validate_protocol(reference, sa):
    differences = {
        key for key in set(reference) | set(sa) if reference.get(key) != sa.get(key)
    }
    if differences != ALLOWED_DIFFERENCES:
        raise ValueError(f"protocol differences do not match allowlist: {sorted(differences)}")
    if sa["model"] != "OLinear_SALCM" or sa["run_seed"] != 2021:
        raise ValueError("SA-LCM probe model/seed mismatch")


def _parse_value(text):
    text = text.strip().rstrip(",)").strip()
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_full_args(log_text):
    best = {}
    pattern = re.compile(r"(?:^|[(,]\s*)([A-Za-z_]\w*):\s*")
    for line in log_text.splitlines():
        matches = list(pattern.finditer(line))
        parsed = {}
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
            parsed[match.group(1)] = _parse_value(line[match.end():end])
        if len(parsed) > len(best):
            best = parsed
    missing = [name for name in REQUIRED_LOG_ARGUMENTS if name not in best]
    if missing:
        raise ValueError(f"args snapshot missing required fields: {missing}")
    return best


def _metrics_complete(value):
    return isinstance(value, Mapping) and all(
        name in value and math.isfinite(float(value[name])) for name in ("mse", "mae")
    )


def assess_completion(*, return_code, full_args, best_epoch, validation,
                      test_metrics, checkpoint, sample_manifest, q_identity,
                      diagnostics, implementation_revision):
    if return_code != 0:
        return {"status": "FAILED", "reason": "run.py exited nonzero"}
    try:
        if not isinstance(full_args, Mapping):
            raise ValueError("full args snapshot missing")
        missing = [name for name in REQUIRED_LOG_ARGUMENTS if name not in full_args]
        if missing:
            raise ValueError(f"full args snapshot missing required fields: {missing}")
        if not isinstance(best_epoch, int):
            raise ValueError("best epoch missing")
        if not _metrics_complete(validation) or not _metrics_complete(test_metrics):
            raise ValueError("validation/test MSE/MAE missing")
        if not isinstance(checkpoint, Mapping) or not checkpoint.get("path") or not checkpoint.get("sha256"):
            raise ValueError("selected checkpoint path/hash missing")
        for split in ("train", "valid", "test"):
            record = sample_manifest.get(split) if isinstance(sample_manifest, Mapping) else None
            if not isinstance(record, Mapping) or record.get("count", 0) <= 0 or not record.get("digest"):
                raise ValueError(f"sample manifest incomplete for {split}")
        if not isinstance(q_identity, Mapping) or q_identity.get("paths_equal") is not True:
            raise ValueError("Q identity missing or unequal")
        if not all((q_identity.get("sha256") or {}).values()):
            raise ValueError("Q hashes missing")
        if not isinstance(diagnostics, Mapping) or any(k not in diagnostics for k in REQUIRED_DIAGNOSTICS):
            raise ValueError("full-validation SA diagnostics missing")
        if implementation_revision != "SA_LCM_V1":
            raise ValueError("implementation revision missing")
    except (AttributeError, TypeError, ValueError) as error:
        return {"status": "MALFORMED", "reason": str(error)}
    return {"status": "COMPLETE", "reason": None}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_manifest(config):
    args = SimpleNamespace(
        task_name="forecasting", data=config["data"], root_path=config["root_path"],
        data_path=config["data_path"], features="M", target="OT", freq="h",
        embed="timeF", seq_len=96, long_len=768, label_len=0,
        pred_len=config["pred_len"], batch_size=config["batch_size"],
        test_batch_size=config["test_batch_size"], num_workers=0,
        train_ratio=1.0, train_part_first=0.0,
    )
    result = {}
    for split in ("train", "valid", "test"):
        dataset, _ = long_context_data_provider(args, split)
        indices = [int(value) for value in target_indices_for_dataset(dataset)]
        result[split] = {
            "count": len(indices),
            "first_target": indices[0] if indices else None,
            "last_target": indices[-1] if indices else None,
            "digest": hashlib.sha256(
                json.dumps(indices, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    return result


def _q_identity(reference, sa):
    records = {}
    paths_equal = True
    for key in ("q_mat_file", "q_out_mat_file"):
        ref = (Path(reference["root_path"]) / reference[key]).resolve()
        actual = (Path(sa["root_path"]) / sa[key]).resolve()
        paths_equal &= ref == actual
        records[key] = actual
    return {
        "paths_equal": bool(paths_equal),
        "files": {key: str(path) for key, path in records.items()},
        "sha256": {key: _sha256(path) if path.is_file() else None for key, path in records.items()},
        "scope": "identity_only_not_train_boundary_proof",
    }


def _command(config, repository_root):
    excluded = {"task", "run_dir", "result_path"}
    command = [sys.executable, "-u", str(repository_root / "run.py")]
    for key, value in sorted(config.items()):
        if key in excluded:
            continue
        command.extend([f"--{key}", str(value)])
    return command


def _parse_best_epoch(text):
    rows = re.findall(r"Epoch:\s*(\d+), Steps:.*?Vali Loss:\s*([0-9.eE+-]+)", text)
    return int(min(rows, key=lambda row: float(row[1]))[0]) if rows else None


def _parse_test_metrics(text):
    rows = re.findall(r"final output mse:\s*([0-9.eE+-]+),\s*mae:\s*([0-9.eE+-]+)", text)
    if not rows:
        return None
    mse, mae = rows[-1]
    return {"mse": float(mse), "mae": float(mae)}


def _evaluate_validation(full_args, checkpoint):
    from experiments.exp_forecast_long_context import Exp_Forecast_LongContext

    args = SimpleNamespace(**dict(full_args))
    experiment = Exp_Forecast_LongContext(args)
    state = torch.load(checkpoint, map_location=experiment.device)
    experiment.model.load_state_dict(state)
    _, loader = experiment._get_data(flag="valid")
    module = experiment.model.module if hasattr(experiment.model, "module") else experiment.model
    module.sa_lcm.reset_diagnostics()
    squared = absolute = 0.0
    elements = samples = 0
    try:
        experiment.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                batch_x = batch_x.float().to(experiment.device)
                batch_y = batch_y.float().to(experiment.device)
                output = experiment.model(
                    batch_x,
                    batch_x_mark.float().to(experiment.device),
                    batch_y,
                    batch_y_mark.float().to(experiment.device),
                )
                if isinstance(output, tuple):
                    output = output[0]
                first = -1 if args.features == "MS" else 0
                output = output[:, -args.pred_len:, first:]
                target = batch_y[:, -args.pred_len:, first:]
                error = output - target
                if not torch.isfinite(error).all().item():
                    raise ValueError("non-finite validation prediction")
                squared += float(error.double().square().sum())
                absolute += float(error.double().abs().sum())
                elements += error.numel()
                samples += error.shape[0]
        expected = len(loader.dataset)
        if samples != expected or not elements:
            raise RuntimeError(f"validation sample count mismatch: {samples} != {expected}")
        diagnostics = module.sa_lcm.finalize_diagnostics(expected)
        return {
            "metrics": {"mse": squared / elements, "mae": absolute / elements, "samples": samples},
            "diagnostics": diagnostics,
            "total_params": sum(p.numel() for p in module.parameters()),
            "sa_lcm_added_params": sum(p.numel() for p in module.sa_lcm.parameters()),
        }
    finally:
        module.sa_lcm.cancel_diagnostics()


def run_one(config, reference, repository_root):
    repository_root = Path(repository_root).resolve()
    run_dir = Path(config["run_dir"])
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = Path(config["result_path"])
    base = {
        "implementation_revision": "SA_LCM_V1",
        "dataset_task": config["task"],
        "arm": "sa_lcm",
        "checkpoint_selection_metric": "validation_weighted_l1",
        "architecture_selection_metric": "preregistered_probe_gate",
    }
    try:
        manifest = _target_manifest(config)
        q_identity = _q_identity(reference, config)
        if not q_identity["paths_equal"] or not all(q_identity["sha256"].values()):
            raise RuntimeError("Q path/hash identity failed")
    except Exception as error:
        base.update(status="MALFORMED", reason=str(error), return_code=None)
        result_path.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n")
        return base

    train_log = (run_dir / "train.log").resolve()
    run_environment = {
        **os.environ,
        "PYTHONPATH": str(repository_root),
        "PYTHONUNBUFFERED": "1",
    }
    started = time.monotonic()
    with train_log.open("w", encoding="utf-8") as log_handle:
        print(f"RUNNING task={config['task']} arm=sa_lcm", flush=True)
        print(f"TRAIN_LOG={train_log}", flush=True)
        completed = subprocess.run(
            _command(config, repository_root), cwd=run_dir,
            env=run_environment,
            stdout=log_handle, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
    runtime = time.monotonic() - started
    completed_output = train_log.read_text(encoding="utf-8", errors="replace")
    full_args = validation = test_metrics = checkpoint_record = diagnostics = None
    best_epoch = total_params = added_params = None
    if completed.returncode == 0:
        try:
            full_args = parse_full_args(completed_output)
            checkpoints = list((run_dir / "checkpoints").rglob("checkpoint.pth"))
            if len(checkpoints) != 1:
                raise RuntimeError(f"expected exactly one checkpoint, found {len(checkpoints)}")
            checkpoint = checkpoints[0].resolve()
            checkpoint_record = {"path": str(checkpoint), "sha256": _sha256(checkpoint)}
            evaluated = _evaluate_validation(full_args, checkpoint)
            validation = evaluated["metrics"]
            diagnostics = evaluated["diagnostics"]
            total_params = evaluated["total_params"]
            added_params = evaluated["sa_lcm_added_params"]
            test_metrics = _parse_test_metrics(completed_output)
            best_epoch = _parse_best_epoch(completed_output)
        except Exception as error:
            base["artifact_error"] = str(error)
    assessment = assess_completion(
        return_code=completed.returncode, full_args=full_args, best_epoch=best_epoch,
        validation=validation, test_metrics=test_metrics, checkpoint=checkpoint_record,
        sample_manifest=manifest, q_identity=q_identity, diagnostics=diagnostics,
        implementation_revision=base["implementation_revision"],
    )
    base.update(assessment)
    base.update(
        return_code=completed.returncode, runtime_seconds=runtime, args=full_args,
        best_epoch=best_epoch, validation=validation, test=test_metrics,
        checkpoint=checkpoint_record, sample_manifest=manifest, q_identity=q_identity,
        diagnostics=diagnostics, total_params=total_params,
        sa_lcm_added_params=added_params,
    )
    result_path.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n")
    return base


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=PROBE_TASKS, required=True)
    parser.add_argument("--run-root", default="runs/sa_lcm/probe8_seed2021")
    parser.add_argument("--repository-root", default=".")
    parser.parse_args(argv)
    parser.error("legacy baseline probe is unavailable in this slim release; "
                 "use tools.sa_lcm_transport.run_pilot with --arm no_diag")


if __name__ == "__main__":
    raise SystemExit(main())
