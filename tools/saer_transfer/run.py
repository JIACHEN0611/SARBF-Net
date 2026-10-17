"""Serial and resumable launcher for the 720-task five-model matrix."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from tools.saer_transfer.profiles import build_tasks


REQUIRED_METRICS = {"mse", "mae", "params", "best_epoch"}


def task_directory(run_root, task):
    return (
        Path(run_root)
        / task.backbone
        / task.dataset
        / f"seed{task.seed}"
        / f"seq{task.seq_len}"
        / f"pred{task.horizon}"
        / task.arm
    )


def is_complete(task_dir):
    metrics_path = Path(task_dir) / "metrics.json"
    if (Path(task_dir) / "failed.json").exists() or not metrics_path.is_file():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(metrics, dict) or not REQUIRED_METRICS <= metrics.keys():
        return False
    return all(
        isinstance(metrics[key], (int, float))
        and not isinstance(metrics[key], bool)
        and math.isfinite(float(metrics[key]))
        for key in REQUIRED_METRICS
    )


def build_command(task, python_executable, data_root):
    dataset = task.dataset_profile
    model = task.backbone_profile
    command = [
        str(python_executable),
        "-u",
        "-m",
        "experiments.saer_transfer.train",
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model_id",
        task.task_id,
        "--model",
        "LongContextBackbone",
        "--backbone",
        task.backbone,
        "--arm",
        task.arm,
        "--data",
        dataset["data"],
        "--root_path",
        str(Path(data_root).resolve()),
        "--data_path",
        dataset["data_path"],
        "--features",
        "M",
        "--seq_len",
        str(task.seq_len),
        "--long_len",
        str(task.long_len),
        "--label_len",
        "0",
        "--pred_len",
        str(task.horizon),
        "--enc_in",
        str(dataset["enc_in"]),
        "--dec_in",
        str(dataset["enc_in"]),
        "--c_out",
        str(dataset["enc_in"]),
        "--cycle",
        str(dataset["cycle"]),
        "--random_seed",
        str(task.seed),
        "--itr",
        "1",
        "--use_norm",
        "1",
    ]
    options = (
        "batch_size",
        "learning_rate",
        "train_epochs",
        "patience",
        "lradj",
        "model_type",
        "pct_start",
        "d_model",
        "d_ff",
        "e_layers",
        "n_heads",
        "dropout",
        "fc_dropout",
        "head_dropout",
        "patch_len",
        "stride",
        "revin",
    )
    for key in options:
        if key in model:
            command.extend([f"--{key}", str(model[key])])
    return command


def _task_metadata(task, command):
    return {
        "task_id": task.task_id,
        "backbone": task.backbone,
        "dataset": task.dataset,
        "seq_len": task.seq_len,
        "long_len": task.long_len,
        "horizon": task.horizon,
        "seed": task.seed,
        "arm": task.arm,
        "command": command,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=Path('runs/saer_transfer'))
    parser.add_argument('--data-root', type=Path, default=Path('dataset_pilot'))
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--backbone', choices=['DLinear','PatchTST','iTransformer','CycleNet','TSMixer'])
    parser.add_argument('--dataset', choices=['ETTm1','Electricity','PEMS03','PEMS07','Solar','Weather'])
    parser.add_argument('--horizon', type=int, choices=[96,192,336,720])
    parser.add_argument('--seed', type=int, choices=[2021,2022,2023])
    parser.add_argument('--arm', choices=['original','sa_frontend'])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list-tasks', action='store_true')
    parser.add_argument('--keep-going', action='store_true')
    args = parser.parse_args(argv)
    tasks = [task for task in build_tasks() if all(
        getattr(args, key) is None or getattr(task, key) == getattr(args, key)
        for key in ('backbone','dataset','horizon','seed','arm'))]
    if args.list_tasks:
        for task in tasks:
            print(task.task_id)
        return 0
    repository_root = Path(__file__).resolve().parents[2]
    run_root, data_root = args.run_root.resolve(), args.data_root.resolve()
    # Make path-bearing interpreters absolute without dereferencing venv symlinks.
    # Bare commands such as "python" continue to use PATH.
    python_executable = args.python
    if os.path.dirname(python_executable):
        python_executable = str(Path(python_executable).expanduser().absolute())
    failures = 0
    for task in tasks:
        folder = task_directory(run_root, task)
        if is_complete(folder):
            print(f'SKIP {task.task_id}', flush=True)
            continue
        command = build_command(task, python_executable, data_root)
        command += ['--checkpoints', str(folder / 'checkpoints'), '--output_root', str(folder)]
        print('RUN ' + json.dumps(command), flush=True)
        if args.dry_run:
            continue
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f'Incomplete run directory exists: {folder}; select a new --run-root')
        data_file = data_root / task.dataset_profile['data_path']
        if not data_file.is_file():
            raise FileNotFoundError(f'Missing dataset: {data_file}')
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'task.json').write_text(json.dumps(_task_metadata(task, command), indent=2) + '\n')
        environment = os.environ.copy()
        environment['SA_FRONTEND_RESULT_JSON'] = str(folder / 'metrics.json')
        with (folder / 'stdout.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=repository_root, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0 or not is_complete(folder):
            failures += 1
            (folder / 'failed.json').write_text(json.dumps({'returncode':result.returncode}) + '\n')
            if not args.keep_going:
                return 1
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())


