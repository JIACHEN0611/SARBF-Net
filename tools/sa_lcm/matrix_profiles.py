"""Locked profiles and input preflight for the paper SA-LCM matrix."""

import argparse
from copy import deepcopy
from pathlib import Path


LONG_HORIZONS = (96, 192, 336, 720)
PEMS_HORIZONS = (12, 24, 48, 96)
HORIZONS = LONG_HORIZONS
LONG_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL",
                 "Exchange", "Traffic", "Weather", "Solar")
PEMS_DATASETS = ("PEMS03", "PEMS04", "PEMS07", "PEMS08")
DATASETS = LONG_DATASETS + PEMS_DATASETS
MATRIX_TASKS = tuple(
    [f"{dataset}-{horizon}" for dataset in LONG_DATASETS
     for horizon in LONG_HORIZONS]
    + [f"{dataset}-{horizon}" for dataset in PEMS_DATASETS
       for horizon in PEMS_HORIZONS]
)


_BASE = {
    "ETTh1": dict(dataset="ETTh1", data="ETTh1", data_path="ETTh1.csv",
                  root="dataset/ETT-small", q_prefix="ETTh1", ratio="0.6",
                  enc_in=7, e_layers=2, patience=8, lradj="type1"),
    "ETTh2": dict(dataset="ETTh2", data="ETTh2", data_path="ETTh2.csv",
                  root="dataset/ETT-small", q_prefix="ETTh2", ratio="0.6",
                  enc_in=7, e_layers=3, patience=8, lradj="type1"),
    "ETTm1": dict(dataset="ETTm1", data="ETTm1", data_path="ETTm1.csv",
                  root="dataset/ETT-small", q_prefix="ETTm1", ratio="0.6",
                  enc_in=7, e_layers=2, patience=8, lradj="type1"),
    # Two layers keep the inter-layer BCIF path active, matching the agreed model setup.
    "ETTm2": dict(dataset="ETTm2", data="ETTm2", data_path="ETTm2.csv",
                  root="dataset/ETT-small", q_prefix="ETTm2", ratio="0.6",
                  enc_in=7, e_layers=2, patience=8, lradj="type1"),
    "ECL": dict(dataset="ECL", data="custom", data_path="electricity.csv",
                root="dataset/electricity", q_prefix="electricity", ratio="0.7",
                enc_in=321, e_layers=3, patience=5, lradj="cosine"),
    "Exchange": dict(dataset="Exchange", data="custom",
                     data_path="exchange_rate.csv", root="dataset/exchange_rate",
                     q_prefix="exchange_rate", ratio="0.7", enc_in=8,
                     e_layers=2, patience=8, lradj="type1"),
    "Traffic": dict(dataset="Traffic", data="custom", data_path="traffic.csv",
                    root="dataset/traffic", q_prefix="traffic", ratio="0.7",
                    enc_in=862, e_layers=3, patience=5, lradj="cosine"),
    "Weather": dict(dataset="Weather", data="custom", data_path="weather.csv",
                    root="dataset/weather", q_prefix="weather", ratio="0.7",
                    enc_in=21, e_layers=2, patience=5, lradj="type3"),
    "Solar": dict(dataset="Solar", data="Solar", data_path="solar_AL.txt",
                  root="dataset/Solar", q_prefix="solar_AL", ratio="0.7",
                  enc_in=137, e_layers=2, patience=5, lradj="type1"),
    "PEMS03": dict(dataset="PEMS03", data="PEMS", data_path="PEMS03.npz",
                   root="dataset/PEMS", q_prefix="PEMS03", ratio="0.6",
                   enc_in=358, e_layers=3, patience=10, lradj="cosine"),
    "PEMS04": dict(dataset="PEMS04", data="PEMS", data_path="PEMS04.npz",
                   root="dataset/PEMS", q_prefix="PEMS04", ratio="0.6",
                   enc_in=307, e_layers=3, patience=10, lradj="cosine"),
    "PEMS07": dict(dataset="PEMS07", data="PEMS", data_path="PEMS07.npz",
                   root="dataset/PEMS", q_prefix="PEMS07", ratio="0.6",
                   q_suffix="",
                   enc_in=883, e_layers=3, patience=10, lradj="cosine"),
    "PEMS08": dict(dataset="PEMS08", data="PEMS", data_path="PEMS08.npz",
                   root="dataset/PEMS", q_prefix="PEMS08", ratio="0.6",
                   enc_in=170, e_layers=3, patience=10, lradj="cosine"),
}


def build_profile(task):
    """Return an independent, fully resolved profile for one registered task."""
    if task not in MATRIX_TASKS:
        raise ValueError(f"unsupported matrix task: {task}")
    dataset, horizon_text = task.rsplit("-", 1)
    horizon = int(horizon_text)
    profile = deepcopy(_BASE[dataset])
    profile["pred_len"] = horizon
    ratio_suffix = profile.get("q_suffix", f"_ratio{profile['ratio']}")
    profile["q_mat_file"] = f"{profile['q_prefix']}_96{ratio_suffix}.npy"
    profile["q_out_mat_file"] = (
        f"{profile['q_prefix']}_{horizon}{ratio_suffix}.npy"
    )
    profile.update(d_model=512, d_ff=512, batch_size=32,
                   test_batch_size=16)

    if dataset == "ETTh1":
        profile.update(learning_rate=1e-4 if horizon == 720 else 5e-4,
                       train_epochs=30, dropout=0.2)
    elif dataset == "ETTh2":
        profile.update(learning_rate=2e-4, train_epochs=30,
                       dropout=0.3 if horizon == 720 else 0.2)
    elif dataset == "ETTm1":
        profile.update(learning_rate=1e-4, train_epochs=30,
                       dropout=0.2 if horizon == 720 else 0.1)
    elif dataset == "ETTm2":
        profile.update(learning_rate=1e-4, train_epochs=30,
                       dropout=0.3 if horizon == 720 else 0.2)
    elif dataset == "ECL":
        profile.update(learning_rate=5e-4,
                       train_epochs=30 if horizon == 96 else 50, dropout=0.0)
    elif dataset == "Exchange":
        widths = {96: 256, 192: 128, 336: 256, 720: 256}
        batches = {96: 64, 192: 32, 336: 64, 720: 32}
        profile.update(learning_rate=1e-4, train_epochs=1, dropout=0.0,
                       d_model=widths[horizon], d_ff=widths[horizon],
                       batch_size=batches[horizon])
    elif dataset == "Traffic":
        profile.update(learning_rate=5e-4, train_epochs=50, dropout=0.0)
    elif dataset == "Weather":
        profile.update(learning_rate=1e-3, train_epochs=30, dropout=0.0)
    elif dataset == "Solar":
        profile.update(learning_rate=5e-4,
                       train_epochs=40 if horizon in (96, 336) else 30,
                       dropout=0.0)
    elif dataset in PEMS_DATASETS:
        profile.update(learning_rate=5e-4, train_epochs=40, dropout=0.0)
    return profile


def all_profiles():
    return {task: build_profile(task) for task in MATRIX_TASKS}


def required_inputs(repository_root, tasks=MATRIX_TASKS):
    """Return the deduplicated data/input-Q/output-Q files for ``tasks``."""
    repository_root = Path(repository_root).resolve()
    paths = set()
    for task in tasks:
        profile = build_profile(task)
        root = repository_root / profile["root"]
        paths.update((root / profile["data_path"],
                      root / profile["q_mat_file"],
                      root / profile["q_out_mat_file"]))
    return tuple(sorted(paths))


def missing_inputs(repository_root, tasks=MATRIX_TASKS):
    """Return required inputs that are not regular files."""
    return tuple(path for path in required_inputs(repository_root, tasks)
                 if not path.is_file())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list-tasks", action="store_true")
    action.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    if args.list_tasks:
        print("\n".join(MATRIX_TASKS))
        return 0
    missing = missing_inputs(args.repository_root)
    if missing:
        print(f"预检失败：缺少 {len(missing)} 个数据或 Q 文件：")
        for path in missing:
            print(f"[缺失] {path}")
        return 1
    print(f"预检通过：{len(MATRIX_TASKS)} 个任务的输入文件完整。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
