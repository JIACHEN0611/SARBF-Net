"""Write detail, paired-seed, and three-seed aggregate result tables."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from tools.saer_transfer.run import is_complete


DETAIL_COLUMNS = [
    "Backbone", "Dataset", "InputLength", "RawContextLength", "Horizon",
    "Seed", "Arm", "MSE", "MAE", "Params", "BestEpoch",
]
PAIR_COLUMNS = [
    "Backbone", "Dataset", "InputLength", "Horizon", "Seed",
    "OriginalMSE", "SAERMSE", "MSEImprovementPct", "OriginalMAE",
    "SAERMAE", "MAEImprovementPct", "BothImproved",
]
AGGREGATE_COLUMNS = [
    "Backbone", "Dataset", "InputLength", "Horizon", "Seeds",
    "OriginalMSEMean", "OriginalMSEStd", "SAERMSEMean", "SAERMSEStd",
    "MSEImprovementPct", "MSEWins", "OriginalMAEMean", "OriginalMAEStd",
    "SAERMAEMean", "SAERMAEStd", "MAEImprovementPct", "MAEWins",
    "BothImprovedSeeds",
]


def collect_rows(run_root):
    rows = []
    seen = set()
    for task_path in sorted(Path(run_root).glob("**/task.json")):
        if not is_complete(task_path.parent):
            continue
        task = json.loads(task_path.read_text(encoding="utf-8"))
        metrics = json.loads(task_path.with_name("metrics.json").read_text(encoding="utf-8"))
        key = tuple(task[name] for name in ("backbone", "dataset", "seq_len", "horizon", "seed", "arm"))
        if key in seen:
            raise ValueError(f"duplicate experiment result: {key}")
        seen.add(key)
        rows.append({
            "Backbone": task["backbone"], "Dataset": task["dataset"],
            "InputLength": task["seq_len"], "RawContextLength": task["long_len"],
            "Horizon": task["horizon"], "Seed": task["seed"], "Arm": task["arm"],
            "MSE": metrics["mse"], "MAE": metrics["mae"],
            "Params": metrics["params"], "BestEpoch": metrics["best_epoch"],
        })
    return rows


def _improvement_pct(original, saer):
    return round((original - saer) / original * 100, 6) if original != 0 else None


def pair_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["Backbone"], row["Dataset"], row["InputLength"], row["Horizon"], row["Seed"])
        grouped.setdefault(key, {})[row["Arm"]] = row
    pairs = []
    for key, arms in sorted(grouped.items()):
        if not {"original", "sa_frontend"} <= set(arms):
            continue
        original, saer = arms["original"], arms["sa_frontend"]
        mse_gain = _improvement_pct(original["MSE"], saer["MSE"])
        mae_gain = _improvement_pct(original["MAE"], saer["MAE"])
        pairs.append({
            "Backbone": key[0], "Dataset": key[1], "InputLength": key[2],
            "Horizon": key[3], "Seed": key[4],
            "OriginalMSE": original["MSE"], "SAERMSE": saer["MSE"],
            "MSEImprovementPct": mse_gain,
            "OriginalMAE": original["MAE"], "SAERMAE": saer["MAE"],
            "MAEImprovementPct": mae_gain,
            "BothImproved": saer["MSE"] < original["MSE"] and saer["MAE"] < original["MAE"],
        })
    return pairs


def aggregate_rows(pairs):
    grouped = {}
    for row in pairs:
        key = (row["Backbone"], row["Dataset"], row["InputLength"], row["Horizon"])
        grouped.setdefault(key, []).append(row)
    aggregates = []
    for key, group in sorted(grouped.items()):
        original_mse = [row["OriginalMSE"] for row in group]
        saer_mse = [row["SAERMSE"] for row in group]
        original_mae = [row["OriginalMAE"] for row in group]
        saer_mae = [row["SAERMAE"] for row in group]
        om, sm, oa, sa = map(mean, (original_mse, saer_mse, original_mae, saer_mae))
        aggregates.append({
            "Backbone": key[0], "Dataset": key[1], "InputLength": key[2], "Horizon": key[3],
            "Seeds": len(group),
            "OriginalMSEMean": om, "OriginalMSEStd": stdev(original_mse) if len(group) > 1 else None,
            "SAERMSEMean": sm, "SAERMSEStd": stdev(saer_mse) if len(group) > 1 else None,
            "MSEImprovementPct": _improvement_pct(om, sm),
            "MSEWins": sum(row["SAERMSE"] < row["OriginalMSE"] for row in group),
            "OriginalMAEMean": oa, "OriginalMAEStd": stdev(original_mae) if len(group) > 1 else None,
            "SAERMAEMean": sa, "SAERMAEStd": stdev(saer_mae) if len(group) > 1 else None,
            "MAEImprovementPct": _improvement_pct(oa, sa),
            "MAEWins": sum(row["SAERMAE"] < row["OriginalMAE"] for row in group),
            "BothImprovedSeeds": sum(bool(row["BothImproved"]) for row in group),
        })
    return aggregates


def _write_csv(path, columns, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_summaries(run_root, detail_output, paired_output, aggregate_output):
    rows = collect_rows(run_root)
    pairs = pair_rows(rows)
    aggregates = aggregate_rows(pairs)
    _write_csv(detail_output, DETAIL_COLUMNS, rows)
    _write_csv(paired_output, PAIR_COLUMNS, pairs)
    _write_csv(aggregate_output, AGGREGATE_COLUMNS, aggregates)
    return len(rows), len(pairs), len(aggregates)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        default="runs/saer_five_model_six_dataset_four_horizon_3seed_v1",
    )
    parser.add_argument("--output")
    parser.add_argument("--paired-output")
    parser.add_argument("--aggregate-output")
    args = parser.parse_args(argv)
    root = Path(args.run_root)
    counts = write_summaries(
        root,
        args.output or root / "summary.csv",
        args.paired_output or root / "paired_summary.csv",
        args.aggregate_output or root / "aggregate_summary.csv",
    )
    print(f"wrote {counts[0]} rows, {counts[1]} pairs, and {counts[2]} aggregates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
