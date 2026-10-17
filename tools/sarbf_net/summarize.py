"""Aggregate finite COMPLETE test results; reject duplicate task/arm/seed."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def summarize(run_root):
    groups, seen = {}, set()
    for path in sorted(Path(run_root).rglob("result.json")):
        r = json.loads(path.read_text())
        if r.get("status") != "COMPLETE":
            continue
        task = r.get("task") or r.get("dataset_task")
        arm = r["arm"]
        seed = int(r["seed"])
        mse = float(r["test"]["mse"])
        mae = float(r["test"]["mae"])
        if not math.isfinite(mse) or not math.isfinite(mae):
            raise ValueError(f"non-finite metrics: {path}")
        identity = (task, arm, seed)
        if identity in seen:
            raise ValueError(f"duplicate task/arm/seed: {identity}")
        seen.add(identity)
        groups.setdefault((task, arm), []).append((seed, mse, mae))
    rows = []
    for (task, arm), values in sorted(groups.items()):
        values.sort()
        row = dict(
            task=task, arm=arm, n=len(values), seeds=",".join(str(v[0]) for v in values)
        )
        for name, index in [("mse", 1), ("mae", 2)]:
            series = [v[index] for v in values]
            row[name + "_mean"] = statistics.mean(series)
            row[name + "_std"] = statistics.stdev(series) if len(series) > 1 else ""
        rows.append(row)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, default=Path("runs/sarbf_net"))
    p.add_argument("--output", type=Path, default=Path("summary.csv"))
    a = p.parse_args(argv)
    rows = summarize(a.run_root)
    if not rows:
        p.exit(1, "No COMPLETE test results; no summary was written.\n")
    with a.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"{len(rows)} configurations -> {a.output}; std is sample standard deviation (ddof=1)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
