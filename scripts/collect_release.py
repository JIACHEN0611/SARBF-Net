#!/usr/bin/env python3
"""Collect matched SARBF-Net assets and optionally source into one portable zip.

Uses only the Python standard library; never trains, deletes or changes source
experiments. Only COMPLETE records of the selected final model are eligible.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import shutil
import statistics
import sys
import tempfile
import zipfile
from pathlib import Path

REVISION = "SA_LCM_COMPACT_FINAL_V1"
MODELS = {"SA_LCM", "SARBF_Net"}
ARMS = {"sa_lcm", "sarbf_net", "pair_free_fold"}
DATA_ROOTS = {
    "ETTh1": "ETT-small",
    "ETTh2": "ETT-small",
    "ETTm1": "ETT-small",
    "ETTm2": "ETT-small",
    "ECL": "electricity",
    "Exchange": "exchange_rate",
    "Traffic": "traffic",
    "Weather": "weather",
    "Solar": "Solar",
    "PEMS03": "PEMS",
    "PEMS04": "PEMS",
    "PEMS07": "PEMS",
    "PEMS08": "PEMS",
    "PEMSD7": "PEMS",
    "PEMS-BAY": "PEMS",
}
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "dataset",
    "datasets",
    "artifacts",
    "checkpoints",
    "test_results",
    "predictions",
}
SOURCE_SKIP = SKIP_DIRS | {
    "results",
    "runs",
    "dataset_template",
    ".idea",
    ".vscode",
    ".codex",
    "replay_output",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def result_files(roots):
    seen = set()
    for root in roots:
        root = Path(root).resolve()
        if root.is_file():
            if root not in seen:
                seen.add(root)
                yield root
            continue
        for folder, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            if "result.json" in files:
                path = Path(folder) / "result.json"
                if path not in seen:
                    seen.add(path)
                    yield path


def locate(stored, expected, base, kind, roots, filename=None):
    if not valid_hash(expected):
        raise ValueError(f"{kind}: recorded SHA256 is missing/invalid")
    candidates = []
    if stored:
        p = Path(stored)
        candidates.append(p if p.is_absolute() else base / p)
    if kind == "checkpoint":
        candidates.extend(sorted((base / "checkpoints").rglob("checkpoint.pth")))
    else:
        basename = filename or (Path(stored).name if stored else "")
        if basename:
            candidates.append(base / basename)
    checked = set()
    for path in candidates:
        if path.is_file() and path.resolve() not in checked:
            checked.add(path.resolve())
            if sha256(path) == expected:
                return path.resolve()
    # Relocated Q arrays may be found by basename, but only the recorded hash
    # can select one. Checkpoints are never guessed elsewhere in the server.
    if kind != "checkpoint" and filename:
        for root in roots:
            root = Path(root)
            if not root.is_dir():
                continue
            for folder, dirs, files in os.walk(root, followlinks=False):
                dirs[:] = [
                    d
                    for d in dirs
                    if d not in {".git", ".venv", "venv", "__pycache__", "node_modules"}
                ]
                if filename in files:
                    path = Path(folder) / filename
                    if path.resolve() not in checked and path.is_file():
                        checked.add(path.resolve())
                        if sha256(path) == expected:
                            return path.resolve()
    raise ValueError(f"{kind}: no file matches recorded SHA256 {expected}: {stored}")


def study_variant(path, roots):
    for root in roots:
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        seed_index = next(
            (i for i, part in enumerate(parts) if re.fullmatch(r"seed[0-9]+", part)),
            None,
        )
        if seed_index is None:
            raise ValueError(
                "analysis result path needs a seed directory to identify its variant"
            )
        return "/".join(parts[:seed_index]) or "default"
    raise ValueError("analysis record is outside its selected root")


def scan(roots, dataset_roots=(), study="main", split_variants=False):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", study):
        raise ValueError("invalid study label")
    roots = [Path(p).resolve() for p in roots]
    report = dict(schema_version=1, runs=[], errors=[], skipped=[], duplicate_copies=[])
    unique = {}
    for path in result_files(roots):
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            report["skipped"].append(f"{path}: unreadable result JSON ({e})")
            continue
        if not isinstance(r, dict) or not isinstance(r.get("args"), dict):
            report["skipped"].append(f"{path}: unsupported JSON/args structure")
            continue
        args = r["args"]
        if (
            not isinstance(r.get("arm"), str)
            or not isinstance(args.get("model"), str)
            or r.get("status") != "COMPLETE"
            or r.get("arm") not in ARMS
            or args.get("model") not in MODELS
        ):
            report["skipped"].append(f"{path}: not a COMPLETE final-model record")
            continue
        try:
            variant = study_variant(path, roots) if split_variants else "standard"
            prefix = (
                "artifacts"
                if study == "main"
                else f"artifacts/studies/{study}/{variant}"
            )
            if r.get("architecture_revision") != REVISION:
                raise ValueError("unexpected architecture revision")
            if r.get("provenance_verified") is not True or r.get("return_code") != 0:
                raise ValueError("original provenance/return-code check did not pass")
            task = r["task"]
            dataset, horizon = task.rsplit("-", 1)
            horizon = int(horizon)
            seed = int(r["seed"])
            if dataset not in DATA_ROOTS or horizon not in (
                (12, 24, 48, 96) if dataset.startswith("PEMS") else (96, 192, 336, 720)
            ):
                raise ValueError("task outside the supported asset collection scope")
            if (
                args.get("seq_len") != 96
                or args.get("long_len") != 768
                or args.get("pred_len") != horizon
            ):
                raise ValueError("input/horizon protocol mismatch")
            for field in ["validation", "test", "checkpoint", "q_identity"]:
                if not isinstance(r.get(field), dict):
                    raise ValueError(f"invalid {field} structure")
            if args.get("run_seed", seed) != seed:
                raise ValueError("record seed and argument seed differ")
            for split in ["validation", "test"]:
                for metric in ["mse", "mae"]:
                    value = float(r[split][metric])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("non-finite or negative metric")
            for name in ["data_sha256", "source_tree_sha256"]:
                if not valid_hash(r.get(name)):
                    raise ValueError(f"missing {name}")
            checkpoint = r["checkpoint"]
            q = r["q_identity"]
            qfiles = q.get("files") or {}
            qhash = q["sha256"]
            weight = locate(
                checkpoint.get("path"),
                checkpoint.get("sha256"),
                path.parent,
                "checkpoint",
                roots,
            )
            matrices = {}
            for key in ["q_mat_file", "q_out_mat_file"]:
                name = Path(args[key]).name
                stored = qfiles.get(key) or str(Path(args["root_path"]) / args[key])
                matrices[key] = locate(
                    stored,
                    qhash.get(key),
                    path.parent,
                    key,
                    [*dataset_roots, *roots],
                    name,
                )
            raw_name = Path(args["data_path"])
            if raw_name.is_absolute() or ".." in raw_name.parts:
                raise ValueError("unsafe relative data path")
            public = dict(
                task=task,
                seed=seed,
                study=study,
                variant=variant,
                arm="sarbf_net",
                model="SARBF_Net",
                checkpoint=dict(
                    path=f"{prefix}/checkpoints/{task}/seed{seed}/checkpoint.pth",
                    sha256=checkpoint["sha256"],
                ),
                q_input=dict(
                    path=f"artifacts/q/{qhash['q_mat_file']}.npy",
                    sha256=qhash["q_mat_file"],
                ),
                q_output=dict(
                    path=f"artifacts/q/{qhash['q_out_mat_file']}.npy",
                    sha256=qhash["q_out_mat_file"],
                ),
                result_path=f"{prefix}/results/{task}/seed{seed}/result.json",
                data_relative_path=(Path(DATA_ROOTS[dataset]) / raw_name).as_posix(),
                data_sha256=r["data_sha256"],
                original_source_sha256=r["source_tree_sha256"],
            )
            relocatable = {
                "root_path",
                "checkpoints",
                "run_dir",
                "result_path",
                "folder_path",
                "q_mat_file",
                "q_out_mat_file",
                "q_channel_file",
                "Q_MAT_file",
                "Q_OUT_MAT_file",
                "model_id",
                "model_id_ori",
                "des",
                "model",
            }
            signature = json.dumps(
                {
                    "checkpoint": checkpoint["sha256"],
                    "q": qhash,
                    "data": r["data_sha256"],
                    "source": r["source_tree_sha256"],
                    "args": {k: v for k, v in args.items() if k not in relocatable},
                    "targets": r.get("sample_manifest"),
                    "best_epoch": r.get("best_epoch"),
                    "selection": r.get("checkpoint_selection_metric"),
                    "validation": r["validation"],
                    "test": r["test"],
                },
                sort_keys=True,
            )
            key = (study, variant, task, seed)
            if key in unique:
                if unique[key]["signature"] != signature:
                    raise ValueError(
                        f"ambiguous duplicate {task} seed{seed}; restrict --run-root to the intended experiment"
                    )
                report["duplicate_copies"].append(str(path))
                continue
            entry = dict(
                public=public,
                record=str(path.resolve()),
                record_sha256=sha256(path),
                checkpoint=str(weight),
                q_input=str(matrices["q_mat_file"]),
                q_output=str(matrices["q_out_mat_file"]),
                signature=signature,
                validation=r["validation"],
                test=r["test"],
            )
            unique[key] = entry
            report["runs"].append(entry)
        except (
            KeyError,
            ValueError,
            TypeError,
            AttributeError,
            OverflowError,
            OSError,
        ) as e:
            report["errors"].append(f"{path}: {e}")
    return finish_report(report)


def scan_groups(main_roots, study_roots=(), dataset_roots=()):
    report = scan(main_roots, dataset_roots)
    groups = {}
    for label, root in study_roots:
        if label == "main":
            raise ValueError("analysis study label cannot be main")
        groups.setdefault(label, []).append(root)
    for label, roots in groups.items():
        addition = scan(roots, dataset_roots, study=label, split_variants=True)
        for field in ("runs", "errors", "skipped", "duplicate_copies"):
            report[field].extend(addition[field])
    return finish_report(report)


def finish_report(report):
    report["runs"].sort(
        key=lambda e: (
            e["public"].get("study", "main") != "main",
            e["public"].get("study", "main"),
            e["public"].get("variant", "standard"),
            e["public"]["task"],
            e["public"]["seed"],
        )
    )
    report["study_counts"] = {}
    for entry in report["runs"]:
        public = entry["public"]
        label = public.get("study", "main") + "/" + public.get("variant", "standard")
        report["study_counts"][label] = report["study_counts"].get(label, 0) + 1
    report["run_count"] = len(report["runs"])
    blobs = {}
    for e in report["runs"]:
        for key in ["checkpoint", "q_input", "q_output"]:
            blobs[e["public"][key]["path"]] = e[key]
    report["asset_bytes"] = sum(Path(p).stat().st_size for p in blobs.values())
    report["asset_mib"] = round(report["asset_bytes"] / 1024**2, 2)
    report["large_files_over_25_mib"] = [
        name for name, p in blobs.items() if Path(p).stat().st_size > 25 * 1024**2
    ]
    report["large_files_over_100_mib"] = [
        name for name, p in blobs.items() if Path(p).stat().st_size > 100 * 1024**2
    ]
    return report


def summaries(runs):
    individual = io.StringIO()
    writer = csv.writer(individual)
    writer.writerow(
        [
            "study",
            "variant",
            "task",
            "seed",
            "validation_mse",
            "validation_mae",
            "test_mse",
            "test_mae",
        ]
    )
    groups = {}
    for entry in runs:
        r, validation, test = entry["public"], entry["validation"], entry["test"]
        identity = (r.get("study", "main"), r.get("variant", "standard"), r["task"])
        writer.writerow(
            [
                *identity,
                r["seed"],
                validation["mse"],
                validation["mae"],
                test["mse"],
                test["mae"],
            ]
        )
        groups.setdefault(identity, []).append(
            (r["seed"], float(test["mse"]), float(test["mae"]))
        )
    aggregate = io.StringIO()
    writer = csv.writer(aggregate)
    writer.writerow(
        [
            "study",
            "variant",
            "task",
            "n",
            "seeds",
            "mse_mean",
            "mse_std",
            "mae_mean",
            "mae_std",
        ]
    )
    for identity, rows in sorted(groups.items()):
        values = []
        for index in (1, 2):
            series = [row[index] for row in rows]
            values.extend(
                [
                    statistics.mean(series),
                    statistics.stdev(series) if len(series) > 1 else "",
                ]
            )
        writer.writerow(
            [*identity, len(rows), ",".join(str(row[0]) for row in rows), *values]
        )
    return individual.getvalue(), aggregate.getvalue()


def write_release(plan, output, source_root=None):
    output = Path(output).resolve()
    if plan["errors"]:
        raise ValueError(
            "collection errors; inspect the report and resolve them before packaging"
        )
    if not plan["runs"]:
        raise ValueError(
            "no verified COMPLETE final-model records found; inspect the report"
        )
    if output.exists():
        raise FileExistsError(output)
    required_space = int(plan.get("asset_bytes", 0) * 1.02) + 64 * 1024**2
    free_space = shutil.disk_usage(output.parent).free
    if free_space < required_space:
        raise OSError(
            f"insufficient free space: need about {required_space / 1024**2:.1f} MiB, available {free_space / 1024**2:.1f} MiB; choose another output disk"
        )
    files = {}
    manifest = dict(
        schema_version=1,
        model="SARBF-Net",
        architecture_revision=REVISION,
        raw_datasets_included=False,
        study_counts=plan.get("study_counts", {}),
        runs=[e["public"] for e in plan["runs"]],
        files=files,
    )
    fd, temporary = tempfile.mkstemp(
        prefix=".sarbf-assets-", suffix=".zip", dir=output.parent
    )
    os.close(fd)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as z:

            def add_file(path, relative, expected=None):
                path = Path(path)
                name = "SARBF-Net/" + relative
                if relative in files:
                    if expected and files[relative]["sha256"] != expected:
                        raise ValueError("asset destination conflict")
                    return
                digest = hashlib.sha256()
                size = 0
                with (
                    path.open("rb") as source,
                    z.open(name, "w", force_zip64=True) as dest,
                ):
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                        size += len(block)
                        dest.write(block)
                actual = digest.hexdigest()
                if expected and actual != expected:
                    raise ValueError(f"file changed during export: {path}")
                files[relative] = dict(sha256=actual, bytes=size)

            def add_text(relative, text):
                data = text.encode("utf-8")
                z.writestr("SARBF-Net/" + relative, data)
                files[relative] = dict(
                    sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)
                )

            if source_root:
                source_root = Path(source_root).resolve()
                for folder, dirs, names in os.walk(source_root, followlinks=False):
                    dirs[:] = sorted(
                        d
                        for d in dirs
                        if d not in SOURCE_SKIP and not (Path(folder) / d).is_symlink()
                    )
                    for name in sorted(names):
                        p = Path(folder) / name
                        if p.is_symlink() or not p.is_file() or p.resolve() == output:
                            continue
                        if (
                            p.relative_to(source_root).as_posix()
                            == "RELEASE_MANIFEST.json"
                        ):
                            continue
                        if p.name.startswith(".env"):
                            continue
                        if p.suffix not in {
                            ".py",
                            ".md",
                            ".txt",
                            ".json",
                            ".sh",
                            ".yml",
                            ".yaml",
                            ".toml",
                        } and p.name not in {".gitignore", ".gitattributes"}:
                            continue
                        if p.suffix in {
                            ".zip",
                            ".gz",
                            ".pth",
                            ".pt",
                            ".ckpt",
                            ".log",
                            ".pyc",
                        }:
                            continue
                        add_file(p, p.relative_to(source_root).as_posix())
                template = source_root / "dataset/README.md"
                if (
                    template.is_file()
                    and not template.is_symlink()
                    and template.resolve().is_relative_to(source_root)
                ):
                    add_file(template, "dataset/README.md")
                original_manifest = source_root / "RELEASE_MANIFEST.json"
                source_manifest = (
                    json.loads(original_manifest.read_text())
                    if original_manifest.is_file()
                    else {}
                )
                source_manifest["files"] = {
                    name: info["sha256"] for name, info in files.items()
                }
                source_manifest["source_snapshot_regenerated_for_bundle"] = True
                add_text(
                    "RELEASE_MANIFEST.json",
                    json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
                )
            for e in plan["runs"]:
                r = e["public"]
                add_file(e["record"], r["result_path"], e["record_sha256"])
                for key in ["checkpoint", "q_input", "q_output"]:
                    add_file(e[key], r[key]["path"], r[key]["sha256"])
            individual, aggregate = summaries(plan["runs"])
            add_text("artifacts/results/per_run.csv", individual)
            add_text("artifacts/results/summary.csv", aggregate)
            packages = sorted(
                {
                    f"{d.metadata['Name']}=={d.version}"
                    for d in importlib.metadata.distributions()
                    if d.metadata["Name"]
                }
            )
            add_text(
                "artifacts/collection_environment.txt",
                "# Environment at collection time; not proof of the original training environment.\n"
                + sys.version
                + "\n"
                + platform.platform()
                + "\n"
                + "\n".join(packages)
                + "\n",
            )
            add_text(
                "artifacts/README.md",
                "# Verified original experiment assets\n\nRaw datasets are excluded. Original result JSONs are preserved byte-for-byte.\n"
                + "Q arrays and selected checkpoint bytes match the hashes in those records.\n"
                + "Run `python -m tools.sarbf_net.evaluate --task ETTm1-96 --seed 2021` after preparing data.\n"
                + "See `docs/ASSETS_zh-CN.md`. `manifest.json` lists the included tasks/seeds; it does not assert all paper tasks exist.\n",
            )
            # The asset manifest hashes files above, and excludes itself.
            z.writestr(
                "SARBF-Net/artifacts/manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            )
        os.link(temporary, output)
        Path(temporary).unlink()
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    scope = p.add_mutually_exclusive_group(required=False)
    scope.add_argument(
        "--search-root",
        type=Path,
        help="Search a project directory for final-model result.json files",
    )
    scope.add_argument(
        "--run-root",
        type=Path,
        nargs="+",
        help="Restrict collection to specified experiment folders",
    )
    p.add_argument(
        "--study-root",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Keep an analysis root separate, splitting variants before seed directories",
    )
    p.add_argument("--dataset-root", type=Path, nargs="*", default=[])
    p.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--output", type=Path, default=Path("SARBF-Net_With_Assets.zip"))
    p.add_argument(
        "--report", type=Path, default=Path("SARBF-Net_Collection_Report.json")
    )
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    roots = [a.search_root] if a.search_root else (a.run_root or [])
    studies = []
    for item in a.study_root:
        label, separator, folder = item.partition("=")
        if (
            not separator
            or not folder
            or label == "main"
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label)
        ):
            p.error("--study-root needs a non-main LABEL=PATH")
        studies.append((label, Path(folder)))
    if not roots and not studies:
        p.error("select --run-root, --search-root or --study-root")
    for root in [*roots, *(folder for _, folder in studies)]:
        if not root.is_dir():
            p.error(f"not a directory: {root}")
    plan = scan_groups(roots, studies, a.dataset_root)
    a.report.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(
        f"Eligible runs: {plan['run_count']}; unique Q/weight bytes: {plan['asset_mib']} MiB; errors: {len(plan['errors'])}",
        flush=True,
    )
    print(f"Report: {a.report.resolve()}", flush=True)
    for label, count in plan["study_counts"].items():
        print(f"  {label}: {count} runs", flush=True)
    if plan["errors"]:
        for message in plan["errors"][:12]:
            print(message, file=sys.stderr)
        return 1
    if not plan["runs"]:
        print(
            "No compatible completed records were found. Send the collection report so the actual result layout can be checked.",
            file=sys.stderr,
        )
        return 1
    if not a.dry_run:
        try:
            write_release(plan, a.output, a.source_root)
        except (OSError, ValueError) as e:
            p.exit(1, str(e) + "\n")
        print(f"Created: {a.output.resolve()}", flush=True)
        print(
            "Raw datasets were excluded. Upload weight files with Git LFS; see docs/ASSETS_zh-CN.md."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
