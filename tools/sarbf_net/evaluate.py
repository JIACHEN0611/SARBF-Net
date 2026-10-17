"""Replay a portable published checkpoint using its original data/Q hashes."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("asset path escapes repository")
    return path


def prepare_replay(
    repository_root,
    dataset_root,
    task,
    seed,
    device="cpu",
    num_workers=0,
    study="main",
    variant="standard",
):
    root = Path(repository_root).resolve()
    dataset_root = Path(dataset_root).resolve()
    manifest = json.loads((root / "artifacts/manifest.json").read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("architecture_revision") != "SA_LCM_COMPACT_FINAL_V1"
    ):
        raise ValueError("unsupported asset manifest/architecture")
    rows = [
        r
        for r in manifest["runs"]
        if r["task"] == task
        and r["seed"] == seed
        and r.get("study", "main") == study
        and r.get("variant", "standard") == variant
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one published run for {task} seed{seed}; found {len(rows)}"
        )
    run = rows[0]

    def verified(relative, expected):
        path = contained(root, relative)
        if sha256(path) != expected:
            raise ValueError(f"asset SHA256 mismatch: {relative}")
        return path

    for key in ["checkpoint", "q_input", "q_output"]:
        verified(run[key]["path"], run[key]["sha256"])
    record_path = verified(
        run["result_path"], manifest["files"][run["result_path"]]["sha256"]
    )
    original = json.loads(record_path.read_text())
    if not isinstance(original, dict) or not isinstance(original.get("args"), dict):
        raise ValueError("invalid original record structure")
    original_q = (original.get("q_identity") or {}).get("sha256") or {}
    expected_links = {
        "checkpoint": (original.get("checkpoint") or {}).get("sha256"),
        "q_input": original_q.get("q_mat_file"),
        "q_output": original_q.get("q_out_mat_file"),
    }
    if (
        original.get("task") != task
        or original.get("seed") != seed
        or original.get("architecture_revision") != manifest["architecture_revision"]
        or original.get("data_sha256") != run["data_sha256"]
        or original["args"].get("model") not in {"SA_LCM", "SARBF_Net"}
        or any(run[key]["sha256"] != digest for key, digest in expected_links.items())
    ):
        raise ValueError("portable assets do not match the original record")
    data = contained(dataset_root, run["data_relative_path"])
    if sha256(data) != run["data_sha256"]:
        raise ValueError(f"dataset SHA256 mismatch: {data}")
    args = dict(original["args"])
    args.update(
        model="SARBF_Net",
        root_path=str(data.parent),
        data_path=data.name,
        q_mat_file=str(contained(root, run["q_input"]["path"])),
        q_out_mat_file=str(contained(root, run["q_output"]["path"])),
        use_gpu=int(device == "cuda"),
        gpu=0,
        use_multi_gpu=False,
        devices="0",
        device_ids=[0],
        num_workers=num_workers,
        send_mail=0,
        save_pdf=0,
        folder_path=str(
            root / "replay_output" / study / variant / task / f"seed{seed}"
        ),
    )
    return args, contained(root, run["checkpoint"]["path"]), run, original


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    p.add_argument("--repository-root", type=Path, default=root)
    p.add_argument("--dataset-root", type=Path, default=root / "dataset")
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--study", default="main")
    p.add_argument("--variant", default="standard")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument("--tolerance", type=float, default=1e-5)
    a = p.parse_args(argv)
    if a.num_workers < 0 or not math.isfinite(a.tolerance) or a.tolerance < 0:
        p.error("invalid workers/tolerance")
    try:
        args, checkpoint, run, original = prepare_replay(
            a.repository_root,
            a.dataset_root,
            a.task,
            a.seed,
            a.device,
            a.num_workers,
            a.study,
            a.variant,
        )
        import torch

        if a.device == "cuda" and not torch.cuda.is_available():
            p.error("CUDA requested but unavailable")
        from tools.sa_lcm_transport.run_pilot import evaluate_split

        replay = evaluate_split(args, checkpoint, a.split)
        key = "validation" if a.split == "valid" else "test"
        metrics = replay[key]
        expected = original[key]
        differences = {
            name: float(metrics[name]) - float(expected[name])
            for name in ["mse", "mae"]
        }
        passed = all(
            math.isfinite(x) and abs(x) <= a.tolerance for x in differences.values()
        )
        print(
            json.dumps(
                dict(
                    study=a.study,
                    variant=a.variant,
                    task=a.task,
                    seed=a.seed,
                    split=a.split,
                    replayed=metrics,
                    recorded=expected,
                    differences=differences,
                    absolute_tolerance=a.tolerance,
                    within_tolerance=passed,
                ),
                indent=2,
            )
        )
        return 0 if passed else 1
    except (OSError, ValueError, KeyError) as e:
        p.exit(1, str(e) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
