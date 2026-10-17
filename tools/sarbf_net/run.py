"""Public SARBF-Net runner using locked profiles and checkpoint replay."""

import argparse
import json
from pathlib import Path

from tools.sa_lcm.matrix_profiles import MATRIX_TASKS, build_profile
from tools.sarbf_net.inputs import check_inputs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=MATRIX_TASKS, required=True)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument(
        "--repository-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    p.add_argument("--run-root", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--validation-only", action="store_true")
    a = p.parse_args(argv)
    root = a.repository_root.resolve()
    if a.dry_run:
        print(
            json.dumps(
                dict(
                    model="SARBF_Net",
                    arm="sarbf_net",
                    task=a.task,
                    seed=a.seed,
                    profile=build_profile(a.task),
                    seq_len=96,
                    long_len=768,
                ),
                indent=2,
            )
        )
        return 0
    errors = check_inputs(root, [a.task])
    if errors:
        p.exit(1, "\n".join(errors) + "\nSee docs/DATA.md for the original Q assets.\n")
    from tools.sa_lcm_transport.run_pilot import run_task

    result = run_task(
        a.task,
        root,
        a.run_root or root / "runs/sarbf_net",
        a.seed,
        "sarbf_net",
        validation_only=a.validation_only,
    )
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "task",
                    "arm",
                    "seed",
                    "status",
                    "validation",
                    "test",
                    "artifact_error",
                )
            },
            indent=2,
        )
    )
    return 0 if result["status"] in ("COMPLETE", "VALIDATED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
