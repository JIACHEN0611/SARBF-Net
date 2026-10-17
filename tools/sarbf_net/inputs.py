"""Check data existence and Q matrix shape/finiteness before training."""

import argparse
from pathlib import Path

import numpy as np

from tools.sa_lcm.matrix_profiles import MATRIX_TASKS, build_profile


def check_inputs(repository_root, tasks):
    errors, checked = [], set()
    for task in tasks:
        profile = build_profile(task)
        root = Path(repository_root) / profile["root"]
        data = root / profile["data_path"]
        if data not in checked:
            checked.add(data)
            if not data.is_file():
                errors.append(f"missing data: {data}")
        for key, length in [
            ("q_mat_file", 96),
            ("q_out_mat_file", profile["pred_len"]),
        ]:
            path = root / profile[key]
            if (path, length) in checked:
                continue
            checked.add((path, length))
            if not path.is_file():
                errors.append(f"missing Q: {path}")
                continue
            try:
                q = np.load(path, allow_pickle=False)
                if q.shape != (length, length):
                    errors.append(
                        f"Q shape {q.shape}, expected {(length, length)}: {path}"
                    )
                elif (
                    not np.issubdtype(q.dtype, np.number)
                    or np.iscomplexobj(q)
                    or not np.isfinite(q).all()
                ):
                    errors.append(f"Q must be finite real numbers: {path}")
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"invalid Q: {path}: {exc}")
    return errors


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repository-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    p.add_argument("--tasks", nargs="+", choices=MATRIX_TASKS, default=MATRIX_TASKS)
    a = p.parse_args(argv)
    errors = check_inputs(a.repository_root, a.tasks)
    print(
        "\n".join(errors)
        if errors
        else f"PASS: {len(a.tasks)} tasks; data exists and Q matrices are finite with expected shapes."
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
