"""Export existing experiment Q arrays byte-for-byte; never synthesize them."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from tools.sa_lcm.matrix_profiles import MATRIX_TASKS, build_profile


def export_q(dataset_root, output, tasks=MATRIX_TASKS):
    dataset_root, output = Path(dataset_root).resolve(), Path(output).resolve()
    paths = {}
    for task in tasks:
        p = build_profile(task)
        relative = Path(p["root"]).relative_to("dataset")
        for key, length in [("q_mat_file", 96), ("q_out_mat_file", p["pred_len"])]:
            paths[relative / p[key]] = length
    manifest = {}
    # Validate everything before opening the archive; fail without a partial file.
    payloads = []
    for relative, length in sorted(paths.items()):
        path = dataset_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"original experiment Q missing: {path}")
        raw = path.read_bytes()
        import io

        q = np.load(io.BytesIO(raw), allow_pickle=False)
        if (
            q.shape != (length, length)
            or not np.issubdtype(q.dtype, np.number)
            or np.iscomplexobj(q)
            or not np.isfinite(q).all()
        ):
            raise ValueError(f"invalid Q: {path}")
        name = (Path("dataset") / relative).as_posix()
        payloads.append((name, raw))
        manifest[name] = dict(
            sha256=hashlib.sha256(raw).hexdigest(), shape=list(q.shape)
        )
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, raw in payloads:
            z.writestr(name, raw)
        z.writestr(
            "Q_ASSETS_MANIFEST.json",
            json.dumps(dict(tasks=list(tasks), files=manifest), indent=2) + "\n",
        )
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("SARBF-Net_Q_assets.zip"))
    p.add_argument("--tasks", nargs="+", choices=MATRIX_TASKS, default=MATRIX_TASKS)
    a = p.parse_args(argv)
    try:
        files = export_q(a.dataset_root, a.output, a.tasks)
    except (OSError, ValueError) as exc:
        p.exit(1, str(exc) + "\n")
    print(f"Exported {len(files)} original Q files to {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
