"""Pure source and split geometry shared by long-context provenance records."""

from pathlib import Path

import numpy as np
import pandas as pd


def source_row_count(path):
    """Return time-axis rows for supported CSV/text and PEMS NPZ sources."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "data" not in archive:
                raise ValueError(f"PEMS archive has no data array: {path}")
            data = archive["data"]
            if data.ndim < 2:
                raise ValueError(f"PEMS data must have at least two axes: {path}")
            return int(data.shape[0])
    return len(pd.read_csv(path, usecols=[0]))


def global_target_start(data, flag, row_count, context_length):
    """Return the global index of the first forecast target in a split."""
    if flag not in ("train", "valid", "test"):
        raise ValueError(f"unknown split: {flag}")
    if row_count <= context_length:
        raise ValueError("dataset is too short for the long context")

    if data in ("ETTh1", "ETTh2"):
        starts = {"train": context_length, "valid": 12 * 30 * 24,
                  "test": (12 + 4) * 30 * 24}
    elif data in ("ETTm1", "ETTm2"):
        starts = {"train": context_length, "valid": 12 * 30 * 24 * 4,
                  "test": (12 + 4) * 30 * 24 * 4}
    elif data == "PEMS":
        train_end = int(row_count * 0.6)
        valid_end = int(row_count * 0.8)
        reference_input = 96
        if (train_end <= context_length
                or min(valid_end - train_end,
                       row_count - valid_end) <= reference_input):
            raise ValueError("PEMS split is too short for the long context")
        starts = {"train": context_length,
                  "valid": train_end + reference_input,
                  "test": valid_end + reference_input}
    else:
        train_end = int(row_count * 0.7)
        if train_end <= context_length:
            raise ValueError("dataset is too short for the long context")
        test_start = row_count - int(row_count * 0.2)
        starts = {"train": context_length, "valid": train_end, "test": test_start}
    start = int(starts[flag])
    if start >= row_count:
        raise ValueError("dataset is too short for the requested split geometry")
    return start
