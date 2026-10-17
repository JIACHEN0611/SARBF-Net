"""Synthetic CPU forward/backward test. Identity Q is ONLY a test fixture."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from model.SARBF_Net import Model


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        np.save(root / "input.npy", np.eye(96, dtype=np.float32))
        np.save(root / "output.npy", np.eye(12, dtype=np.float32))
        cfg = SimpleNamespace(
            seq_len=96,
            long_len=768,
            pred_len=12,
            enc_in=3,
            d_model=16,
            d_ff=32,
            embed_size=4,
            e_layers=2,
            temp_patch_len=16,
            temp_stride=8,
            CKA_flag=0,
            dropout=0.0,
            activation="gelu",
            Q_chan_indep=0,
            q_mat_file="input.npy",
            q_out_mat_file="output.npy",
            root_path=folder,
        )
        torch.manual_seed(2021)
        model = Model(cfg).to("cpu")
        x = torch.randn(2, 768, 3)
        prediction = model(x)
        assert prediction.shape == (2, 12, 3)
        assert torch.isfinite(prediction).all()
        prediction.square().mean().backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        print("PASS: SARBF-Net CPU forward/backward; [2,768,3] -> [2,12,3].")
        print("Synthetic fixture only; this does not reproduce paper metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
