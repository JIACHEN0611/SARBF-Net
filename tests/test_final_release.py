"""Contracts for the paper-facing compact SA-LCM release."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class FinalReleaseSourceTests(unittest.TestCase):
    def test_one_primary_model_replaces_the_two_candidate_wrappers(self):
        self.assertIsNotNone(importlib.util.find_spec("model.SA_LCM"))
        self.assertFalse(
            (ROOT / "model/OLinear_SALCM_BCIF_OutputFold_Clean.py").exists()
        )
        self.assertFalse(
            (ROOT / "model/OLinear_SALCM_BCIF_OutputFold_PairFree.py").exists()
        )

        from tools.sa_lcm_transport.run_pilot import (
            ARCHITECTURES,
            PRIMARY_ARM,
            SAFE_MODELS,
        )

        self.assertEqual(PRIMARY_ARM, "sa_lcm")
        self.assertEqual(SAFE_MODELS[PRIMARY_ARM], "SA_LCM")
        self.assertEqual(SAFE_MODELS["pair_free_fold"], "SA_LCM")
        self.assertEqual(
            ARCHITECTURES[PRIMARY_ARM], ARCHITECTURES["pair_free_fold"]
        )
        self.assertNotIn("output_fold_clean", SAFE_MODELS)

    def test_mechanism_ablations_are_outside_the_primary_model_package(self):
        self.assertTrue(
            (ROOT / "ablations/models/OLinear_SALCM_BCIF_OutputFold_NoMass.py").is_file()
        )
        self.assertTrue(
            (ROOT / "ablations/models/OLinear_SALCM_BCIF_OutputFold_Unbalanced.py").is_file()
        )
        self.assertFalse(
            (ROOT / "model/OLinear_SALCM_BCIF_OutputFold_NoMass.py").exists()
        )
        self.assertFalse(
            (ROOT / "model/OLinear_SALCM_BCIF_OutputFold_Unbalanced.py").exists()
        )


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required")
class FinalReleaseNumericalTests(unittest.TestCase):
    def _configs(self, folder, horizon=5, channels=3, layers=2):
        np.save(folder / "input.npy", np.eye(96, dtype=np.float32))
        np.save(folder / "output.npy", np.eye(horizon, dtype=np.float32))
        return SimpleNamespace(
            seq_len=96,
            long_len=768,
            pred_len=horizon,
            enc_in=channels,
            d_model=16,
            d_ff=32,
            embed_size=4,
            e_layers=layers,
            temp_patch_len=16,
            temp_stride=8,
            CKA_flag=0,
            dropout=0.0,
            activation="gelu",
            Q_chan_indep=0,
            q_mat_file="input.npy",
            q_out_mat_file="output.npy",
            root_path=str(folder),
        )

    def test_final_model_exactly_matches_the_selected_pair_free_graph(self):
        import torch

        from layers.CosineHistoricalSelector import CosineHistoricalSelector
        from layers.SelectiveAnchoredLongContext import anchor_normalize_past
        from model.OLinear_SALCM_BCIF_OutputFold import Model as Folded
        from model.SA_LCM import Model as Final

        class SelectedPairFreeReference(Folded):
            def __init__(self, configs):
                super().__init__(configs)
                self.sa_lcm.selector = CosineHistoricalSelector.from_existing(
                    self.sa_lcm.selector
                )

            def forward(
                self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None
            ):
                if x.ndim != 3 or x.shape[1:] != (768, self.enc_in):
                    raise ValueError("selected Pair-Free model expects [B,768,C]")
                recent = x[:, -96:, :]
                z_recent = self.revin_layer(recent, mode="norm")
                z_past = anchor_normalize_past(x[:, :672, :], self.revin_layer)
                refined, history_mass = self.sa_lcm.refine_with_mass(
                    z_recent, z_past
                )
                encoded = self._bcif_transform(
                    self.tokenEmb(refined, self.embeddings), history_mass
                )
                preactivation = self.fc[0](encoded) + self.delta_head(
                    self.delta2.squeeze(2)
                )
                prediction = self.fc[2](self.fc[1](preactivation)).transpose(-1, -2)
                return self.revin_layer(self.dropout(prediction), mode="denorm")

        with tempfile.TemporaryDirectory() as directory:
            cfg = self._configs(Path(directory), horizon=7, channels=4, layers=3)
            torch.manual_seed(2021)
            reference = SelectedPairFreeReference(cfg).eval()
            expected_rng = torch.random.get_rng_state().clone()
            torch.manual_seed(2021)
            final = Final(cfg).eval()

            self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))
            self.assertEqual(set(reference.state_dict()), set(final.state_dict()))
            final.load_state_dict(reference.state_dict(), strict=True)
            self.assertFalse(
                any("pair_scorer" in name for name, _ in final.named_parameters())
            )

            device = final.Q_mat.device
            reference = reference.to(device)
            final = final.to(device)
            torch.manual_seed(99)
            x = torch.randn(2, 768, cfg.enc_in, device=device)
            with torch.no_grad():
                expected = reference(x)
                actual = final(x)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
