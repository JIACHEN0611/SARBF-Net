"""Contracts retained from selecting the final Pair-Free SA-LCM graph."""

import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FINAL_MODEL = "SA_LCM"


class CompactSourceTests(unittest.TestCase):
    def test_final_modules_exist_and_pair_free_source_has_no_pair_path(self):
        for module in (
            "layers.CosineHistoricalSelector",
            f"model.{FINAL_MODEL}",
        ):
            with self.subTest(module=module):
                self.assertIsNotNone(importlib.util.find_spec(module))

        source = (ROOT / "layers/CosineHistoricalSelector.py").read_text()
        self.assertNotIn("pair_scorer", source)
        self.assertNotIn("torch.cat([q7", source)
        tree = ast.parse(source)
        classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        self.assertIn("CosineHistoricalSelector", classes)

    def test_runner_registers_one_primary_graph_and_legacy_arm_alias(self):
        from tools.sa_lcm_transport.run_pilot import (
            ARCHITECTURES,
            FINAL_REVISION,
            PRIMARY_ARM,
            SAFE_MODELS,
        )

        self.assertEqual(PRIMARY_ARM, "sa_lcm")
        self.assertEqual(SAFE_MODELS[PRIMARY_ARM], FINAL_MODEL)
        self.assertEqual(SAFE_MODELS["pair_free_fold"], FINAL_MODEL)
        self.assertEqual(ARCHITECTURES[PRIMARY_ARM], FINAL_REVISION)
        self.assertEqual(ARCHITECTURES["pair_free_fold"], FINAL_REVISION)
        self.assertNotIn("output_fold_clean", SAFE_MODELS)

        experiment_source = (ROOT / "experiments/exp_basic.py").read_text()
        run_source = (ROOT / "run.py").read_text()
        self.assertIn(FINAL_MODEL, experiment_source)
        self.assertIn(FINAL_MODEL, run_source)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required on server")
class CompactNumericalTests(unittest.TestCase):
    def _configs(self, folder, horizon=96, channels=7, layers=2):
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

    def test_pair_free_removes_parameters_without_changing_initial_prediction_or_rng(self):
        import torch
        from model.OLinear_SALCM_BCIF_OutputFold import Model as Full
        from model.SA_LCM import Model as PairFree

        with tempfile.TemporaryDirectory() as directory:
            cfg = self._configs(Path(directory), horizon=7, channels=3)
            torch.manual_seed(2021)
            full = Full(cfg).eval()
            full_rng = torch.random.get_rng_state().clone()
            torch.manual_seed(2021)
            candidate = PairFree(cfg).eval()
            self.assertTrue(torch.equal(full_rng, torch.random.get_rng_state()))
            self.assertEqual(
                sum(parameter.numel() for parameter in full.parameters())
                - sum(parameter.numel() for parameter in candidate.parameters()),
                16513,
            )
            self.assertFalse(any("pair_scorer" in name
                                 for name, _ in candidate.named_parameters()))
            candidate_state = candidate.state_dict()
            clean_state = full.state_dict()
            self.assertEqual(
                set(candidate_state),
                {key for key in clean_state if "selector.pair_scorer" not in key},
            )
            for key, value in candidate_state.items():
                self.assertTrue(torch.equal(value, clean_state[key]), key)
            device = full.Q_mat.device
            full = full.to(device)
            candidate = candidate.to(device)
            torch.manual_seed(99)
            x = torch.randn(2, 768, cfg.enc_in, device=device)
            with torch.no_grad():
                torch.testing.assert_close(candidate(x), full(x), atol=0, rtol=0)

    def test_cosine_selector_matches_manual_softmax_and_keeps_mass_identity(self):
        import torch
        from layers.CosineHistoricalSelector import CosineHistoricalSelector
        from layers.SelectiveAnchoredLongContext import HistoricalSelector

        full = HistoricalSelector()
        selector = CosineHistoricalSelector.from_existing(full)
        with torch.no_grad():
            selector.null_head.weight.zero_()
            selector.null_head.bias.fill_(0.25)
            selector.age_bias.copy_(torch.linspace(-0.3, 0.3, 7))
            q = torch.randn(2, 3, 64)
            h = torch.randn(2, 3, 7, 64)
            output = selector(q, h)
            logits = torch.cat([
                torch.full((2, 3, 1), 0.25),
                4.0 * output.cosine_scores + selector.age_bias,
            ], dim=-1)
            expected = torch.softmax(logits, dim=-1)
            torch.testing.assert_close(output.history_weights, expected[..., 1:])
            torch.testing.assert_close(output.null_weight, expected[..., 0])
            torch.testing.assert_close(
                output.history_mass,
                output.history_weights.sum(-1),
                atol=0,
                rtol=0,
            )

    def test_pair_free_keeps_frontend_and_bcif_gradients(self):
        import torch
        from model.SA_LCM import Model as PairFree

        with tempfile.TemporaryDirectory() as directory:
            cfg = self._configs(Path(directory), horizon=5, channels=4)
            torch.manual_seed(31)
            model = PairFree(cfg).train()
            device = model.Q_mat.device
            model = model.to(device)
            with torch.no_grad():
                model.sa_lcm.correction[-1].weight.normal_(std=0.02)
            output = model(torch.randn(2, 768, cfg.enc_in, device=device))
            self.assertEqual(tuple(output.shape), (2, cfg.pred_len, cfg.enc_in))
            output.square().mean().backward()
            for name, parameter in (
                ("age_bias", model.sa_lcm.selector.age_bias),
                ("null_head", model.sa_lcm.selector.null_head.weight),
                ("correction", model.sa_lcm.correction[-1].weight),
                ("bcif_up", model.coordinators[0].up.weight),
            ):
                with self.subTest(name=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
            with self.assertRaises(ValueError):
                model(torch.randn(2, 767, cfg.enc_in, device=device))


if __name__ == "__main__":
    unittest.main()
