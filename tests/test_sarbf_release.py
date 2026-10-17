"""Public model, input validation and export regression contracts."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseTests(unittest.TestCase):
    def test_external_gpu_visibility_is_preserved_for_train_and_replay(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch

        from experiments.exp_basic import Exp_Basic

        args = SimpleNamespace(use_gpu=True, gpu=0, use_multi_gpu=False, devices="0")
        experiment = SimpleNamespace(args=args)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}):
            device = Exp_Basic._acquire_device(experiment)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")
            self.assertEqual(str(device), "cuda:0")

    def test_checkpoint_epoch_uses_saved_marker_on_equal_losses(self):
        import contextlib
        import io

        import torch

        from tools.sa_lcm_transport.run_pilot import selected_checkpoint_epoch
        from utils.tools import EarlyStopping

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = torch.nn.Linear(2, 1)
            stop = EarlyStopping()
            with contextlib.redirect_stdout(io.StringIO()):
                stop(0.12345671, model, folder, epoch=1)
                stop(0.12345670, model, folder, epoch=2)
                stop(0.12345670, model, folder, epoch=3)
            self.assertEqual(selected_checkpoint_epoch(root / "checkpoint.pth"), 3)
            (root / "Epoch_99.txt").touch()
            with self.assertRaisesRegex(ValueError, "epoch marker"):
                selected_checkpoint_epoch(root / "checkpoint.pth")

    def test_new_model_entry_is_available(self):
        self.assertIsNotNone(importlib.util.find_spec("model.SARBF_Net"))

    def test_public_dry_run_selects_new_model_without_data_or_training(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.sarbf_net.run",
                "--task",
                "ETTm1-96",
                "--dry-run",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(record["model"], "SARBF_Net")
        self.assertEqual(record["arm"], "sarbf_net")
        self.assertEqual(record["profile"]["enc_in"], 7)

    def test_input_checker_rejects_missing_corrupt_and_wrong_shape_q(self):
        from tools.sa_lcm.matrix_profiles import build_profile
        from tools.sarbf_net.inputs import check_inputs

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            errors = check_inputs(root, ["ETTm1-96"])
            self.assertTrue(any("missing" in error for error in errors))
            p = build_profile("ETTm1-96")
            data_root = root / p["root"]
            data_root.mkdir(parents=True)
            (data_root / p["data_path"]).write_text("date,OT\n2020-01-01,1\n")
            q = data_root / p["q_mat_file"]
            np.save(q, np.eye(95))
            self.assertTrue(
                any("shape" in error for error in check_inputs(root, ["ETTm1-96"]))
            )
            q.write_bytes(b"not a numpy matrix")
            self.assertTrue(
                any("invalid" in error for error in check_inputs(root, ["ETTm1-96"]))
            )
            bad = np.eye(96)
            bad[0, 0] = np.nan
            np.save(q, bad)
            self.assertTrue(
                any("finite" in error for error in check_inputs(root, ["ETTm1-96"]))
            )
            np.save(q, np.eye(96))
            self.assertEqual(check_inputs(root, ["ETTm1-96"]), [])

    def test_q_export_uses_actual_arrays_and_no_dataset_bytes(self):
        from tools.sa_lcm.matrix_profiles import build_profile
        from tools.sarbf_net.export_q import export_q

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset_root = root / "dataset"
            profile = build_profile("ETTm1-96")
            q = root / profile["root"] / profile["q_mat_file"]
            q.parent.mkdir(parents=True)
            np.save(q, np.eye(96))
            (q.parent / "ETTm1.csv").write_text("private data")
            output = root / "q.zip"
            export_q(dataset_root, output, ["ETTm1-96"])
            with zipfile.ZipFile(output) as archive:
                self.assertNotIn("dataset/ETT-small/ETTm1.csv", archive.namelist())
                self.assertEqual(
                    archive.read("dataset/ETT-small/" + q.name), q.read_bytes()
                )
                self.assertIn("Q_ASSETS_MANIFEST.json", archive.namelist())
            with self.assertRaises(FileExistsError):
                export_q(dataset_root, output, ["ETTm1-96"])

    def test_summary_does_not_average_failed_or_duplicate_seeds(self):
        from tools.sarbf_net.summarize import summarize

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i, (seed, status, mse) in enumerate(
                [(2021, "COMPLETE", 0.2), (2022, "COMPLETE", 0.4), (2023, "FAILED", 99)]
            ):
                p = root / str(i) / "result.json"
                p.parent.mkdir()
                p.write_text(
                    json.dumps(
                        dict(
                            task="ETTm1-96",
                            arm="sarbf_net",
                            seed=seed,
                            status=status,
                            test=dict(mse=mse, mae=0.1),
                        )
                    )
                )
            rows = summarize(root)
            self.assertEqual(rows[0]["seeds"], "2021,2022")
            self.assertEqual(rows[0]["n"], 2)
            self.assertAlmostEqual(rows[0]["mse_mean"], 0.3)
            self.assertAlmostEqual(rows[0]["mse_std"], np.std([0.2, 0.4], ddof=1))
            duplicate = root / "duplicate/result.json"
            duplicate.parent.mkdir()
            duplicate.write_bytes((root / "0/result.json").read_bytes())
            with self.assertRaisesRegex(ValueError, "duplicate"):
                summarize(root)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required")
class RenamingNumericalTests(unittest.TestCase):
    def test_q_buffers_follow_dtype_without_entering_legacy_checkpoint(self):
        import torch
        from test_final_release import FinalReleaseNumericalTests

        from layers.SAER import SAER
        from model.SARBF_Net import Model

        with tempfile.TemporaryDirectory() as folder:
            cfg = FinalReleaseNumericalTests()._configs(Path(folder), horizon=7)
            model = Model(cfg)
            self.assertNotIn("Q_mat", model.state_dict())
            self.assertNotIn("Q_out_mat", model.state_dict())
            frontend = SAER()
            frontend.load_state_dict(model.saer.state_dict(), strict=True)
            recent = torch.randn(2, 96, cfg.enc_in)
            past = torch.randn(2, 672, cfg.enc_in)
            with torch.no_grad():
                for actual, expected in zip(
                    frontend.refine_with_mass(recent, past),
                    model.saer.refine_with_mass(recent, past),
                ):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            model = model.double()
            self.assertEqual(model.Q_mat.dtype, torch.float64)
            self.assertEqual(model.Q_out_mat.dtype, torch.float64)
            self.assertTrue(
                torch.isfinite(
                    model(torch.randn(2, 768, cfg.enc_in, dtype=torch.float64))
                ).all()
            )

    def test_new_entry_preserves_rng_keys_predictions_and_strict_loading(self):
        import torch
        from test_final_release import FinalReleaseNumericalTests

        from model.SA_LCM import Model as Legacy
        from model.SARBF_Net import Model as Current

        with tempfile.TemporaryDirectory() as folder:
            cfg = FinalReleaseNumericalTests()._configs(Path(folder), horizon=7)
            torch.manual_seed(2021)
            old = Legacy(cfg).eval()
            rng = torch.get_rng_state()
            torch.manual_seed(2021)
            new = Current(cfg).eval()
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(set(old.state_dict()), set(new.state_dict()))
            new.load_state_dict(old.state_dict(), strict=True)
            self.assertIs(new.saer, new.sa_lcm)
            x = torch.randn(2, 768, cfg.enc_in)
            with torch.no_grad():
                torch.testing.assert_close(new(x), old(x), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
