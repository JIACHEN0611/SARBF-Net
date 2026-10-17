"""Behavioral contracts for the expanded long-horizon experiment matrix."""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require_module(testcase, name):
    try:
        return __import__(name, fromlist=["*"])
    except ModuleNotFoundError:
        testcase.fail(f"required module is missing: {name}")


class MatrixProfileTests(unittest.TestCase):
    def test_pilot_accepts_every_registered_matrix_task(self):
        from tools.sa_lcm.matrix_profiles import MATRIX_TASKS
        from tools.sa_lcm_transport.run_pilot import PILOT

        self.assertEqual(PILOT, MATRIX_TASKS)

    def test_all_available_dataset_horizon_pairs_build_complete_profiles(self):
        module = require_module(self, "tools.sa_lcm.matrix_profiles")
        MATRIX_TASKS, build_profile = module.MATRIX_TASKS, module.build_profile

        long_datasets = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "ECL",
                         "Exchange", "Traffic", "Weather", "Solar")
        pems_datasets = ("PEMS03", "PEMS04", "PEMS07", "PEMS08")
        expected = tuple(
            [f"{dataset}-{horizon}" for dataset in long_datasets
             for horizon in (96, 192, 336, 720)]
            + [f"{dataset}-{horizon}" for dataset in pems_datasets
               for horizon in (12, 24, 48, 96)]
        )
        self.assertEqual(MATRIX_TASKS, expected)
        self.assertEqual(len(set(MATRIX_TASKS)), 52)

        for task in expected:
            with self.subTest(task=task):
                profile = build_profile(task)
                self.assertEqual(profile["pred_len"], int(task.rsplit("-", 1)[1]))
                self.assertGreater(profile["enc_in"], 0)
                self.assertGreaterEqual(profile["e_layers"], 2)
                self.assertIn(profile["lradj"], ("type1", "type3", "cosine"))

    def test_profiles_preserve_dataset_recipes_and_enable_ettm2_feedback(self):
        build_profile = require_module(
            self, "tools.sa_lcm.matrix_profiles"
        ).build_profile

        cases = {
            "ETTh1-96": ("ETTh1", "ETTh1.csv", 7, 5e-4, 30, 2, 0.2, "type1"),
            "ETTh1-720": ("ETTh1", "ETTh1.csv", 7, 1e-4, 30, 2, 0.2, "type1"),
            "ETTh2-720": ("ETTh2", "ETTh2.csv", 7, 2e-4, 30, 3, 0.3, "type1"),
            "ETTm1-336": ("ETTm1", "ETTm1.csv", 7, 1e-4, 30, 2, 0.1, "type1"),
            "ETTm2-336": ("ETTm2", "ETTm2.csv", 7, 1e-4, 30, 2, 0.2, "type1"),
            "ECL-96": ("custom", "electricity.csv", 321, 5e-4, 30, 3, 0.0, "cosine"),
            "Traffic-192": ("custom", "traffic.csv", 862, 5e-4, 50, 3, 0.0, "cosine"),
            "Weather-720": ("custom", "weather.csv", 21, 1e-3, 30, 2, 0.0, "type3"),
            "Solar-336": ("Solar", "solar_AL.txt", 137, 5e-4, 40, 2, 0.0, "type1"),
        }
        for task, expected in cases.items():
            with self.subTest(task=task):
                profile = build_profile(task)
                actual = tuple(profile[key] for key in (
                    "data", "data_path", "enc_in", "learning_rate",
                    "train_epochs", "e_layers", "dropout", "lradj",
                ))
                self.assertEqual(actual, expected)

        exchange = build_profile("Exchange-192")
        self.assertEqual(
            (exchange["data"], exchange["root"], exchange["data_path"],
             exchange["enc_in"], exchange["d_model"], exchange["batch_size"],
             exchange["q_mat_file"], exchange["q_out_mat_file"]),
            ("custom", "dataset/exchange_rate", "exchange_rate.csv", 8,
             128, 32, "exchange_rate_96_ratio0.7.npy",
             "exchange_rate_192_ratio0.7.npy"),
        )

        pems = build_profile("PEMS03-12")
        self.assertEqual(
            (pems["data"], pems["root"], pems["data_path"], pems["enc_in"],
             pems["pred_len"], pems["e_layers"], pems["train_epochs"],
             pems["q_mat_file"], pems["q_out_mat_file"]),
            ("PEMS", "dataset/PEMS", "PEMS03.npz", 358, 12, 3, 40,
             "PEMS03_96_ratio0.6.npy", "PEMS03_12_ratio0.6.npy"),
        )
        pems07 = build_profile("PEMS07-96")
        self.assertEqual(pems07["enc_in"], 883)
        self.assertEqual(pems07["q_mat_file"], "PEMS07_96.npy")
        self.assertEqual(pems07["q_out_mat_file"], "PEMS07_96.npy")

        with self.assertRaisesRegex(ValueError, "unsupported matrix task"):
            build_profile("PEMS03-192")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required")
    def test_config_builder_uses_profile_width_batch_and_q_names(self):
        from tools.sa_lcm.run_controlled import build_configs

        _, exchange = build_configs("Exchange-192", ROOT / "runs", ROOT)
        self.assertEqual(exchange["d_model"], 128)
        self.assertEqual(exchange["d_ff"], 128)
        self.assertEqual(exchange["batch_size"], 32)
        self.assertEqual(exchange["q_mat_file"], "exchange_rate_96_ratio0.7.npy")

        _, pems = build_configs("PEMS07-12", ROOT / "runs", ROOT)
        self.assertEqual(pems["data"], "PEMS")
        self.assertEqual(pems["q_mat_file"], "PEMS07_96.npy")
        self.assertEqual(pems["q_out_mat_file"], "PEMS07_12.npy")

    def test_preflight_reports_every_missing_data_or_q_input_before_training(self):
        from tools.sa_lcm.matrix_profiles import missing_inputs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = missing_inputs(root, ("Exchange-96", "PEMS03-12"))
            expected = {
                root / "dataset/exchange_rate/exchange_rate.csv",
                root / "dataset/exchange_rate/exchange_rate_96_ratio0.7.npy",
                root / "dataset/PEMS/PEMS03.npz",
                root / "dataset/PEMS/PEMS03_96_ratio0.6.npy",
                root / "dataset/PEMS/PEMS03_12_ratio0.6.npy",
            }
            self.assertEqual(set(missing), expected)
            for path in expected:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual(missing_inputs(root, ("Exchange-96", "PEMS03-12")), ())


class TargetGeometryTests(unittest.TestCase):
    def test_target_start_matches_each_loader_split_geometry(self):
        global_target_start = require_module(
            self, "data_provider.split_geometry"
        ).global_target_start

        hourly = {"train": 768, "valid": 8640, "test": 11520}
        minute = {"train": 768, "valid": 34560, "test": 46080}
        custom = {"train": 768, "valid": 7000, "test": 8000}
        for dataset in ("ETTh1", "ETTh2"):
            self.assertEqual(
                {flag: global_target_start(dataset, flag, 17420, 768)
                 for flag in ("train", "valid", "test")},
                hourly,
            )
        for dataset in ("ETTm1", "ETTm2"):
            self.assertEqual(
                {flag: global_target_start(dataset, flag, 69680, 768)
                 for flag in ("train", "valid", "test")},
                minute,
            )
        self.assertEqual(
            {flag: global_target_start("custom", flag, 10000, 768)
             for flag in ("train", "valid", "test")},
            custom,
        )
        self.assertEqual(
            {flag: global_target_start("PEMS", flag, 10000, 768)
             for flag in ("train", "valid", "test")},
            {"train": 768, "valid": 6096, "test": 8096},
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required")
    def test_pems_long_context_keeps_the_original_96_input_eval_targets(self):
        import numpy as np
        from data_provider.data_loader import Dataset_PEMS

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "PEMS03.npz"
            values = np.arange(5000 * 3, dtype=np.float32).reshape(5000, 3, 1)
            np.savez(path, data=values)
            for split in ("valid", "test"):
                reference = Dataset_PEMS(
                    directory, flag=split, size=[96, 0, 12],
                    data_path=path.name, features="M",
                )
                long_context = Dataset_PEMS(
                    directory, flag=split, size=[768, 0, 12],
                    data_path=path.name, features="M",
                )
                self.assertEqual(len(long_context), len(reference))

    def test_long_context_row_count_supports_pems_npz(self):
        import numpy as np
        from data_provider.split_geometry import source_row_count

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "PEMS03.npz"
            np.savez(path, data=np.zeros((1234, 358, 1), dtype=np.float32))
            self.assertEqual(source_row_count(path), 1234)

    def test_target_start_rejects_unknown_split_and_short_dataset(self):
        global_target_start = require_module(
            self, "data_provider.split_geometry"
        ).global_target_start

        with self.assertRaisesRegex(ValueError, "unknown split"):
            global_target_start("custom", "future", 10000, 768)
        with self.assertRaisesRegex(ValueError, "too short"):
            global_target_start("custom", "train", 1000, 768)


class MatrixScriptTests(unittest.TestCase):
    def test_list_tasks_exposes_each_matrix_task_once_without_training(self):
        script = ROOT / "scripts/run_sa_lcm_matrix.sh"
        completed = subprocess.run(
            ["bash", str(script), "--list-tasks"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        tasks = completed.stdout.splitlines()
        self.assertEqual(len(tasks), 52)
        self.assertEqual(len(set(tasks)), 52)
        self.assertEqual(tasks[0], "ETTh1-96")
        self.assertEqual(tasks[-1], "PEMS08-96")

    def test_matrix_script_preflights_all_inputs_and_selects_one_gpu(self):
        source = (ROOT / "scripts/run_sa_lcm_matrix.sh").read_text()
        self.assertIn("--check-inputs", source)
        self.assertIn("CUDA_VISIBLE_DEVICES", source)


if __name__ == "__main__":
    unittest.main()
