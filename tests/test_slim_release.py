"""Static contract checks that run before PyTorch and datasets are installed."""

import ast
import argparse
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = (
    "run.py",
    "model/SARBF_Net.py", "model/SA_LCM.py",
    "layers/SAER.py",
    "model/OLinear.py", "model/OLinear_SALCM.py",
    "model/OLinear_SALCM_BCIF.py", "model/OLinear_SALCM_BCIF_NoDiag.py",
    "model/OLinear_SALCM_BCIF_OutputFold.py",
    "ablations/models/OLinear_SALCM_BCIF_OutputFold_NoMass.py",
    "ablations/models/OLinear_SALCM_BCIF_OutputFold_Unbalanced.py",
    "layers/BalancedCrossIntraFeedback.py", "layers/Embed.py", "layers/FANLayer.py",
    "layers/RevIN.py", "layers/SafePruneFrontend.py",
    "layers/CosineHistoricalSelector.py",
    "layers/SelectiveAnchoredLongContext.py", "layers/SelfAttention_Family.py",
    "layers/Transformer_EncDec.py", "layers/newLinear.py",
    "experiments/exp_basic.py", "experiments/exp_forecast.py",
    "experiments/exp_forecast_long_context.py",
    "data_provider/data_factory.py", "data_provider/data_loader.py",
    "data_provider/dataset_physio.py", "data_provider/dataset_pm25.py",
    "data_provider/long_context_factory.py", "data_provider/m4.py",
    "data_provider/split_geometry.py",
    "utils/CKA.py", "utils/masking.py", "utils/metrics.py",
    "utils/timefeatures.py", "utils/tools.py",
    "tools/sa_lcm/run_controlled.py", "tools/sa_lcm/matrix_profiles.py",
    "tools/sa_lcm_transport/run_pilot.py",
    "tools/sa_lcm_safe_prune/replay.py",
)


def validated_record(arm, model, architecture):
    n = 2161
    record = {
        "status": "VALIDATED", "implementation_revision": "SA_LCM_TRANSPORT_V1",
        "architecture_revision": architecture, "task": "ETTh1-720",
        "arm": arm, "seed": 2021, "return_code": 0,
        "evaluation_scope": "validation_only",
        "args": {"model": model, "seq_len": 96, "long_len": 768,
                 "run_seed": 2021},
        "validation": {"mse": 1.5, "mae": 0.8, "samples": n},
        "test": None,
        "sample_manifest": {split: {"count": n, "digest": "b" * 64}
                            for split in ("train", "valid")},
        "q_identity": {"paths_equal": True, "sha256": {
            "q_mat_file": "a" * 64, "q_out_mat_file": "a" * 64}},
        "checkpoint": {"path": "/old/checkpoint.pth", "sha256": "c" * 64},
        "data_sha256": "d" * 64, "source_tree_sha256": "e" * 64,
        "provenance_verified": True, "best_epoch": 3,
        "diagnostics": {"diagnostics_scope": "full_validation", "samples": n},
        "validation_inference_seconds": 1.0,
    }
    return record


class SlimSourceTests(unittest.TestCase):
    def test_runtime_closure_is_present_with_protected_models_and_layers(self):
        self.assertEqual(len(RUNTIME_FILES), 40)
        self.assertEqual(len(set(RUNTIME_FILES)), 40)
        self.assertTrue(all((ROOT / name).is_file() for name in RUNTIME_FILES))
        source_lock = json.loads((ROOT / "MODEL_LAYER_SHA256.json").read_text())
        protected = [name for name in RUNTIME_FILES
                     if name.startswith(("model/", "layers/", "ablations/models/"))]
        self.assertEqual(set(source_lock), set(protected))
        for name in protected:
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                                 source_lock[name])

    def test_digest_api_hashes_exact_shipped_python_sources(self):
        from tools.sa_lcm_transport.run_pilot import SOURCE_FILES, source_tree_digest

        shipped = {str(path.relative_to(ROOT)) for path in ROOT.rglob("*.py")
                   if "__pycache__" not in path.parts}
        self.assertEqual(set(SOURCE_FILES), shipped)
        self.assertEqual(len(SOURCE_FILES), len(set(SOURCE_FILES)))
        self.assertTrue(set(RUNTIME_FILES) <= set(SOURCE_FILES))
        self.assertRegex(source_tree_digest(ROOT), r"^[0-9a-f]{64}$")
        with tempfile.TemporaryDirectory() as folder:
            copy_root = Path(folder)
            for name in SOURCE_FILES:
                path = copy_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / name).read_bytes())
            locked = source_tree_digest(copy_root)
            for name in SOURCE_FILES:
                path = copy_root / name
                path.write_bytes(path.read_bytes() + b"\n# mutation\n")
                self.assertNotEqual(source_tree_digest(copy_root), locked, name)
                path.write_bytes((ROOT / name).read_bytes())

    def test_registered_fresh_arms_and_old_bcif_read_only(self):
        from tools.sa_lcm_transport.run_pilot import (
            ARMS, SAFE_MODELS, assess_result, assess_legacy_bcif_reference, run_task,
        )
        from tools.sa_lcm_safe_prune.replay import MODELS

        self.assertEqual(ARMS, (
            "sa_lcm", "no_diag", "output_fold", "no_mass_bcif",
            "unbalanced_bcif", "pair_free_fold", "sarbf_net",
        ))
        self.assertEqual(SAFE_MODELS, {
            "sa_lcm": "SA_LCM",
            "no_diag": "OLinear_SALCM_BCIF_NoDiag",
            "output_fold": "OLinear_SALCM_BCIF_OutputFold",
            "no_mass_bcif": "OLinear_SALCM_BCIF_OutputFold_NoMass",
            "unbalanced_bcif": "OLinear_SALCM_BCIF_OutputFold_Unbalanced",
            "pair_free_fold": "SA_LCM",
            "sarbf_net": "SARBF_Net",
        })
        self.assertEqual(MODELS, {"bcif": "OLinear_SALCM_BCIF",
                                  "no_diag": "OLinear_SALCM_BCIF_NoDiag",
                                  "output_fold": "OLinear_SALCM_BCIF_OutputFold"})
        current = validated_record("no_diag", MODELS["no_diag"],
                                   "SA_LCM_SAFE_PRUNE_V1")
        legacy = validated_record("bcif", MODELS["bcif"], "SA_LCM_BCIF_V1")
        self.assertEqual(assess_result(current), "VALIDATED")
        self.assertEqual(assess_result(legacy), "MALFORMED")
        self.assertEqual(assess_legacy_bcif_reference(legacy), "VALIDATED")
        for name, value in (("implementation_revision", "wrong"),
                            ("architecture_revision", "wrong"),
                            ("provenance_verified", False),
                            ("source_tree_sha256", None)):
            with self.subTest(name=name):
                bad = copy.deepcopy(legacy)
                bad[name] = value
                self.assertEqual(assess_legacy_bcif_reference(bad), "MALFORMED")
        with self.assertRaisesRegex(ValueError, "not registered"):
            run_task("ETTh1-720", ROOT, ROOT / "runs", arm="bcif",
                     validation_only=True)
        tree = ast.parse((ROOT / "experiments/exp_basic.py").read_text())
        imported = {alias.name for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module == "model"
                    for alias in node.names}
        self.assertEqual(imported, {
            "SARBF_Net", "SA_LCM",
            "OLinear_SALCM_BCIF", "OLinear_SALCM_BCIF_NoDiag",
            "OLinear_SALCM_BCIF_OutputFold",
        })
        ablation_imported = {alias.name for node in ast.walk(tree)
                             if isinstance(node, ast.ImportFrom)
                             and node.module == "ablations.models"
                             for alias in node.names}
        self.assertEqual(ablation_imported, {
            "OLinear_SALCM_BCIF_OutputFold_NoMass",
            "OLinear_SALCM_BCIF_OutputFold_Unbalanced",
        })
        run_source = (ROOT / "run.py").read_text()
        self.assertIn("if args.model not in allowed_models:", run_source)

    def test_validation_screen_never_touches_test_split(self):
        from tools.sa_lcm_transport.run_pilot import run_task

        source = (ROOT / "tools/sa_lcm_transport/run_pilot.py").read_text()
        self.assertIn('if not validation_only:', source)
        self.assertIn('evaluate_split(result["args"], selected, "test")', source)
        self.assertIn('env["SA_LCM_PRUNE_VALIDATION_ONLY"] = "1"', source)
        # Exercise the actual upstream training redirect without importing PyTorch.
        flags = []
        class FakeForecast:
            def train(self, setting):
                self._get_data("test")

        module = ast.parse((ROOT / "experiments/exp_forecast_long_context.py").read_text())
        cls = next(node for node in module.body if isinstance(node, ast.ClassDef))
        isolated = ast.Module(body=[cls], type_ignores=[])
        context = {"Exp_Forecast": FakeForecast, "os": os,
                   "long_context_data_provider":
                       lambda args, flag, test_batch_size=None: flags.append(flag)}
        exec(compile(ast.fix_missing_locations(isolated), "<experiment>", "exec"), context)
        experiment = context["Exp_Forecast_LongContext"]()
        experiment.args = SimpleNamespace()
        with patch.dict(os.environ, {"SA_LCM_PRUNE_TEST_BLIND": "1"}), \
             redirect_stdout(io.StringIO()):
            experiment.train("unit-test")
        self.assertEqual(flags, ["valid"])
        experiment._get_data("test")
        self.assertEqual(flags, ["valid", "test"])

    def test_legacy_probe_entrypoint_cannot_launch_unshipped_baselines(self):
        tree = ast.parse((ROOT / "tools/sa_lcm/run_controlled.py").read_text())
        method = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "main")
        namespace = {"argparse": argparse, "PROBE_TASKS": ("ETTh1-720",),
                     "build_configs": lambda *_args: self.fail("launched old baseline")}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     "<legacy_runner>", "exec"), namespace)
        with redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit) as terminated:
            namespace["main"](["--task", "ETTh1-720"])
        self.assertEqual(terminated.exception.code, 2)


class ReplayReferenceTests(unittest.TestCase):
    def test_old_source_root_required_and_verified_against_old_record(self):
        from tools.sa_lcm_safe_prune.replay import (
            original_source_digest, sha256, verify_reference,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            (original / "tools/sa_lcm_transport").mkdir(parents=True)
            (original / "tools/sa_lcm_transport/run_pilot.py").write_text(
                "SOURCE_FILES = ('original_model.py',)\n")
            (original / "original_model.py").write_text("original weights architecture\n")
            old_result = root / "runs/seed2021/ETTh1-720/bcif/validation/result.json"
            old_result.parent.mkdir(parents=True)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"original weights")
            data_root = root / "dataset"
            data_root.mkdir()
            (data_root / "ETTh1.csv").write_text("original data\n")
            record = validated_record("bcif", "OLinear_SALCM_BCIF", "SA_LCM_BCIF_V1")
            record["source_tree_sha256"] = original_source_digest(original)
            record["checkpoint"] = {"path": str(checkpoint), "sha256": sha256(checkpoint)}
            record["data_sha256"] = sha256(data_root / "ETTh1.csv")
            record["args"].update(root_path=str(data_root), data_path="ETTh1.csv")
            q = record["q_identity"]
            fake_control = types.ModuleType("tools.sa_lcm.run_controlled")
            fake_control._q_identity = lambda _left, _right: q
            with patch.dict(sys.modules, {"tools.sa_lcm.run_controlled": fake_control}), \
                 patch("tools.sa_lcm_safe_prune.replay.scope_target_manifest",
                       return_value=record["sample_manifest"]):
                hashes = verify_reference(record, old_result, checkpoint, original)
                self.assertEqual(hashes["source"], record["source_tree_sha256"])
                with self.assertRaises((FileNotFoundError, TypeError, ValueError)):
                    verify_reference(record, old_result, checkpoint, root / "not_original")
                corrupted = copy.deepcopy(record)
                corrupted["sample_manifest"]["valid"]["digest"] = "0" * 64
                with self.assertRaises(ValueError):
                    verify_reference(corrupted, old_result, checkpoint, original)
                corrupted = copy.deepcopy(record)
                corrupted["q_identity"]["sha256"]["q_mat_file"] = "0" * 64
                with self.assertRaises(ValueError):
                    verify_reference(corrupted, old_result, checkpoint, original)


if __name__ == "__main__":
    unittest.main()
