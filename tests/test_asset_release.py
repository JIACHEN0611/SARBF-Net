"""Collection must keep real records matched to their Q arrays and checkpoint."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def collector():
    path = ROOT / "scripts/collect_release.py"
    spec = importlib.util.spec_from_file_location("asset_collector", path)
    if spec is None or not path.exists():
        raise AssertionError("asset collector is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(
    root, seed=2021, name="run", checkpoint=b"trained-checkpoint", q=b"original-q"
):
    folder = root / name
    folder.mkdir(parents=True)
    weights = folder / "checkpoints/checkpoint.pth"
    weights.parent.mkdir()
    weights.write_bytes(checkpoint)
    matrix = folder / "original.npy"
    matrix.write_bytes(q)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    record = dict(
        status="COMPLETE",
        arm="sa_lcm",
        architecture_revision="SA_LCM_COMPACT_FINAL_V1",
        task="ETTm1-96",
        seed=seed,
        args=dict(
            model="SA_LCM",
            seq_len=96,
            long_len=768,
            pred_len=96,
            enc_in=7,
            root_path=str(folder),
            data_path="ETTm1.csv",
            q_mat_file=matrix.name,
            q_out_mat_file=matrix.name,
        ),
        checkpoint=dict(path=str(weights), sha256=sha(checkpoint)),
        q_identity=dict(
            files=dict(q_mat_file=str(matrix), q_out_mat_file=str(matrix)),
            sha256=dict(q_mat_file=sha(q), q_out_mat_file=sha(q)),
        ),
        data_sha256="d" * 64,
        source_tree_sha256="e" * 64,
        validation=dict(mse=0.3, mae=0.4, samples=5),
        test=dict(mse=0.4, mae=0.5, samples=5),
        provenance_verified=True,
        return_code=0,
        best_epoch=1,
    )
    path = folder / "result.json"
    path.write_text(json.dumps(record))
    (folder / "ETTm1.csv").write_text("raw data must not be bundled")
    return path, record


class AssetCollectionTests(unittest.TestCase):
    def test_real_records_weights_q_and_source_are_packed_without_raw_data(self):
        m = collector()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, record = fixture(root)
            plan = m.scan([root])
            self.assertEqual(plan["errors"], [])
            self.assertEqual(len(plan["runs"]), 1)
            source = root / "source"
            source.mkdir()
            (source / "README.md").write_text("SARBF-Net")
            (source / "dataset").mkdir()
            (source / "dataset/private.csv").write_text("private data")
            output = root / "release.zip"
            m.write_release(plan, output, source)
            with zipfile.ZipFile(output) as z:
                names = z.namelist()
                self.assertNotIn("SARBF-Net/dataset/private.csv", names)
                manifest = json.loads(z.read("SARBF-Net/artifacts/manifest.json"))
                run = manifest["runs"][0]
                self.assertEqual(
                    z.read("SARBF-Net/" + run["checkpoint"]["path"]),
                    b"trained-checkpoint",
                )
                self.assertEqual(
                    z.read("SARBF-Net/" + run["q_input"]["path"]), b"original-q"
                )
                self.assertEqual(
                    json.loads(z.read("SARBF-Net/" + run["result_path"])), record
                )
                self.assertEqual(len([n for n in names if n.endswith(".npy")]), 1)

    def test_modified_checkpoint_causes_error_and_prevents_bundle(self):
        m = collector()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, r = fixture(root)
            Path(r["checkpoint"]["path"]).write_bytes(b"wrong weights")
            plan = m.scan([root])
            self.assertTrue(plan["errors"])
            with self.assertRaisesRegex(ValueError, "collection errors"):
                m.write_release(plan, root / "bad.zip", None)
            self.assertFalse((root / "bad.zip").exists())

    def test_different_duplicate_seed_is_rejected_not_best_score_selected(self):
        m = collector()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root, name="first")
            fixture(root, name="second", checkpoint=b"different")
            plan = m.scan([root])
            self.assertTrue(any("ambiguous" in e for e in plan["errors"]))

    def test_identical_duplicate_and_relocated_checkpoint_are_handled(self):
        m = collector()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, r = fixture(root, name="first")
            r["checkpoint"]["path"] = "/old/server/path/checkpoint.pth"
            path.write_text(json.dumps(r))
            fixture(root, name="second")
            plan = m.scan([root])
            self.assertEqual(plan["errors"], [])
            self.assertEqual(len(plan["runs"]), 1)

    def test_unsupported_model_and_failed_runs_are_not_included(self):
        m = collector()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, r = fixture(root)
            r["args"]["model"] = "AnotherModel"
            path.write_text(json.dumps(r))
            plan = m.scan([root])
            self.assertEqual(plan["runs"], [])
            self.assertTrue(plan["skipped"])

    def test_different_data_or_protocol_is_not_silently_deduplicated(self):
        m = collector()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root, name="first")
            path, r = fixture(root, name="second")
            r["data_sha256"] = "c" * 64
            r["args"]["enc_in"] = 21
            path.write_text(json.dumps(r))
            self.assertTrue(any("ambiguous" in e for e in m.scan([root])["errors"]))

    def test_unrelated_json_shapes_do_not_abort_a_broad_scan(self):
        m = collector()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root, name="valid")
            for name, value in [
                ("list", []),
                ("args", dict(status="COMPLETE", arm="sa_lcm", args="invalid")),
                ("arm", dict(status="COMPLETE", arm=[], args=dict(model="SA_LCM"))),
                ("model", dict(status="COMPLETE", arm="sa_lcm", args=dict(model=[]))),
            ]:
                p = root / name / "result.json"
                p.parent.mkdir()
                p.write_text(json.dumps(value))
            plan = m.scan([root])
            self.assertEqual(len(plan["runs"]), 1)
            self.assertGreaterEqual(len(plan["skipped"]), 4)

    def test_bundled_source_manifest_matches_included_source_files(self):
        m = collector()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            source = root / "source"
            source.mkdir()
            (source / "README.md").write_text("SARBF-Net")
            (source / "dataset").mkdir()
            (source / "dataset/README.md").write_text("Dataset instructions")
            (source / "RELEASE_MANIFEST.json").write_text(
                json.dumps(dict(files={"dataset/README.md": "old-digest"}))
            )
            output = root / "release.zip"
            m.write_release(m.scan([root]), output, source)
            with zipfile.ZipFile(output) as z:
                manifest = json.loads(z.read("SARBF-Net/RELEASE_MANIFEST.json"))
                for name, digest in manifest["files"].items():
                    self.assertEqual(
                        hashlib.sha256(z.read("SARBF-Net/" + name)).hexdigest(), digest
                    )
                self.assertIn("SARBF-Net/dataset/README.md", z.namelist())

    def test_nonfinite_metrics_are_rejected(self):
        m = collector()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, r = fixture(root)
            r["test"]["mse"] = float("nan")
            path.write_text(json.dumps(r))
            self.assertTrue(m.scan([root])["errors"])

    def test_source_symlink_and_checkpoint_overwrite_are_not_followed(self):
        m = collector()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            source = root / "source"
            source.mkdir()
            (source / "README.md").write_text("SARBF-Net")
            secret = root / "outside.txt"
            secret.write_text("outside")
            (source / "outside.txt").symlink_to(secret)
            plan = m.scan([root])
            output = root / "release.zip"
            m.write_release(plan, output, source)
            with zipfile.ZipFile(output) as z:
                self.assertNotIn("SARBF-Net/outside.txt", z.namelist())
            with self.assertRaises(FileExistsError):
                m.write_release(plan, output, source)


class PortableReplayTests(unittest.TestCase):
    def test_replay_resolves_portable_paths_and_rejects_tampered_files(self):
        from tools.sarbf_net.evaluate import prepare_replay

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, record = fixture(root / "original")
            raw = Path(record["args"]["root_path"]) / "ETTm1.csv"
            record["data_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
            path.write_text(json.dumps(record))
            module = collector()
            plan = module.scan([root / "original"])
            output = root / "export.zip"
            module.write_release(plan, output, None)
            with zipfile.ZipFile(output) as z:
                z.extractall(root / "extract")
            repository = root / "extract/SARBF-Net"
            data = root / "datasets/ETT-small"
            data.mkdir(parents=True)
            (data / "ETTm1.csv").write_bytes(raw.read_bytes())
            args, checkpoint, run, original = prepare_replay(
                repository, root / "datasets", "ETTm1-96", 2021, "cpu", 0
            )
            self.assertEqual(args["model"], "SARBF_Net")
            self.assertEqual(args["num_workers"], 0)
            self.assertTrue(Path(args["q_mat_file"]).is_file())
            self.assertTrue(checkpoint.is_file())
            checkpoint.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                prepare_replay(
                    repository, root / "datasets", "ETTm1-96", 2021, "cpu", 0
                )

    def test_replay_rejects_a_manifest_pointing_to_an_unrelated_q(self):
        from tools.sarbf_net.evaluate import prepare_replay

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, record = fixture(root / "original")
            raw = Path(record["args"]["root_path"]) / "ETTm1.csv"
            record["data_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
            path.write_text(json.dumps(record))
            module = collector()
            output = root / "export.zip"
            module.write_release(module.scan([root / "original"]), output, None)
            with zipfile.ZipFile(output) as z:
                z.extractall(root / "extract")
            repository = root / "extract/SARBF-Net"
            data = root / "datasets/ETT-small"
            data.mkdir(parents=True)
            (data / "ETTm1.csv").write_bytes(raw.read_bytes())
            other = b"unrelated q"
            digest = hashlib.sha256(other).hexdigest()
            (repository / "artifacts/q" / f"{digest}.npy").write_bytes(other)
            mp = repository / "artifacts/manifest.json"
            manifest = json.loads(mp.read_text())
            manifest["runs"][0]["q_input"] = {
                "path": f"artifacts/q/{digest}.npy",
                "sha256": digest,
            }
            mp.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "original record"):
                prepare_replay(
                    repository, root / "datasets", "ETTm1-96", 2021, "cpu", 0
                )

    def test_replay_requires_the_original_dataset_bytes(self):
        from tools.sarbf_net.evaluate import prepare_replay

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, record = fixture(root / "original")
            module = collector()
            plan = module.scan([root / "original"])
            output = root / "export.zip"
            module.write_release(plan, output, None)
            with zipfile.ZipFile(output) as z:
                z.extractall(root / "extract")
            data = root / "datasets/ETT-small"
            data.mkdir(parents=True)
            (data / "ETTm1.csv").write_text("different")
            with self.assertRaisesRegex(ValueError, "dataset SHA256"):
                prepare_replay(
                    root / "extract/SARBF-Net",
                    root / "datasets",
                    "ETTm1-96",
                    2021,
                    "cpu",
                    0,
                )


class ExpandedCollectionTests(unittest.TestCase):
    def test_additional_traffic_datasets_keep_recorded_assets(self):
        m = collector()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for task, channels in [("PEMSD7-12", 228), ("PEMS-BAY-24", 325)]:
                path, r = fixture(root, name=task)
                r["task"] = task
                r["args"].update(
                    pred_len=int(task.rsplit("-", 1)[1]),
                    enc_in=channels,
                    data="PEMS",
                    data_path=task.rsplit("-", 1)[0] + ".npz",
                )
                path.write_text(json.dumps(r))
            plan = m.scan([root])
            self.assertEqual(plan["errors"], [])
            self.assertEqual(len(plan["runs"]), 2)

    def test_depth_and_main_results_are_kept_separate_in_manifest_and_summary(self):
        import csv
        import io

        m = collector()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            main = root / "main"
            depth = root / "depth"
            fixture(main, name="seed2021/ETTm1-96/sa_lcm")
            for k, score in [("K3", 0.6), ("K4", 0.8)]:
                path, r = fixture(
                    depth, name=k + "/seed2021/ETTm1-96/sa_lcm", checkpoint=k.encode()
                )
                r["args"]["e_layers"] = int(k[1:])
                r["test"]["mse"] = score
                path.write_text(json.dumps(r))
            plan = m.scan_groups([main], [("depth", depth)])
            self.assertEqual(plan["errors"], [])
            self.assertEqual(len(plan["runs"]), 3)
            self.assertEqual(
                len({e["public"]["checkpoint"]["path"] for e in plan["runs"]}), 3
            )
            _, summary = m.summaries(plan["runs"])
            rows = list(csv.DictReader(io.StringIO(summary)))
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                [float(r["mse_mean"]) for r in rows if r["study"] == "main"], [0.4]
            )
            manifest = m.write_release(plan, root / "release.zip", None)
            self.assertEqual(
                {(r["study"], r["variant"]) for r in manifest["runs"]},
                {("main", "standard"), ("depth", "K3"), ("depth", "K4")},
            )

    def test_conflicts_within_one_depth_variant_still_stop_collection(self):
        m = collector()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root / "first", name="K3/seed2021/ETTm1-96/sa_lcm")
            fixture(
                root / "second", name="K3/seed2021/ETTm1-96/sa_lcm", checkpoint=b"other"
            )
            plan = m.scan_groups(
                [], [("depth", root / "first"), ("depth", root / "second")]
            )
            self.assertTrue(any("ambiguous" in e for e in plan["errors"]))

    def test_disk_space_check_stops_before_creating_a_large_output(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        m = collector()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            plan = m.scan([root])
            with patch.object(
                m.shutil, "disk_usage", return_value=SimpleNamespace(free=0)
            ):
                with self.assertRaisesRegex(OSError, "space"):
                    m.write_release(plan, root / "output.zip", None)
            self.assertFalse((root / "output.zip").exists())


class StudyReplayTests(unittest.TestCase):
    def test_replay_selects_main_or_depth_explicitly(self):
        from tools.sarbf_net.evaluate import prepare_replay

        m = collector()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            main = root / "main"
            depth = root / "depth"
            records = [
                fixture(main, name="seed2021/ETTm1-96/sa_lcm"),
                fixture(depth, name="K3/seed2021/ETTm1-96/sa_lcm", checkpoint=b"K3"),
            ]
            for path, r in records:
                raw = Path(r["args"]["root_path"]) / "ETTm1.csv"
                r["data_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
                path.write_text(json.dumps(r))
            plan = m.scan_groups([main], [("depth", depth)])
            output = root / "release.zip"
            m.write_release(plan, output, None)
            with zipfile.ZipFile(output) as z:
                z.extractall(root / "extract")
            repository = root / "extract/SARBF-Net"
            data = root / "data/ETT-small"
            data.mkdir(parents=True)
            (data / "ETTm1.csv").write_text("raw data must not be bundled")
            args, weight, run, _ = prepare_replay(
                repository, root / "data", "ETTm1-96", 2021
            )
            self.assertEqual(run["study"], "main")
            self.assertEqual(weight.read_bytes(), b"trained-checkpoint")
            args, weight, run, _ = prepare_replay(
                repository, root / "data", "ETTm1-96", 2021, study="depth", variant="K3"
            )
            self.assertEqual(weight.read_bytes(), b"K3")


if __name__ == "__main__":
    unittest.main()
