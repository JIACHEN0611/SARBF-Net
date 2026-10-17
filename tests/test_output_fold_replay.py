"""Checkpoint-loading contract for output-fold validation replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch


class FakeParameter:
    def __init__(self, count):
        self.count = count

    def numel(self):
        return self.count


class FakeModel:
    def __init__(self, keys, count):
        self.keys = keys
        self.count = count
        self.loaded = None

    def to(self, _device):
        return self

    def state_dict(self):
        return {key: None for key in self.keys}

    def load_state_dict(self, weights, strict):
        if not strict or set(weights) != set(self.keys):
            raise ValueError("not loaded strictly")
        self.loaded = dict(weights)

    def eval(self):
        return self

    def parameters(self):
        return [FakeParameter(self.count)]


class ReplayCheckpointTests(unittest.TestCase):
    def test_output_fold_uses_converted_checkpoint_while_old_arms_stay_strict(self):
        # Catches loading original keys into the folded architecture, skipping
        # no_diag, and rejecting an intentional parameter-count reduction.
        from tools.sa_lcm_safe_prune import replay
        self.assertTrue(callable(getattr(replay, "_load_replay_models", None)))

        weights = {"linear.weight": 7, "output.weight": 11}
        models = {
            "bcif": FakeModel(weights, 200),
            "no_diag": FakeModel(weights, 200),
            "output_fold": FakeModel(("linear.weight", "folded.weight"), 47),
        }
        names = {arm: f"Example_{arm}" for arm in models}

        def fake_import(name):
            arm = name.removeprefix("model.Example_")
            kwargs = {"Model": lambda _args: models[arm]}
            if arm == "output_fold":
                kwargs["convert_old_bcif_state_dict"] = (
                    lambda _model, state: {
                        "linear.weight": state["linear.weight"],
                        "folded.weight": state["output.weight"] * 2,
                    }
                )
            return SimpleNamespace(**kwargs)

        with patch.object(replay, "MODELS", names), \
             patch.object(replay, "SAFE_MODELS", {arm: names[arm] for arm in ("no_diag", "output_fold")}), \
             patch.object(replay.importlib, "import_module", side_effect=fake_import):
            modules, counts = replay._load_replay_models(
                SimpleNamespace(pred_len=5, embed_size=3, d_model=7, d_ff=11),
                "cpu", weights)

        self.assertEqual(set(modules), set(models))
        self.assertEqual(counts, {"bcif": 200, "no_diag": 200, "output_fold": 47})
        self.assertEqual(models["bcif"].loaded, weights)
        self.assertEqual(models["no_diag"].loaded, weights)
        self.assertEqual(models["output_fold"].loaded,
                         {"linear.weight": 7, "folded.weight": 22})
        self.assertEqual(weights, {"linear.weight": 7, "output.weight": 11})

    def test_output_fold_rejects_incomplete_converted_state(self):
        # Catches an accidental relaxed load that silently leaves new weights
        # initialized randomly instead of coming from the old checkpoint.
        from tools.sa_lcm_safe_prune import replay
        self.assertTrue(callable(getattr(replay, "_load_replay_models", None)))

        old = FakeModel(("old",), 3)
        folded = FakeModel(("folded",), 2)

        def fake_import(name):
            if name == "model.Example_bcif":
                return SimpleNamespace(Model=lambda _args: old)
            return SimpleNamespace(Model=lambda _args: folded,
                                   convert_old_bcif_state_dict=lambda _model, _state: {})

        with patch.object(replay, "MODELS", {"bcif": "Example_bcif", "output_fold": "Example_fold"}), \
             patch.object(replay, "SAFE_MODELS", {"output_fold": "Example_fold"}), \
             patch.object(replay.importlib, "import_module", side_effect=fake_import):
            with self.assertRaisesRegex(ValueError, "output_fold.*keys"):
                replay._load_replay_models(SimpleNamespace(), "cpu", {"old": 5})

    def test_output_fold_rejects_wrong_parameter_reduction(self):
        from tools.sa_lcm_safe_prune import replay

        old = FakeModel(("shared",), 200)
        folded = FakeModel(("shared", "delta_head"), 48)

        def fake_import(name):
            if name == "model.Example_bcif":
                return SimpleNamespace(Model=lambda _args: old)
            return SimpleNamespace(Model=lambda _args: folded,
                                   convert_old_bcif_state_dict=lambda _model, state: {
                                       **state, "delta_head": 0})

        with patch.object(replay, "MODELS", {"bcif": "Example_bcif", "output_fold": "Example_fold"}), \
             patch.object(replay.importlib, "import_module", side_effect=fake_import):
            with self.assertRaisesRegex(ValueError, "parameter reduction"):
                replay._load_replay_models(
                    SimpleNamespace(pred_len=5, embed_size=3, d_model=7, d_ff=11),
                    "cpu", {"shared": 5})


if __name__ == "__main__":
    unittest.main()
