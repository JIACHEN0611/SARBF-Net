"""Output-fold algebra, strict migration, and validation-screen contracts."""

import ast
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests.test_slim_release import ROOT, validated_record


class OutputFoldAlgebraTests(unittest.TestCase):
    def test_non_square_embedding_and_nonzero_delta_match_unfolded_head(self):
        rng = np.random.default_rng(123)
        batch, channels, horizon, embedding, width, feedforward = 2, 4, 5, 3, 7, 11
        state = rng.normal(size=(batch, channels, width))
        w = rng.normal(size=(horizon * embedding, width))
        ortho_bias = rng.normal(size=horizon * embedding)
        q = rng.normal(size=(horizon, horizon))
        a = rng.normal(size=(feedforward, horizon * embedding))
        head_bias = rng.normal(size=feedforward)
        delta = rng.normal(size=(channels, horizon))

        # Independently evaluate the old graph including its horizon/embedding order.
        old_projection = (state @ w.T + ortho_bias).reshape(batch, channels, embedding, horizon)
        old_output = old_projection @ q + delta[None, :, None, :]
        old_preactivation = old_output.transpose(0, 1, 3, 2).reshape(
            batch, channels, horizon * embedding
        ) @ a.T + head_bias

        # The new graph has a [width, feedforward] core and a horizon-only delta head.
        tensor = a.reshape(feedforward, horizon, embedding)
        folded_mix = np.einsum("fve,tv->fet", tensor, q)
        core_weight = np.einsum("fet,etd->fd", folded_mix,
                                w.reshape(embedding, horizon, width))
        core_bias = head_bias + np.einsum(
            "fet,et->f", folded_mix, ortho_bias.reshape(embedding, horizon)
        )
        delta_weight = tensor.sum(axis=2)
        folded_preactivation = (state @ core_weight.T + core_bias
                                + delta @ delta_weight.T)
        np.testing.assert_allclose(folded_preactivation, old_preactivation,
                                   rtol=2e-13, atol=2e-13)


class OutputFoldRegistryTests(unittest.TestCase):
    def test_registry_accepts_all_fresh_arms_with_distinct_revisions(self):
        from tools.sa_lcm_transport.run_pilot import (
            ARMS, FINAL_REVISION, NO_MASS_REVISION, OUTPUT_FOLD_REVISION,
            SAFE_REVISION, SAFE_MODELS,
            UNBALANCED_REVISION,
            assess_result, assess_legacy_bcif_reference, run_task,
        )

        self.assertEqual(ARMS, (
            "sa_lcm", "no_diag", "output_fold", "no_mass_bcif",
            "unbalanced_bcif", "pair_free_fold", "sarbf_net",
        ))
        self.assertEqual(SAFE_MODELS["output_fold"],
                         "OLinear_SALCM_BCIF_OutputFold")
        self.assertNotEqual(OUTPUT_FOLD_REVISION, SAFE_REVISION)
        self.assertNotEqual(OUTPUT_FOLD_REVISION, FINAL_REVISION)
        fold = validated_record("output_fold", SAFE_MODELS["output_fold"],
                                OUTPUT_FOLD_REVISION)
        self.assertEqual(assess_result(fold), "VALIDATED")
        self.assertEqual(assess_result(validated_record(
            "no_diag", SAFE_MODELS["no_diag"], SAFE_REVISION)), "VALIDATED")
        self.assertEqual(assess_result(validated_record(
            "no_mass_bcif", SAFE_MODELS["no_mass_bcif"], NO_MASS_REVISION)),
            "VALIDATED")
        self.assertEqual(assess_result(validated_record(
            "unbalanced_bcif", SAFE_MODELS["unbalanced_bcif"],
            UNBALANCED_REVISION)), "VALIDATED")
        self.assertEqual(assess_result(validated_record(
            "sa_lcm", SAFE_MODELS["sa_lcm"], FINAL_REVISION)), "VALIDATED")
        self.assertEqual(assess_result(validated_record(
            "pair_free_fold", SAFE_MODELS["pair_free_fold"],
            FINAL_REVISION)), "VALIDATED")
        self.assertEqual(assess_legacy_bcif_reference(fold), "MALFORMED")
        for field, value in (("architecture_revision", SAFE_REVISION),
                             ("test", {"mse": 0.0}),
                             ("evaluation_scope", "complete")):
            bad = copy.deepcopy(fold)
            bad[field] = value
            self.assertEqual(assess_result(bad), "MALFORMED", field)
        with self.assertRaisesRegex(ValueError, "not registered"):
            run_task("ETTh1-720", ROOT, ROOT / "runs", arm="bcif", validation_only=True)

    def test_new_source_is_locked_and_every_shipped_python_file_is_hashed(self):
        from tools.sa_lcm_transport.run_pilot import SOURCE_FILES, source_tree_digest

        model = "model/OLinear_SALCM_BCIF_OutputFold.py"
        self.assertIn(model, SOURCE_FILES)
        self.assertEqual(set(SOURCE_FILES), {
            str(path.relative_to(ROOT)) for path in ROOT.rglob("*.py")
            if "__pycache__" not in path.parts
        })
        hashes = json.loads((ROOT / "MODEL_LAYER_SHA256.json").read_text())
        self.assertIn(model, hashes)
        import hashlib
        self.assertEqual(hashes[model], hashlib.sha256((ROOT / model).read_bytes()).hexdigest())
        self.assertEqual(len(source_tree_digest(ROOT)), 64)

    def test_entrypoints_route_new_arm_through_long_context(self):
        source = (ROOT / "run.py").read_text()
        self.assertIn("SA_LCM", source)
        self.assertIn("OLinear_SALCM_BCIF_OutputFold", source)
        tree = ast.parse((ROOT / "experiments/exp_basic.py").read_text())
        imported = {alias.name for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module == "model"
                    for alias in node.names}
        from model.legacy.registry import LEGACY_MODELS
        self.assertEqual(LEGACY_MODELS["OLinear_SALCM_BCIF_OutputFold"].__name__,
                         "model.components.folded_head_model")
        self.assertIn("SA_LCM", imported)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch required on the server")
class OutputFoldTorchTests(unittest.TestCase):
    def _configs(self, folder, q=None):
        q = np.eye(5, dtype=np.float32) if q is None else q
        np.save(folder / "input.npy", np.eye(96, dtype=np.float32))
        np.save(folder / "output.npy", q.astype(np.float32))
        return SimpleNamespace(seq_len=96, long_len=768, pred_len=5, enc_in=3,
                               d_model=8, d_ff=16, embed_size=3, e_layers=2,
                               temp_patch_len=16, temp_stride=8, CKA_flag=0,
                               dropout=0.0, activation="gelu", Q_chan_indep=0,
                               q_mat_file="input.npy", q_out_mat_file="output.npy",
                               root_path=str(folder))

    def test_constructor_preserves_rng_and_full_inference_after_strict_migration(self):
        import torch
        from model.OLinear_SALCM_BCIF_NoDiag import Model as Old
        from model.OLinear_SALCM_BCIF_OutputFold import (
            Model as Fold, convert_old_bcif_state_dict,
        )

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            cfg = self._configs(folder, np.random.default_rng(2021).normal(
                size=(5, 5)).astype(np.float32))
            torch.manual_seed(371)
            old = Old(cfg)
            old.delta2.data.copy_(torch.randn_like(old.delta2))
            weights = copy.deepcopy(old.state_dict())

            torch.manual_seed(371)
            expected_rng = Old(cfg)
            del expected_rng
            cpu_expected = torch.random.get_rng_state().clone()
            cuda_expected = (torch.cuda.get_rng_state_all() if torch.cuda.is_available()
                             else None)
            torch.manual_seed(371)
            folded = Fold(cfg)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), cpu_expected))
            if cuda_expected is not None:
                self.assertTrue(all(torch.equal(left, right) for left, right in
                                    zip(torch.cuda.get_rng_state_all(), cuda_expected)))
            converted = convert_old_bcif_state_dict(folded, weights)
            self.assertEqual(set(converted), set(folded.state_dict()))
            self.assertIn("ortho_trans.1.norm.weight", converted)
            self.assertIn("encoder.norm.weight", converted)
            folded.load_state_dict(converted, strict=True)
            self.assertIs(folded.ortho_trans[1], folded.encoder)
            self.assertEqual(folded.fc[0].in_features, cfg.d_model)
            self.assertEqual(folded.delta_head.in_features, cfg.pred_len)
            self.assertEqual(folded.delta_head.out_features, cfg.d_ff)
            self.assertEqual(folded.delta2.shape[-1], cfg.pred_len)
            self.assertIsInstance(folded.ortho_trans[2], torch.nn.Identity)
            old = old.eval().to(old.Q_out_mat.device)
            folded = folded.eval().to(folded.Q_out_mat.device)
            x = torch.randn(2, 768, cfg.enc_in, device=old.Q_out_mat.device)
            # The loaded Q_out was folded at conversion; it must not be read
            # again in the new forward graph.
            folded.Q_out_mat = torch.full_like(folded.Q_out_mat, float("nan"))
            with torch.no_grad():
                difference = (old(x) - folded(x)).abs().max().item()
            self.assertLess(difference, 2e-4)

    def test_small_tensor_prediction_and_gradients_match_old_affine_chain(self):
        import torch
        from torch.nn import functional as F
        from model.OLinear_SALCM_BCIF_OutputFold import _fold_head

        torch.manual_seed(208)
        batch, channels, horizon, embedding, width, feedforward = 2, 3, 7, 4, 9, 13
        def parameter(*shape):
            return torch.randn(*shape, dtype=torch.float64, requires_grad=True)

        x = parameter(batch, channels, width)
        ortho_weight = parameter(horizon * embedding, width)
        ortho_bias = parameter(horizon * embedding)
        head_weight = parameter(feedforward, horizon * embedding)
        head_bias = parameter(feedforward)
        delta = parameter(1, channels, 1, horizon)
        final_weight = parameter(horizon, feedforward)
        final_bias = parameter(horizon)
        q = torch.randn(horizon, horizon, dtype=torch.float64)
        model = SimpleNamespace(pred_len=horizon, seq_len=7, Q_mat=torch.eye(7),
                                embed_size=embedding, d_model=width,
                                d_ff=feedforward, Q_chan_indep=0, enc_in=channels,
                                Q_out_mat=q, delta2=delta)
        old = F.linear(x, ortho_weight, ortho_bias).reshape(
            batch, channels, embedding, horizon
        )
        old = torch.einsum("bcet,tv->bcev", old, q) + delta
        old = old.transpose(-1, -2).reshape(batch, channels, horizon * embedding)
        old_output = F.linear(F.gelu(F.linear(old, head_weight, head_bias)),
                              final_weight, final_bias)
        core_weight, core_bias, delta_weight = _fold_head(
            model, ortho_weight, ortho_bias, head_weight, head_bias)
        folded_output = F.linear(
            F.gelu(F.linear(x, core_weight, core_bias)
                   + F.linear(delta.squeeze(2), delta_weight)),
            final_weight, final_bias)
        torch.testing.assert_close(old_output, folded_output, rtol=1e-12, atol=1e-12)
        parameters = (x, ortho_weight, ortho_bias, head_weight, head_bias,
                      delta, final_weight, final_bias)
        loss_weights = torch.randn_like(old_output)
        (old_output * loss_weights).sum().backward(retain_graph=True)
        old_grads = [tensor.grad.clone() for tensor in parameters]
        for tensor in parameters:
            tensor.grad = None
        (folded_output * loss_weights).sum().backward()
        for old_grad, tensor in zip(old_grads, parameters):
            torch.testing.assert_close(old_grad, tensor.grad, rtol=1e-11, atol=1e-11)

    def test_migration_rejects_extra_missing_misaligned_and_nonfinite_q(self):
        import torch
        from model.OLinear_SALCM_BCIF_NoDiag import Model as Old
        from model.OLinear_SALCM_BCIF_OutputFold import (
            Model as Fold, convert_old_bcif_state_dict,
        )

        with tempfile.TemporaryDirectory() as directory:
            cfg = self._configs(Path(directory))
            old = Old(cfg)
            folded = Fold(cfg)
            state = old.state_dict()
            for altered in (dict(state, unexpected=torch.tensor(1.0)),
                            {key: value for key, value in state.items()
                             if key != "fc.0.weight"}):
                with self.assertRaisesRegex(ValueError, "keys"):
                    convert_old_bcif_state_dict(folded, altered)
            altered = dict(state, **{"fc.0.weight": state["fc.0.weight"][:2]})
            with self.assertRaisesRegex(ValueError, "shape"):
                convert_old_bcif_state_dict(folded, altered)
            alias = "ortho_trans.1.norm.weight"
            altered = dict(state, **{alias: state[alias] + 1})
            with self.assertRaisesRegex(ValueError, "alias"):
                convert_old_bcif_state_dict(folded, altered)
            original_q = folded.Q_out_mat
            folded.Q_out_mat = original_q[:4, :4]
            with self.assertRaisesRegex(ValueError, "Q_out"):
                convert_old_bcif_state_dict(folded, state)
            folded.Q_out_mat = original_q
            folded.Q_mat[0, 0] = float("nan")
            with self.assertRaisesRegex(ValueError, "Q_mat"):
                convert_old_bcif_state_dict(folded, state)
            folded.Q_mat[0, 0] = 1.0
            folded.Q_out_mat[0, 0] = float("nan")
            with self.assertRaisesRegex(ValueError, "Q_out"):
                convert_old_bcif_state_dict(folded, state)

    def test_constructor_rejects_channel_independent_q(self):
        from model.OLinear_SALCM_BCIF_OutputFold import Model as Fold
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._configs(Path(directory))
            cfg.Q_chan_indep = 1
            with self.assertRaisesRegex(ValueError, "Q_chan_indep=0"):
                Fold(cfg)


if __name__ == "__main__":
    unittest.main()
