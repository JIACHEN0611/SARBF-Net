"""Formula, model, and runner contracts for the two BCIF ablations."""

import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Static release checks also run without PyTorch.
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    update = torch.tensor(
        [[[1.0, -0.5], [0.2, 1.5], [-0.7, 0.4]]], dtype=torch.float32
    )
    history_mass = torch.tensor([[0.2, 0.7, 0.4]], dtype=torch.float32)
    prior = torch.tensor(
        [[[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]]],
        dtype=torch.float32,
    )
    next_prior = torch.tensor(
        [[[0.5, 0.4, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]]],
        dtype=torch.float32,
    )
    next_hidden = torch.tensor(
        [[[1.0, 2.0], [0.5, -1.5], [2.0, 0.25]]], dtype=torch.float32
    )
    return update, history_mass, prior, next_prior, next_hidden


def _identity_projection(module):
    with torch.no_grad():
        module.down.weight.copy_(torch.eye(2))
        module.up.weight.copy_(torch.eye(2))


def _bounded(module, discrepancy, next_hidden):
    scale = next_hidden.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    return module.cap * scale * torch.tanh(discrepancy / scale)


@unittest.skipIf(torch is None, "PyTorch numerical checks run in the server environment")
class CoordinatorFormulaTests(unittest.TestCase):
    def test_no_mass_uses_balanced_p_next_n_minus_n_and_ignores_external_mass(self):
        from layers.BalancedCrossIntraFeedback import BCIFNoMassCoordinator

        update, mass, prior, next_prior, next_hidden = _inputs()
        module = BCIFNoMassCoordinator(d_model=2, rank=2, cap=0.1)
        _identity_projection(module)

        centered = update - update[:, :1, :]
        novelty = centered - torch.matmul(prior, centered)
        discrepancy = torch.matmul(next_prior, novelty) - novelty
        expected = _bounded(module, discrepancy, next_hidden)

        actual = module(update, mass, prior, next_prior, next_hidden)
        changed_mass = module(update, 1.0 - mass, prior, next_prior, next_hidden)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(changed_mass, expected)

    def test_unbalanced_is_exactly_p_next_of_mass_weighted_novelty(self):
        from layers.BalancedCrossIntraFeedback import BCIFUnbalancedCoordinator

        update, mass, prior, next_prior, next_hidden = _inputs()
        module = BCIFUnbalancedCoordinator(d_model=2, rank=2, cap=0.1)
        _identity_projection(module)

        centered = update - update[:, :1, :]
        novelty = centered - torch.matmul(prior, centered)
        discrepancy = torch.matmul(next_prior, mass.unsqueeze(-1) * novelty)
        expected = _bounded(module, discrepancy, next_hidden)

        actual = module(update, mass, prior, next_prior, next_hidden)
        torch.testing.assert_close(actual, expected)

    def test_both_ablation_coordinators_keep_zero_initial_feedback_and_up_gradient(self):
        from layers.BalancedCrossIntraFeedback import (
            BCIFNoMassCoordinator,
            BCIFUnbalancedCoordinator,
        )

        values = _inputs()
        for coordinator in (BCIFNoMassCoordinator, BCIFUnbalancedCoordinator):
            with self.subTest(coordinator=coordinator.__name__):
                module = coordinator(d_model=2, rank=2, cap=0.1)
                output = module(*values)
                self.assertEqual(torch.count_nonzero(output).item(), 0)
                output.sum().backward()
                self.assertIsNotNone(module.up.weight.grad)
                self.assertGreater(module.up.weight.grad.abs().sum().item(), 0)


@unittest.skipIf(torch is None, "PyTorch model checks run in the server environment")
class ModelContractTests(unittest.TestCase):
    def test_models_subclass_output_fold_and_select_only_the_new_coordinator(self):
        from layers.BalancedCrossIntraFeedback import (
            BCIFNoMassCoordinator,
            BCIFUnbalancedCoordinator,
        )
        from model.OLinear_SALCM_BCIF_OutputFold import Model as OutputFold
        from ablations.models.OLinear_SALCM_BCIF_OutputFold_NoMass import Model as NoMass
        from ablations.models.OLinear_SALCM_BCIF_OutputFold_Unbalanced import Model as Unbalanced

        self.assertTrue(issubclass(NoMass, OutputFold))
        self.assertTrue(issubclass(Unbalanced, OutputFold))
        self.assertIs(NoMass.coordinator_type, BCIFNoMassCoordinator)
        self.assertIs(Unbalanced.coordinator_type, BCIFUnbalancedCoordinator)

class RunnerContractTests(unittest.TestCase):
    def test_runner_registers_separate_revision_locked_arms(self):
        from tools.sa_lcm_transport.run_pilot import (
            ARCHITECTURES,
            SAFE_MODELS,
        )

        self.assertEqual(
            SAFE_MODELS["no_mass_bcif"],
            "OLinear_SALCM_BCIF_OutputFold_NoMass",
        )
        self.assertEqual(
            SAFE_MODELS["unbalanced_bcif"],
            "OLinear_SALCM_BCIF_OutputFold_Unbalanced",
        )
        self.assertEqual(ARCHITECTURES["no_mass_bcif"], "SA_LCM_BCIF_NO_MASS_V1")
        self.assertEqual(
            ARCHITECTURES["unbalanced_bcif"],
            "SA_LCM_BCIF_UNBALANCED_V1",
        )

        experiment_source = (ROOT / "experiments/exp_basic.py").read_text()
        run_source = (ROOT / "run.py").read_text()
        for model in SAFE_MODELS.values():
            self.assertIn(model, experiment_source)
            self.assertIn(model, run_source)


if __name__ == "__main__":
    unittest.main()
