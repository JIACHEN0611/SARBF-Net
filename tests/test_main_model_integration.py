import importlib
import tempfile
import unittest
from pathlib import Path

import torch



class MainIntegrationTests(unittest.TestCase):
    def test_main_runtime_uses_components_and_legacy_strict_loads(self):
        from test_final_release import FinalReleaseNumericalTests
        from model.SARBF_Net import Model
        self.assertTrue(any(c.__module__ == 'model.components.folded_head_model'
                            for c in Model.__mro__), 'Main runtime still depends on legacy modules')
        with tempfile.TemporaryDirectory() as folder:
            cfg = FinalReleaseNumericalTests()._configs(Path(folder), horizon=7)
            torch.manual_seed(2021)
            model = Model(cfg)
            for name in ['OLinear_SALCM_BCIF_OutputFold', 'SA_LCM']:
                legacy = importlib.import_module('model.' + name).Model(cfg)
                if name == 'SA_LCM':
                    legacy.load_state_dict(model.state_dict(), strict=True)
            self.assertIs(model.saer, model.sa_lcm)
            self.assertIs(model.encoder, model.ortho_trans[1])
            self.assertNotIn('Q_mat', model.state_dict())
