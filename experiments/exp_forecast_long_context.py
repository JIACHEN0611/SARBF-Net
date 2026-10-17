import os

from data_provider.long_context_factory import long_context_data_provider
from experiments.exp_forecast import Exp_Forecast


class Exp_Forecast_LongContext(Exp_Forecast):
    def train(self, setting=None):
        # The locked upstream trainer requests a test loader every epoch.
        # Redirect that request to validation during pruning experiments,
        # then restore actual test access for final checkpoint replay.
        self._prune_training = os.environ.get("SA_LCM_PRUNE_TEST_BLIND") == "1"
        try:
            return super().train(setting)
        finally:
            self._prune_training = False

    def _get_data(self, flag=None, test_batch_size=None):
        if flag == "test" and getattr(self, "_prune_training", False):
            print("SA_LCM_PRUNE_TEST_BLIND: training Test Loss repeats validation; "
                  "test loader not opened", flush=True)
            flag = "valid"
        return long_context_data_provider(
            self.args,
            flag,
            test_batch_size=test_batch_size,
        )
