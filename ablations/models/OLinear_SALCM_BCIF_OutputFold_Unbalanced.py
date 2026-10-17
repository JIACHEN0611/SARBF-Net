"""Output-fold SA-LCM whose BCIF omits balanced self-cancellation."""

from layers.BalancedCrossIntraFeedback import BCIFUnbalancedCoordinator
from model.OLinear_SALCM_BCIF_OutputFold import Model as OutputFoldModel


class Model(OutputFoldModel):
    coordinator_type = BCIFUnbalancedCoordinator
