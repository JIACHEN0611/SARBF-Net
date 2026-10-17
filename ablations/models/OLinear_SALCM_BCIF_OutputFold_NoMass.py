"""Output-fold SA-LCM whose BCIF ignores external history confidence."""

from layers.BalancedCrossIntraFeedback import BCIFNoMassCoordinator
from model.OLinear_SALCM_BCIF_OutputFold import Model as OutputFoldModel


class Model(OutputFoldModel):
    coordinator_type = BCIFNoMassCoordinator
