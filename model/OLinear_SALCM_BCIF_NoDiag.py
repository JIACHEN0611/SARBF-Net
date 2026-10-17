"""Original BCIF model without unused per-batch frontend scalar reporting."""

import torch

from layers.SafePruneFrontend import SafePruneFrontend
from model.OLinear_SALCM_BCIF import Model as BCIFModel


class Model(BCIFModel):
    def __init__(self, configs):
        super().__init__(configs)
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            replacement = SafePruneFrontend(hidden_dim=64)
            replacement.load_state_dict(self.sa_lcm.state_dict(), strict=True)
            self.sa_lcm = replacement
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
