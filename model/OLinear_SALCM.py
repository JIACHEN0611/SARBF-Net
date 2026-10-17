"""SA-LCM V1 integration with the unchanged Official OLinear body."""

import torch

from layers.SelectiveAnchoredLongContext import (
    SelectiveAnchoredLongContext,
    anchor_normalize_past,
)
from model.OLinear import Model as OfficialOLinear


class Model(OfficialOLinear):
    def __init__(self, configs):
        if int(configs.seq_len) != 96:
            raise ValueError("SA-LCM V1 requires model seq_len=96")
        if int(getattr(configs, "long_len", 768)) != 768:
            raise ValueError("SA-LCM V1 requires long_len=768")
        super().__init__(configs)
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            self.sa_lcm = SelectiveAnchoredLongContext(hidden_dim=64)
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    def forward(
        self,
        x,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
    ):
        if x.ndim != 3 or x.shape[1] != 768:
            raise ValueError("OLinear_SALCM expects [B,768,C]")
        past = x[:, :672, :]
        recent = x[:, -96:, :]

        # Only the real Recent96 establishes Official RevIN statistics.
        z_recent = self.revin_layer(recent, mode="norm")
        z_past = anchor_normalize_past(past, self.revin_layer)
        z_refined = self.sa_lcm(z_recent, z_past)

        encoded = self.tokenEmb(z_refined, self.embeddings)
        encoded = self.Fre_Trans(encoded)
        output = self.fc(encoded.flatten(-2)).transpose(-1, -2)
        output = self.dropout(output)
        return self.revin_layer(output, mode="denorm")
