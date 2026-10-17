"""SARBF-Net: Selective Anchored Retrieval & Balanced Feedback Network."""

from layers.CosineHistoricalSelector import CosineHistoricalSelector
from layers.SelectiveAnchoredLongContext import anchor_normalize_past
from model.OLinear_SALCM_BCIF_OutputFold import Model as OutputFoldModel


class Model(OutputFoldModel):
    """SAER frontend, BCIF interaction and folded prediction head."""

    def __init__(self, configs):
        super().__init__(configs)
        # Replace the learned 16,513-parameter pair scorer without consuming
        # random numbers.  Constructing the inherited graph first preserves
        # the exact seeded initialization used by the selected experiments.
        self.sa_lcm.selector = CosineHistoricalSelector.from_existing(
            self.sa_lcm.selector
        )

    @property
    def saer(self):
        """Frontend under its paper name; keep sa_lcm state keys for old weights."""
        return self.sa_lcm

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x.ndim != 3 or x.shape[1:] != (768, self.enc_in):
            raise ValueError("SARBF_Net expects [B,768,C]")
        recent = x[:, -96:, :]
        z_recent = self.revin_layer(recent, mode="norm")
        z_past = anchor_normalize_past(x[:, :672, :], self.revin_layer)
        refined, history_mass = self.saer.refine_with_mass(z_recent, z_past)
        encoded = self._bcif_transform(
            self.tokenEmb(refined, self.embeddings), history_mass
        )
        preactivation = self.fc[0](encoded) + self.delta_head(self.delta2.squeeze(2))
        prediction = self.fc[2](self.fc[1](preactivation)).transpose(-1, -2)
        return self.revin_layer(self.dropout(prediction), mode="denorm")
