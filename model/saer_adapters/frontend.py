"""Raw-scale, backbone-agnostic SA-LCM input frontend."""

from torch import nn

from model.saer_adapters._layers.CosineHistoricalSelector import CosineHistoricalSelector
from model.backbones._layers.RevIN import RevIN
from model.saer_adapters._layers.SelectiveAnchoredLongContext import (
    SelectiveAnchoredLongContext,
    anchor_normalize_past,
)


class SALCMInputFrontend(nn.Module):
    """Refine a 96-step recent window using seven preceding 96-step blocks."""

    def __init__(self, enc_in, long_len=768, seq_len=96):
        super().__init__()
        if int(enc_in) <= 0:
            raise ValueError("enc_in must be positive")
        if int(long_len) != 768 or int(seq_len) != 96:
            raise ValueError("SA-LCM frontend requires long_len=768 and seq_len=96")
        self.enc_in = int(enc_in)
        self.long_len = int(long_len)
        self.seq_len = int(seq_len)
        self.revin = RevIN(self.enc_in, affine=True)
        self.refiner = SelectiveAnchoredLongContext(hidden_dim=64)
        self.refiner.selector = CosineHistoricalSelector.from_existing(
            self.refiner.selector
        )

    def forward(self, long_x, return_diagnostics=False):
        if (
            long_x.ndim != 3
            or long_x.shape[1] != self.long_len
            or long_x.shape[2] != self.enc_in
        ):
            raise ValueError(
                f"SALCMInputFrontend expects [B,{self.long_len},{self.enc_in}]"
            )
        past = long_x[:, : self.long_len - self.seq_len, :]
        recent = long_x[:, -self.seq_len :, :]
        z_recent = self.revin(recent, mode="norm")
        z_past = anchor_normalize_past(past, self.revin)
        refined = self.refiner(z_recent, z_past)
        output = self.revin(refined, mode="denorm")
        if return_diagnostics:
            return output, self.refiner.last_diagnostics()
        return output
