"""SA-LCM with balanced cross-series/intra-series feedback in its encoder."""

import torch
from torch import nn

from layers.BalancedCrossIntraFeedback import BCIFCoordinator, BCIFLinearEncoder
from layers.SelectiveAnchoredLongContext import anchor_normalize_past
from model.OLinear_SALCM import Model as OriginalSALCM


def load_sa_lcm_state(model, state_dict):
    """Require all original SA-LCM keys and allow only BCIF parameters missing."""
    if not isinstance(model, Model):
        raise TypeError("expected OLinear_SALCM_BCIF.Model")
    expected = set(model.state_dict())
    common = {key for key in expected if not key.startswith("coordinators.")}
    received = set(state_dict)
    if received != common:
        raise ValueError("original SA-LCM checkpoint keys mismatch: "
                         f"missing={sorted(common - received)}, "
                         f"unexpected={sorted(received - common)}")
    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys or set(result.missing_keys) != expected - common:
        raise ValueError("only BCIF coordinator keys may be absent")
    return model


class Model(OriginalSALCM):
    coordinator_type = BCIFCoordinator

    def __init__(self, configs):
        super().__init__(configs)
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            depth = len(self.encoder.attn_layers)
            if depth < 1:
                raise ValueError("BCIF requires at least one encoder layer")
            for index, original in enumerate(self.encoder.attn_layers):
                layer = BCIFLinearEncoder(
                    d_model=configs.d_model, d_ff=configs.d_ff, CovMat=None,
                    dropout=configs.dropout, activation=configs.activation,
                    token_num=self.enc_in,
                )
                layer.load_state_dict(original.state_dict(), strict=True)
                self.encoder.attn_layers[index] = layer
            self.coordinators = nn.ModuleList([
                self.coordinator_type(configs.d_model, rank=16, cap=0.1)
                for _ in range(depth - 1)
            ])
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    def _bcif_transform(self, embedded, history_mass):
        batch, channels, length, width = embedded.shape
        if channels != self.enc_in or length != self.seq_len:
            raise ValueError("BCIF transform channel/sequence dimensions mismatch")
        state = embedded.transpose(-1, -2)
        if self.Q_chan_indep:
            state = torch.einsum("bndt,ntv->bndv", state, self.Q_mat.transpose(-1, -2))
        else:
            state = torch.einsum("bndt,tv->bndv", state, self.Q_mat.transpose(-1, -2)) + self.delta1
        state = self.ortho_trans[0](state.flatten(-2))

        transfer = None
        for index, layer in enumerate(self.encoder.attn_layers):
            state, update = layer(state, transfer=transfer)
            if index < len(self.coordinators):
                next_layer = self.encoder.attn_layers[index + 1]
                transfer = self.coordinators[index](
                    update, history_mass, layer._prior(), next_layer._prior(), state
                )
        if self.encoder.norm is not None:
            state = self.encoder.norm(state)
        state = self.ortho_trans[2](state).reshape(batch, channels, width, self.pred_len)
        if self.Q_chan_indep:
            state = torch.einsum("bndt,ntv->bndv", state, self.Q_out_mat)
        else:
            state = torch.einsum("bndt,tv->bndv", state, self.Q_out_mat) + self.delta2
        return state.transpose(-1, -2)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x.ndim != 3 or x.shape[1:] != (768, self.enc_in):
            raise ValueError("OLinear_SALCM_BCIF expects [B,768,C]")
        recent = x[:, -96:, :]
        z_recent = self.revin_layer(recent, mode="norm")
        z_past = anchor_normalize_past(x[:, :672, :], self.revin_layer)
        refined, _, _, history_mass = self.sa_lcm(z_recent, z_past, return_evidence=True)
        encoded = self._bcif_transform(self.tokenEmb(refined, self.embeddings), history_mass)
        prediction = self.fc(encoded.flatten(-2)).transpose(-1, -2)
        return self.revin_layer(self.dropout(prediction), mode="denorm")
