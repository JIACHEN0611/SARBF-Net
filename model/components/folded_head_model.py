"""BCIF with its output affine chain folded into one encoder-width projection.

For Q_chan_indep=0, the historical path mapped [B,C,D] through
ortho_trans[2] (D -> E*H), Q_out+delta2, then fc[0] (E*H -> F).
The new path maps [B,C,D] directly to F and adds a separate delta2 head.
The nonlinearity and everything outside that affine chain are unchanged.
"""

import torch
from torch import nn

from layers.SelectiveAnchoredLongContext import anchor_normalize_past
from model.components.compact_frontend_model import Model as NoDiagModel


def _check_q(model):
    horizon = model.pred_len
    q = model.Q_out_mat
    if model.Q_chan_indep != 0:
        raise ValueError("output fold requires Q_chan_indep=0")
    input_q = model.Q_mat
    if (input_q.ndim != 2
            or tuple(input_q.shape) != (model.seq_len, model.seq_len)
            or not bool(torch.isfinite(input_q).all())):
        raise ValueError("Q_mat must be finite [seq_len,seq_len]")
    if q.ndim != 2 or tuple(q.shape) != (horizon, horizon):
        raise ValueError(f"Q_out must be finite [{horizon},{horizon}], got {tuple(q.shape)}")
    if not bool(torch.isfinite(q).all()):
        raise ValueError("Q_out must contain only finite values")
    if tuple(model.delta2.shape) != (1, model.enc_in, 1, horizon):
        raise ValueError("delta2 shape does not match the shared output Q")


def _fold_head(model, ortho_weight, ortho_bias, head_weight, head_bias):
    """Return core weight/bias and delta weight, using double on Q's device."""
    _check_q(model)
    horizon, embedding, width, feedforward = (
        model.pred_len, model.embed_size, model.d_model, model.d_ff
    )
    shapes = ((ortho_weight, (horizon * embedding, width)),
              (ortho_bias, (horizon * embedding,)),
              (head_weight, (feedforward, horizon * embedding)),
              (head_bias, (feedforward,)))
    for tensor, expected in shapes:
        if tuple(tensor.shape) != expected:
            raise ValueError(f"old output head shape {tuple(tensor.shape)} != {expected}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("old output head contains nonfinite parameters")

    # The old _bcif_transform transposes [B,C,E,H] to [B,C,H,E] before fc[0].
    # A[f,v,e] indexes output horizon v, then embedding e in fc[0].
    device = model.Q_out_mat.device
    a = head_weight.to(device=device, dtype=torch.float64).reshape(
        feedforward, horizon, embedding
    )
    q = model.Q_out_mat.to(dtype=torch.float64)
    g = torch.einsum("fve,tv->fet", a, q)
    flat_g = g.reshape(feedforward, embedding * horizon)
    old_weight = ortho_weight.to(device=device, dtype=torch.float64)
    old_bias = ortho_bias.to(device=device, dtype=torch.float64)
    core_weight = flat_g @ old_weight
    core_bias = (head_bias.to(device=device, dtype=torch.float64)
                 + flat_g @ old_bias)
    delta_weight = a.sum(dim=2)
    return (core_weight.to(device=ortho_weight.device, dtype=ortho_weight.dtype),
            core_bias.to(device=head_bias.device, dtype=head_bias.dtype),
            delta_weight.to(device=head_weight.device, dtype=head_weight.dtype))


def convert_old_bcif_state_dict(model, old_state_dict):
    """Convert an exact old BCIF/no_diag checkpoint without mutating the input.

    Original OLinear does not register Q as a buffer. The caller must construct
    ``model`` with the verified original Q file; the replay verifies its SHA256.
    Both duplicate encoder and ortho_trans.1 alias keys must be present and
    identical. The converted result is intended for ``load_state_dict(strict=True)``.
    """
    if not isinstance(model, Model):
        raise TypeError("expected OLinear_SALCM_BCIF_OutputFold.Model")
    _check_q(model)
    current = model.state_dict()
    expected_old = ((set(current) - {"delta_head.weight"})
                    | {"ortho_trans.2.weight", "ortho_trans.2.bias"})
    if set(old_state_dict) != expected_old:
        raise ValueError("old BCIF state_dict keys mismatch: "
                         f"missing={sorted(expected_old - set(old_state_dict))}, "
                         f"unexpected={sorted(set(old_state_dict) - expected_old)}")
    for key, expected in current.items():
        if key in ("fc.0.weight", "fc.0.bias", "delta_head.weight"):
            continue
        if tuple(old_state_dict[key].shape) != tuple(expected.shape):
            raise ValueError(f"old BCIF state_dict shape mismatch: {key}")
    # PyTorch exposes this shared encoder under both names in state_dict().
    for key in tuple(expected_old):
        if key.startswith("encoder."):
            alias = "ortho_trans.1." + key[len("encoder."):]
            if alias not in old_state_dict or not torch.equal(old_state_dict[key],
                                                               old_state_dict[alias]):
                raise ValueError(f"old BCIF state_dict alias mismatch: {key}")

    with torch.no_grad():
        core_weight, core_bias, delta_weight = _fold_head(
            model, old_state_dict["ortho_trans.2.weight"],
            old_state_dict["ortho_trans.2.bias"],
            old_state_dict["fc.0.weight"], old_state_dict["fc.0.bias"],
        )
    converted = {key: value for key, value in old_state_dict.items()
                 if not key.startswith("ortho_trans.2.")}
    converted.update({"fc.0.weight": core_weight, "fc.0.bias": core_bias,
                      "delta_head.weight": delta_weight})
    if set(converted) != set(current):
        raise ValueError("converted state_dict keys mismatch")
    return converted


class Model(NoDiagModel):
    def __init__(self, configs):
        if int(configs.Q_chan_indep) != 0:
            raise ValueError("output fold requires Q_chan_indep=0")
        super().__init__(configs)
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            _check_q(self)
            original_ortho = self.ortho_trans[2]
            original_fc = self.fc[0]
            with torch.no_grad():
                core_weight, core_bias, delta_weight = _fold_head(
                    self, original_ortho.weight, original_ortho.bias,
                    original_fc.weight, original_fc.bias,
                )
            self.ortho_trans[2] = nn.Identity()
            self.fc[0] = nn.Linear(self.d_model, self.d_ff).to(
                device=original_fc.weight.device, dtype=original_fc.weight.dtype
            )
            self.delta_head = nn.Linear(self.pred_len, self.d_ff, bias=False).to(
                device=original_fc.weight.device, dtype=original_fc.weight.dtype
            )
            with torch.no_grad():
                self.fc[0].weight.copy_(core_weight)
                self.fc[0].bias.copy_(core_bias)
                self.delta_head.weight.copy_(delta_weight)
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    def _bcif_transform(self, embedded, history_mass):
        batch, channels, length, width = embedded.shape
        if channels != self.enc_in or length != self.seq_len:
            raise ValueError("BCIF transform channel/sequence dimensions mismatch")
        state = embedded.transpose(-1, -2)
        state = (torch.einsum("bndt,tv->bndv", state, self.Q_mat.transpose(-1, -2))
                 + self.delta1)
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
        return state

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x.ndim != 3 or x.shape[1:] != (768, self.enc_in):
            raise ValueError("OLinear_SALCM_BCIF_OutputFold expects [B,768,C]")
        recent = x[:, -96:, :]
        z_recent = self.revin_layer(recent, mode="norm")
        z_past = anchor_normalize_past(x[:, :672, :], self.revin_layer)
        refined, _, _, history_mass = self.sa_lcm(z_recent, z_past,
                                                   return_evidence=True)
        encoded = self._bcif_transform(self.tokenEmb(refined, self.embeddings),
                                       history_mass)
        # delta2 was broadcast over E after Q_out, then passed through fc[0].
        preactivation = (self.fc[0](encoded)
                         + self.delta_head(self.delta2.squeeze(2)))
        prediction = self.fc[2](self.fc[1](preactivation)).transpose(-1, -2)
        return self.revin_layer(self.dropout(prediction), mode="denorm")
