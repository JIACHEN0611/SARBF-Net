"""Balanced feedback of the actual intra-series update to a later CSL."""

import torch
from torch import nn
from torch.nn import functional as F

from layers.Transformer_EncDec import LinearEncoder


class BCIFLinearEncoder(LinearEncoder):
    """Original CSL and ISL, with opposite transfer at their interface."""

    def _prior(self):
        if self.CovMat is not None:
            raw = F.softmax(self.CovMat, dim=-1) + F.softplus(self.weight_mat)
        else:
            raw = F.softplus(self.weight_mat)
        return F.normalize(raw, p=1, dim=-1)

    def forward(self, x, *, transfer=None):
        if x.ndim != 3 or x.shape[1:] != (self.token_num, self.d_model):
            raise ValueError("BCIF encoder expects [batch, channels, d_model]")
        if transfer is not None and transfer.shape != x.shape:
            raise ValueError("transfer must have the same shape as the input")

        offered = x if transfer is None else x + transfer
        values = self.v_proj(offered)
        prior = self._prior()
        mixed = self.dropout(prior) @ values
        z = self.norm1(offered + self.dropout(self.out_proj(mixed)))
        local = z if transfer is None else z - transfer
        hidden = self.dropout(self.activation(self.conv1(local.transpose(-1, 1))))
        update = self.dropout(self.conv2(hidden).transpose(-1, 1))
        return self.norm2(local + update), update


class BCIFCoordinator(nn.Module):
    """Return a bounded update formed from reliable channel disagreements."""

    def __init__(self, d_model, rank=16, cap=0.1):
        super().__init__()
        if d_model < 1 or rank < 1 or not (0 < cap <= 1):
            raise ValueError("d_model, rank, and cap must be positive")
        self.d_model = d_model
        self.cap = cap
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, update, history_mass, prior, next_prior, next_hidden):
        if update.ndim != 3 or update.shape != next_hidden.shape:
            raise ValueError("update and next_hidden must have matching [B,C,D] shape")
        batch, channels, width = update.shape
        if (width != self.d_model or history_mass.shape != (batch, channels)
                or prior.shape != (1, channels, channels)
                or next_prior.shape != (1, channels, channels)):
            raise ValueError("coordinator has incompatible width, mass, or relation shape")
        if not (torch.isfinite(history_mass) & (history_mass >= 0)
                & (history_mass <= 1)).all():
            raise ValueError("history_mass must be finite within [0,1]")

        # Center first: identical channel updates give exactly zero, including
        # at C=1, regardless of floating-point row-sum rounding in the prior.
        centered = update - update[:, :1, :]
        novelty = centered - torch.matmul(prior, centered)
        reliable = history_mass.unsqueeze(-1) * novelty
        neighbors = torch.matmul(next_prior, reliable)
        mass = torch.matmul(next_prior, history_mass.unsqueeze(-1))
        discrepancy = neighbors - mass * novelty
        scale = next_hidden.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return self.cap * scale * torch.tanh(self.up(self.down(discrepancy)) / scale)


class BCIFNoMassCoordinator(BCIFCoordinator):
    """Balanced BCIF with no external SA-LCM confidence in its transfer."""

    def forward(self, update, history_mass, prior, next_prior, next_hidden):
        if update.ndim != 3 or update.shape != next_hidden.shape:
            raise ValueError("update and next_hidden must have matching [B,C,D] shape")
        batch, channels, width = update.shape
        if (width != self.d_model or history_mass.shape != (batch, channels)
                or prior.shape != (1, channels, channels)
                or next_prior.shape != (1, channels, channels)):
            raise ValueError("coordinator has incompatible width, mass, or relation shape")
        if not (torch.isfinite(history_mass) & (history_mass >= 0)
                & (history_mass <= 1)).all():
            raise ValueError("history_mass must be finite within [0,1]")

        centered = update - update[:, :1, :]
        novelty = centered - torch.matmul(prior, centered)
        discrepancy = torch.matmul(next_prior, novelty) - novelty
        scale = next_hidden.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return self.cap * scale * torch.tanh(self.up(self.down(discrepancy)) / scale)


class BCIFUnbalancedCoordinator(BCIFCoordinator):
    """BCIF transport without the neighbor-mass self-cancellation term."""

    def forward(self, update, history_mass, prior, next_prior, next_hidden):
        if update.ndim != 3 or update.shape != next_hidden.shape:
            raise ValueError("update and next_hidden must have matching [B,C,D] shape")
        batch, channels, width = update.shape
        if (width != self.d_model or history_mass.shape != (batch, channels)
                or prior.shape != (1, channels, channels)
                or next_prior.shape != (1, channels, channels)):
            raise ValueError("coordinator has incompatible width, mass, or relation shape")
        if not (torch.isfinite(history_mass) & (history_mass >= 0)
                & (history_mass <= 1)).all():
            raise ValueError("history_mass must be finite within [0,1]")

        centered = update - update[:, :1, :]
        novelty = centered - torch.matmul(prior, centered)
        discrepancy = torch.matmul(next_prior, history_mass.unsqueeze(-1) * novelty)
        scale = next_hidden.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return self.cap * scale * torch.tanh(self.up(self.down(discrepancy)) / scale)
