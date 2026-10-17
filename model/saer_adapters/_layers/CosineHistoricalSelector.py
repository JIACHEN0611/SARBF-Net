"""Pair-free historical selection used by the compact SA-LCM frontend."""

import torch
from torch import nn
from torch.nn import functional as F

from model.saer_adapters._layers.SelectiveAnchoredLongContext import SelectionOutput


class CosineHistoricalSelector(nn.Module):
    def __init__(self, source):
        super().__init__()
        if (
            source.hidden_dim != 64
            or source.cosine_scale != 4.0
            or source.num_blocks != 7
        ):
            raise ValueError("pair-free selector requires fixed SA-LCM geometry")
        self.hidden_dim = source.hidden_dim
        self.cosine_scale = source.cosine_scale
        self.num_blocks = source.num_blocks
        self.null_head = source.null_head
        self.age_bias = source.age_bias

    @classmethod
    def from_existing(cls, source):
        return cls(source)

    def forward(self, q_recent, h_blocks):
        if q_recent.ndim != 3 or h_blocks.ndim != 4:
            raise ValueError("selector expects q=[B,C,64], h=[B,C,7,64]")
        if h_blocks.shape[:-2] != q_recent.shape[:-1]:
            raise ValueError("selector batch/channel geometry mismatch")
        if h_blocks.shape[-2:] != (self.num_blocks, self.hidden_dim):
            raise ValueError("selector history geometry mismatch")

        q_history = q_recent.unsqueeze(-2).expand_as(h_blocks)
        cosine = (
            F.normalize(q_history, dim=-1, eps=1e-8)
            * F.normalize(h_blocks, dim=-1, eps=1e-8)
        ).sum(-1)
        history_scores = self.cosine_scale * cosine + self.age_bias
        null_score = self.null_head(q_recent).squeeze(-1)
        weights = torch.softmax(
            torch.cat([null_score.unsqueeze(-1), history_scores], dim=-1), dim=-1
        )
        history_weights = weights[..., 1:]
        null_weight = 1.0 - history_weights.sum(-1)
        history_mass = 1.0 - null_weight
        weighted = (history_weights.unsqueeze(-1) * h_blocks).sum(-2)
        context = torch.where(
            history_mass.unsqueeze(-1) > 1e-6,
            weighted / history_mass.clamp_min(1e-6).unsqueeze(-1),
            torch.zeros_like(weighted),
        )
        return SelectionOutput(
            context=context,
            history_mass=history_mass,
            null_weight=null_weight,
            history_weights=history_weights,
            cosine_scores=cosine,
        )
