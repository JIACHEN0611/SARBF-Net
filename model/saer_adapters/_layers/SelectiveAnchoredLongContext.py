"""Selective anchored long-context refinement extracted from SA-LCM."""

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


def anchor_normalize_past(past, revin_layer):
    """Normalize history with statistics already cached from the recent window."""
    if not hasattr(revin_layer, "stdev"):
        raise RuntimeError("recent RevIN statistics are not initialized")
    center_name = "last" if revin_layer.subtract_last else "mean"
    if not hasattr(revin_layer, center_name):
        raise RuntimeError("recent RevIN statistics are not initialized")
    center = getattr(revin_layer, center_name)
    normalized = (past - center) / revin_layer.stdev
    if revin_layer.affine:
        normalized = normalized * revin_layer.affine_weight
        normalized = normalized + revin_layer.affine_bias
    return normalized


def split_history_blocks(sequence):
    """Convert `[B,T,C]` into ordered `[B,C,P,96]` blocks."""
    if sequence.ndim != 3 or sequence.shape[1] not in (96, 672):
        raise ValueError("SA-LCM blocks require T=96 or T=672")
    batch, length, channels = sequence.shape
    return sequence.transpose(1, 2).contiguous().view(
        batch, channels, length // 96, 96
    )


class ComponentEncoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(96, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
        )

    def forward(self, blocks):
        return self.net(blocks)


class SharedBlockEncoder(nn.Module):
    """Channel-independent encoder shared by recent and historical blocks."""

    def __init__(self, hidden_dim=64, kernel_size=25):
        super().__init__()
        if hidden_dim != 64 or kernel_size != 25:
            raise ValueError("SA-LCM fixes hidden_dim=64 and kernel_size=25")
        self.trend_encoder = ComponentEncoder(hidden_dim)
        self.residual_encoder = ComponentEncoder(hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
        )
        self.kernel_size = kernel_size

    def decompose(self, sequence):
        if sequence.ndim != 3:
            raise ValueError("expected [B,T,C]")
        values = sequence.transpose(1, 2)
        pad = self.kernel_size // 2
        trend = F.avg_pool1d(
            F.pad(values, (pad, pad), mode="replicate"),
            kernel_size=self.kernel_size,
            stride=1,
        ).transpose(1, 2)
        return trend, sequence - trend

    def encode_components_as_blocks(self, trend, residual):
        encoded_trend = self.trend_encoder(split_history_blocks(trend))
        encoded_residual = self.residual_encoder(split_history_blocks(residual))
        return self.fusion(torch.cat([encoded_trend, encoded_residual], dim=-1))


class SelectionOutput(NamedTuple):
    context: torch.Tensor
    history_mass: torch.Tensor
    null_weight: torch.Tensor
    history_weights: torch.Tensor
    cosine_scores: torch.Tensor


class HistoricalSelector(nn.Module):
    """Construction-compatible selector replaced by the pair-free selector."""

    def __init__(self, hidden_dim=64, cosine_scale=4.0, num_blocks=7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cosine_scale = cosine_scale
        self.num_blocks = num_blocks
        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64), nn.GELU(), nn.Linear(64, 1)
        )
        nn.init.zeros_(self.pair_scorer[-1].weight)
        nn.init.zeros_(self.pair_scorer[-1].bias)
        self.null_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.null_head.weight)
        nn.init.constant_(self.null_head.bias, 1.0)
        self.age_bias = nn.Parameter(torch.zeros(num_blocks))


class SelectiveAnchoredLongContext(nn.Module):
    """Seven-block, NULL-aware correction operating in normalized space."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        if hidden_dim != 64:
            raise ValueError("SA-LCM fixes hidden_dim=64")
        self.block_encoder = SharedBlockEncoder(hidden_dim, kernel_size=25)
        self.selector = HistoricalSelector(hidden_dim, 4.0, 7)
        self.correction = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.GELU(),
            nn.Linear(128, 96),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self._last_diagnostics = None

    def forward(self, z_recent, z_past, return_evidence=False):
        if z_recent.ndim != 3 or z_recent.shape[1] != 96:
            raise ValueError("z_recent must have shape [B,96,C]")
        if z_past.ndim != 3 or z_past.shape[1] != 672:
            raise ValueError("z_past must have shape [B,672,C]")
        if z_recent.shape[0] != z_past.shape[0] or z_recent.shape[2] != z_past.shape[2]:
            raise ValueError("recent and history batch/channel dimensions must match")

        full = torch.cat([z_past, z_recent], dim=1)
        trend, residual = self.block_encoder.decompose(full)
        h_blocks = self.block_encoder.encode_components_as_blocks(
            trend[:, :672], residual[:, :672]
        )
        q_recent = self.block_encoder.encode_components_as_blocks(
            trend[:, -96:], residual[:, -96:]
        ).squeeze(-2)
        selection = self.selector(q_recent, h_blocks)
        features = torch.cat(
            [
                q_recent,
                selection.context,
                (q_recent - selection.context).abs(),
                q_recent * selection.context,
            ],
            dim=-1,
        )
        delta = torch.tanh(self.correction(features))
        effective = selection.history_mass.unsqueeze(-1) * delta
        refined = z_recent.transpose(1, 2) + effective

        with torch.no_grad():
            recent_norm = z_recent.detach().double().square().sum().sqrt()
            self._last_diagnostics = {
                "null_weight_mean": float(selection.null_weight.detach().double().mean()),
                "history_mass_mean": float(selection.history_mass.detach().double().mean()),
                "bounded_correction_rms": float(
                    effective.detach().double().square().mean().sqrt()
                ),
                "bounded_correction_max": float(effective.detach().abs().max()),
                "effective_refinement_ratio": float(
                    effective.detach().double().square().sum().sqrt()
                    / (recent_norm + 1e-12)
                ),
            }

        refined = refined.transpose(1, 2)
        if return_evidence:
            return refined, q_recent, selection.context, selection.history_mass
        return refined

    def last_diagnostics(self):
        return None if self._last_diagnostics is None else dict(self._last_diagnostics)
