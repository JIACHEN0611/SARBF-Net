"""Selective anchored long-context components for SA-LCM V1."""

import torch
from torch import nn
from torch.nn import functional as F
from typing import NamedTuple


def anchor_normalize_past(past, revin_layer):
    """Normalize Past with the already-cached Recent RevIN statistics."""
    if not hasattr(revin_layer, "stdev"):
        raise RuntimeError("Recent RevIN statistics are not initialized")
    center_name = "last" if getattr(revin_layer, "subtract_last", False) else "mean"
    if not hasattr(revin_layer, center_name):
        raise RuntimeError("Recent RevIN statistics are not initialized")
    center = getattr(revin_layer, center_name)
    normalized = (past - center) / revin_layer.stdev
    if revin_layer.affine:
        normalized = normalized * revin_layer.affine_weight
        normalized = normalized + revin_layer.affine_bias
    return normalized


def split_history_blocks(sequence):
    """Convert [B,T,C] into ordered [B,C,P,96] blocks."""
    if sequence.ndim != 3 or sequence.shape[1] not in (96, 672):
        raise ValueError("SA-LCM blocks require T=96 or T=672")
    batch, length, channels = sequence.shape
    return sequence.transpose(1, 2).contiguous().view(batch, channels, length // 96, 96)


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
    """One channel-independent encoder shared by Recent and all Past blocks."""

    def __init__(self, hidden_dim=64, kernel_size=25):
        super().__init__()
        if hidden_dim != 64 or kernel_size != 25:
            raise ValueError("SA-LCM V1 fixes hidden_dim=64 and kernel_size=25")
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.trend_encoder = ComponentEncoder(hidden_dim)
        self.residual_encoder = ComponentEncoder(hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
        )

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
        trend_blocks = split_history_blocks(trend)
        residual_blocks = split_history_blocks(residual)
        encoded_trend = self.trend_encoder(trend_blocks)
        encoded_residual = self.residual_encoder(residual_blocks)
        return self.fusion(torch.cat([encoded_trend, encoded_residual], dim=-1))

    def encode_sequence_as_blocks(self, sequence):
        trend, residual = self.decompose(sequence)
        return self.encode_components_as_blocks(trend, residual)


class SelectionOutput(NamedTuple):
    context: torch.Tensor
    history_mass: torch.Tensor
    null_weight: torch.Tensor
    history_weights: torch.Tensor
    cosine_scores: torch.Tensor


class HistoricalSelector(nn.Module):
    def __init__(self, hidden_dim=64, cosine_scale=4.0, num_blocks=7):
        super().__init__()
        if hidden_dim != 64 or cosine_scale != 4.0 or num_blocks != 7:
            raise ValueError("SA-LCM V1 fixes hidden=64, cosine_scale=4, blocks=7")
        self.hidden_dim = hidden_dim
        self.cosine_scale = cosine_scale
        self.num_blocks = num_blocks
        self.pair_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.pair_scorer[-1].weight)
        nn.init.zeros_(self.pair_scorer[-1].bias)
        self.null_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.null_head.weight)
        nn.init.constant_(self.null_head.bias, 1.0)
        self.age_bias = nn.Parameter(torch.zeros(num_blocks))

    def forward(self, q_recent, h_blocks):
        if q_recent.ndim != 3 or h_blocks.ndim != 4:
            raise ValueError("selector expects q=[B,C,64], h=[B,C,7,64]")
        if h_blocks.shape[:-2] != q_recent.shape[:-1]:
            raise ValueError("selector batch/channel geometry mismatch")
        if h_blocks.shape[-2:] != (self.num_blocks, self.hidden_dim):
            raise ValueError("selector history geometry mismatch")
        q7 = q_recent.unsqueeze(-2).expand_as(h_blocks)
        cosine = (
            F.normalize(q7, dim=-1, eps=1e-8)
            * F.normalize(h_blocks, dim=-1, eps=1e-8)
        ).sum(-1)
        pair = torch.cat([q7, h_blocks, (q7 - h_blocks).abs(), q7 * h_blocks], dim=-1)
        history_scores = (
            self.cosine_scale * cosine
            + self.pair_scorer(pair).squeeze(-1)
            + self.age_bias
        )
        null_score = self.null_head(q_recent).squeeze(-1)
        weights = torch.softmax(torch.cat([null_score.unsqueeze(-1), history_scores], -1), -1)
        history_weights = weights[..., 1:]
        # Reconstruct the mathematically identical NULL complement so the
        # probability simplex and history_mass identity remain exact in fp32.
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


class SelectiveAnchoredLongContext(nn.Module):
    """Seven-block NULL-aware correction operating only in normalized space."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        if hidden_dim != 64:
            raise ValueError("SA-LCM V1 fixes hidden_dim=64")
        self.block_encoder = SharedBlockEncoder(hidden_dim=hidden_dim, kernel_size=25)
        self.selector = HistoricalSelector(
            hidden_dim=hidden_dim, cosine_scale=4.0, num_blocks=7
        )
        self.correction = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.GELU(),
            nn.Linear(128, 96),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self._last_diagnostics = None
        self._diagnostics_state = None

    def _validate(self, z_recent, z_past):
        if z_recent.ndim != 3 or z_recent.shape[1] != 96:
            raise ValueError("z_recent must have shape [B,96,C]")
        if z_past.ndim != 3 or z_past.shape[1] != 672:
            raise ValueError("z_past must have shape [B,672,C]")
        if z_recent.shape[0] != z_past.shape[0] or z_recent.shape[2] != z_past.shape[2]:
            raise ValueError("z_recent/z_past batch and channel dimensions must match")

    def forward(self, z_recent, z_past, return_evidence=False):
        self._validate(z_recent, z_past)
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
        raw_delta = self.correction(features)
        delta = torch.tanh(raw_delta)
        effective = selection.history_mass.unsqueeze(-1) * delta
        refined = z_recent.transpose(1, 2) + effective
        if self._diagnostics_state is not None:
            self.accumulate_diagnostics(selection, effective, z_recent)
        with torch.no_grad():
            recent_norm = z_recent.detach().double().square().sum().sqrt()
            self._last_diagnostics = {
                "null_weight_mean": float(selection.null_weight.detach().double().mean()),
                "history_mass_mean": float(selection.history_mass.detach().double().mean()),
                "bounded_correction_rms": float(effective.detach().double().square().mean().sqrt()),
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
        if self._last_diagnostics is None:
            return None
        return dict(self._last_diagnostics)

    def reset_diagnostics(self):
        """Begin bounded-memory full-validation aggregation.

        Only one detached float64 scalar per sample/channel (plus seven block
        weights and a few norms) is retained on CPU; no model tensor or graph
        is retained.
        """
        self._diagnostics_state = {
            "samples": 0,
            "batches": 0,
            "null": [],
            "mass": [],
            "history_weights": [],
            "entropy": [],
            "cosine_max": [],
            "cosine_mean": [],
            "correction_sumsq": [],
            "correction_max": [],
            "recent_sumsq": [],
        }

    def cancel_diagnostics(self):
        self._diagnostics_state = None

    def accumulate_diagnostics(self, selection, effective, z_recent):
        state = self._diagnostics_state
        if state is None:
            raise RuntimeError("full-validation diagnostics are not active")
        with torch.no_grad():
            all_weights = torch.cat(
                [selection.null_weight.unsqueeze(-1), selection.history_weights], -1
            )
            values = (
                selection.null_weight,
                selection.history_mass,
                selection.history_weights,
                selection.cosine_scores,
                effective,
                z_recent,
            )
            if not all(torch.isfinite(value).all().item() for value in values):
                self.cancel_diagnostics()
                raise ValueError("non-finite SA-LCM diagnostic input")
            weights64 = all_weights.detach().cpu().double()
            state["samples"] += int(z_recent.shape[0])
            state["batches"] += 1
            state["null"].append(selection.null_weight.detach().cpu().double().reshape(-1))
            state["mass"].append(selection.history_mass.detach().cpu().double().reshape(-1))
            state["history_weights"].append(
                selection.history_weights.detach().cpu().double().reshape(-1, 7)
            )
            state["entropy"].append(
                (-(weights64.clamp_min(1e-300).log() * weights64).sum(-1)).reshape(-1)
            )
            cosine = selection.cosine_scores.detach().cpu().double().reshape(-1, 7)
            state["cosine_max"].append(cosine.max(-1).values)
            state["cosine_mean"].append(cosine.mean(-1))
            correction = effective.detach().cpu().double().reshape(-1, 96)
            recent = z_recent.detach().cpu().double().transpose(1, 2).reshape(-1, 96)
            state["correction_sumsq"].append(correction.square().sum(-1))
            state["correction_max"].append(correction.abs().max(-1).values)
            state["recent_sumsq"].append(recent.square().sum(-1))

    @staticmethod
    def _stable_mean(values):
        return float(torch.sort(values.reshape(-1)).values.mean())

    def finalize_diagnostics(self, expected_samples):
        state = self._diagnostics_state
        self._diagnostics_state = None
        if state is None:
            raise RuntimeError("full-validation diagnostics are not active")
        if state["samples"] != int(expected_samples):
            raise RuntimeError(
                f"validation sample count mismatch: {state['samples']} != {expected_samples}"
            )
        if not state["null"]:
            raise RuntimeError("full-validation diagnostics have no batches")

        cat = lambda name: torch.cat(state[name], dim=0)
        null = cat("null")
        mass = cat("mass")
        history_weights = cat("history_weights")
        entropy = cat("entropy")
        cosine_max = cat("cosine_max")
        cosine_mean = cat("cosine_mean")
        correction_sumsq = cat("correction_sumsq")
        correction_max = cat("correction_max")
        recent_sumsq = cat("recent_sumsq")

        block_means = [
            self._stable_mean(history_weights[:, index]) for index in range(7)
        ]
        correction_total = torch.sort(correction_sumsq).values.sum()
        recent_total = torch.sort(recent_sumsq).values.sum()
        return {
            "diagnostics_scope": "full_validation",
            "samples": state["samples"],
            "batches": state["batches"],
            "null_weight_mean": self._stable_mean(null),
            "null_weight_p50": float(torch.quantile(null, 0.50)),
            "null_weight_p90": float(torch.quantile(null, 0.90)),
            "history_mass_mean": self._stable_mean(mass),
            "history_mass_p10": float(torch.quantile(mass, 0.10)),
            "history_mass_p50": float(torch.quantile(mass, 0.50)),
            "history_mass_p90": float(torch.quantile(mass, 0.90)),
            "history_block_weight_mean": block_means,
            "attention_entropy_mean": self._stable_mean(entropy),
            "cosine_max_mean": self._stable_mean(cosine_max),
            "cosine_mean": self._stable_mean(cosine_mean),
            "bounded_correction_rms": float(
                torch.sqrt(correction_total / (correction_sumsq.numel() * 96))
            ),
            "bounded_correction_max": float(correction_max.max()),
            "effective_refinement_ratio": float(
                torch.sqrt(correction_total) / (torch.sqrt(recent_total) + 1e-12)
            ),
            "fraction_history_mass_lt_0_1": float((mass < 0.1).double().mean()),
            "fraction_history_mass_gt_0_5": float((mass > 0.5).double().mean()),
        }
