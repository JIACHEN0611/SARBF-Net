"""SA-LCM frontend without unused per-batch scalar diagnostics.

The model parameters, evidence, and opt-in full-validation diagnostics are
inherited from the release frontend. Only the unconditional CPU scalar
conversion in ``forward`` is omitted.
"""

import torch

from layers.SelectiveAnchoredLongContext import SelectiveAnchoredLongContext


class SafePruneFrontend(SelectiveAnchoredLongContext):
    """Drop single-batch reporting while preserving the original forward path."""

    def _refine(self, z_recent, z_past):
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
        refined = refined.transpose(1, 2)
        return refined, q_recent, selection.context, selection.history_mass

    def refine_with_mass(self, z_recent, z_past):
        """Return only the tensors consumed by the forecasting backbone."""
        refined, _, _, history_mass = self._refine(z_recent, z_past)
        return refined, history_mass

    def forward(self, z_recent, z_past, return_evidence=False):
        refined, q_recent, context, history_mass = self._refine(z_recent, z_past)
        if return_evidence:
            return refined, q_recent, context, history_mass
        return refined
