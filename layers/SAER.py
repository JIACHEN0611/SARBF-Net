"""Paper-name API for Selective Anchored Retrieval (SAER).

The full-model constructor installs CosineHistoricalSelector after building the
inherited graph to preserve the original random initialization order.
"""

from layers.CosineHistoricalSelector import CosineHistoricalSelector
from layers.SafePruneFrontend import SafePruneFrontend as _Frontend
from layers.SelectiveAnchoredLongContext import anchor_normalize_past


class SAER(_Frontend):
    def __init__(self, hidden_dim=64):
        super().__init__(hidden_dim=hidden_dim)
        self.selector = CosineHistoricalSelector.from_existing(self.selector)


__all__ = ["SAER", "anchor_normalize_past"]
