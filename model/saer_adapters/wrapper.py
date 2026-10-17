"""One long-context input wrapper shared by unchanged forecasting backbones."""

import torch
from torch import nn

from model.saer_adapters.frontend import SALCMInputFrontend
from model.saer_adapters.registry import BACKBONE_MODELS


class LongContextBackbone(nn.Module):
    """Slice or refine the recent window, then call an official backbone."""

    def __init__(self, configs):
        super().__init__()
        self.backbone_name = str(configs.backbone)
        self.arm = str(configs.arm)
        self.long_len = int(configs.long_len)
        self.seq_len = int(configs.seq_len)
        self.enc_in = int(configs.enc_in)
        if self.backbone_name not in BACKBONE_MODELS:
            raise ValueError(f"unsupported backbone: {self.backbone_name}")
        if self.arm not in {"original", "sa_frontend"}:
            raise ValueError("arm must be original or sa_frontend")
        if self.seq_len != 96 or self.long_len != 768:
            raise ValueError(
                "wrapper requires seq_len in {96,336} and long_len=seq_len+672"
            )
        if self.seq_len == 336 and self.backbone_name not in {
            "DLinear",
            "PatchTST",
        }:
            raise ValueError(
                f"{self.backbone_name} has no official 336 arm in this experiment"
            )

        # Construct first so paired arms have identical backbone parameters.
        self.backbone = BACKBONE_MODELS[self.backbone_name].Model(configs)
        self.frontend = None
        if self.arm == "sa_frontend":
            cpu_state = torch.random.get_rng_state()
            cuda_state = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            try:
                self.frontend = SALCMInputFrontend(
                    enc_in=self.enc_in,
                    long_len=768,
                    seq_len=96,
                )
            finally:
                torch.random.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)

    def _recent_values(self, long_x):
        if (
            long_x.ndim != 3
            or long_x.shape[1] != self.long_len
            or long_x.shape[2] != self.enc_in
        ):
            raise ValueError(
                f"LongContextBackbone expects [B,{self.long_len},{self.enc_in}]"
            )
        if self.frontend is None:
            return long_x[:, -self.seq_len :, :]
        if self.seq_len == 96:
            return self.frontend(long_x)

        # The SA-LCM core remains the locked 672-history + 96-recent module.
        # With a 336-step official backbone input, preserve its first 240 recent
        # observations exactly and replace only the final 96 with the refinement.
        recent = long_x[:, -self.seq_len :, :]
        frontend_input = torch.cat(
            [long_x[:, :672, :], recent[:, -96:, :]], dim=1
        )
        refined_tail = self.frontend(frontend_input)
        return torch.cat([recent[:, :-96, :], refined_tail], dim=1)

    def _recent_marks(self, x_mark_enc, batch_size):
        if x_mark_enc is None:
            return None
        if (
            x_mark_enc.ndim != 3
            or x_mark_enc.shape[0] != batch_size
            or x_mark_enc.shape[1] != self.long_len
        ):
            raise ValueError(
                f"x_mark_enc must align with the {self.long_len}-step input"
            )
        return x_mark_enc[:, -self.seq_len :, :]

    def forward(
        self,
        long_x,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        cycle_index=None,
        mask=None,
    ):
        recent = self._recent_values(long_x)
        recent_marks = self._recent_marks(x_mark_enc, long_x.shape[0])

        if self.backbone_name in {"DLinear", "PatchTST"}:
            return self.backbone(recent)
        if self.backbone_name == "CycleNet":
            if cycle_index is None:
                raise ValueError("CycleNet requires cycle_index")
            return self.backbone(recent, cycle_index)
        return self.backbone(
            recent,
            recent_marks,
            x_dec,
            x_mark_dec,
            mask=mask,
        )


Model = LongContextBackbone
