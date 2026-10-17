"""Locked 36-task official-configuration strong-baseline matrix."""

from dataclasses import dataclass


ARMS = ("original", "sa_frontend")
DATASETS = {
    "ETTm1": {
        "data": "ETTm1",
        "data_path": "ETTm1.csv",
        "enc_in": 7,
        "cycle": 96,
    },
    "Electricity": {
        "data": "custom",
        "data_path": "electricity.csv",
        "enc_in": 321,
        "cycle": 168,
    },
    "Traffic": {
        "data": "custom",
        "data_path": "traffic.csv",
        "enc_in": 862,
        "cycle": 168,
    },
}
INPUT_LENGTHS = {
    "DLinear": (96, 336),
    "PatchTST": (96, 336),
    "iTransformer": (96,),
    "CycleNet": (96,),
}


DL_CONFIG = {
    "ETTm1": {"batch_size": 8, "learning_rate": 0.0001},
    "Electricity": {"batch_size": 16, "learning_rate": 0.001},
    "Traffic": {"batch_size": 16, "learning_rate": 0.05},
}
PATCH_CONFIG = {
    "ETTm1": {"batch_size": 128, "patience": 20, "pct_start": 0.4},
    "Electricity": {"batch_size": 32, "patience": 10, "pct_start": 0.2},
    "Traffic": {"batch_size": 24, "patience": 10, "pct_start": 0.2},
}
ITRANSFORMER_CONFIG = {
    "ETTm1": {
        "batch_size": 32,
        "learning_rate": 0.0001,
        "d_model": 128,
        "d_ff": 128,
        "e_layers": 2,
    },
    "Electricity": {
        "batch_size": 16,
        "learning_rate": 0.0005,
        "d_model": 512,
        "d_ff": 512,
        "e_layers": 3,
    },
    "Traffic": {
        "batch_size": 16,
        "learning_rate": 0.001,
        "d_model": 512,
        "d_ff": 512,
        "e_layers": 4,
    },
}
CYCLENET_CONFIG = {
    "ETTm1": {"batch_size": 256},
    "Electricity": {"batch_size": 64},
    "Traffic": {"batch_size": 64},
}


@dataclass(frozen=True)
class StrongPilotTask:
    backbone: str
    dataset: str
    seq_len: int
    arm: str
    horizon: int = 96
    seed: int = 2021

    @property
    def long_len(self):
        return self.seq_len + 672

    @property
    def task_id(self):
        return (
            f"{self.backbone}-{self.dataset}-sl{self.seq_len}-h{self.horizon}-"
            f"{self.arm}-s{self.seed}"
        )

    @property
    def dataset_profile(self):
        return dict(DATASETS[self.dataset])

    @property
    def backbone_profile(self):
        if self.backbone == "DLinear":
            return {
                **DL_CONFIG[self.dataset],
                "train_epochs": 10,
                "patience": 3,
                "lradj": "type1",
            }
        if self.backbone == "PatchTST":
            return {
                **PATCH_CONFIG[self.dataset],
                "learning_rate": 0.0001,
                "train_epochs": 100,
                "lradj": "TST",
                "d_model": 128,
                "d_ff": 256,
                "e_layers": 3,
                "n_heads": 16,
                "dropout": 0.2,
                "fc_dropout": 0.2,
                "head_dropout": 0.0,
                "patch_len": 16,
                "stride": 8,
                "revin": 1,
            }
        if self.backbone == "iTransformer":
            return {
                **ITRANSFORMER_CONFIG[self.dataset],
                "train_epochs": 10,
                "patience": 3,
                "lradj": "type1",
                "dropout": 0.1,
                "n_heads": 8,
            }
        return {
            **CYCLENET_CONFIG[self.dataset],
            "learning_rate": 0.01,
            "train_epochs": 30,
            "patience": 5,
            "lradj": "type3",
            "model_type": "linear",
        }


def build_strong_pilot_tasks():
    return [
        StrongPilotTask(backbone, dataset, seq_len, arm)
        for backbone, seq_lengths in INPUT_LENGTHS.items()
        for dataset in DATASETS
        for seq_len in seq_lengths
        for arm in ARMS
    ]
