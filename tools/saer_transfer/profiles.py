"""Locked 720-task matrix for the five-backbone SAER generality study."""

from dataclasses import dataclass
from itertools import product

from tools.saer_transfer.strong_profiles import StrongPilotTask


BACKBONES = ("DLinear", "PatchTST", "iTransformer", "CycleNet", "TSMixer")
# PatchTST is deliberately last because it is the slowest backbone in this setup.
RUN_ORDER = ("DLinear", "CycleNet", "TSMixer", "iTransformer", "PatchTST")
ARMS = ("original", "sa_frontend")
SEEDS = (2021, 2022, 2023)
HORIZONS = (96, 192, 336, 720)

DATASETS = {
    "ETTm1": {"data": "ETTm1", "data_path": "ETTm1.csv", "enc_in": 7, "cycle": 96},
    # ECL is kept as Electricity internally to preserve completed pilot paths.
    "Electricity": {
        "data": "custom",
        "data_path": "electricity.csv",
        "enc_in": 321,
        "cycle": 168,
    },
    "PEMS03": {"data": "PEMS", "data_path": "PEMS03.npz", "enc_in": 358, "cycle": 288},
    "PEMS07": {"data": "PEMS", "data_path": "PEMS07.npz", "enc_in": 883, "cycle": 288},
    "Solar": {"data": "Solar", "data_path": "solar_AL.txt", "enc_in": 137, "cycle": 144},
    "Weather": {"data": "custom", "data_path": "weather.csv", "enc_in": 21, "cycle": 144},
}

# The new datasets use conservative batches for the long-context frontend. Both
# arms always receive the same profile, preserving a controlled comparison.
DL_CONFIGS = {
    "PEMS03": {"batch_size": 32, "learning_rate": 0.01},
    "PEMS07": {"batch_size": 16, "learning_rate": 0.01},
    "Solar": {"batch_size": 128, "learning_rate": 0.01},
    "Weather": {"batch_size": 128, "learning_rate": 0.001},
}
PATCH_CONFIGS = {
    "PEMS03": {"batch_size": 8},
    "PEMS07": {"batch_size": 4},
    "Solar": {"batch_size": 16},
    "Weather": {"batch_size": 32},
}
ITRANSFORMER_CONFIGS = {
    "PEMS03": {"batch_size": 8, "d_model": 128, "d_ff": 128, "e_layers": 2},
    "PEMS07": {"batch_size": 2, "d_model": 128, "d_ff": 128, "e_layers": 2},
    "Solar": {"batch_size": 16, "d_model": 256, "d_ff": 256, "e_layers": 2},
    "Weather": {"batch_size": 32, "d_model": 512, "d_ff": 512, "e_layers": 3},
}
CYCLENET_CONFIGS = {
    "PEMS03": {"batch_size": 32, "learning_rate": 0.01},
    "PEMS07": {"batch_size": 16, "learning_rate": 0.01},
    "Solar": {"batch_size": 128, "learning_rate": 0.01},
    "Weather": {"batch_size": 128, "learning_rate": 0.001},
}
TSMIXER_CONFIGS = {
    "ETTm1": {"batch_size": 32, "learning_rate": 0.0001, "d_model": 512, "d_ff": 2048},
    "Electricity": {"batch_size": 32, "learning_rate": 0.0001, "d_model": 256, "d_ff": 512},
    "PEMS03": {"batch_size": 16, "learning_rate": 0.001, "d_model": 256, "d_ff": 32},
    "PEMS07": {"batch_size": 8, "learning_rate": 0.001, "d_model": 256, "d_ff": 32},
    "Solar": {"batch_size": 32, "learning_rate": 0.0001, "d_model": 256, "d_ff": 512},
    "Weather": {"batch_size": 32, "learning_rate": 0.0001, "d_model": 32, "d_ff": 32},
}


@dataclass(frozen=True)
class FiveModelTask:
    backbone: str
    dataset: str
    seed: int
    arm: str
    seq_len: int = 96
    horizon: int = 96

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
        if self.dataset in {"ETTm1", "Electricity"} and self.backbone != "TSMixer":
            return StrongPilotTask(
                backbone=self.backbone,
                dataset=self.dataset,
                seq_len=self.seq_len,
                arm=self.arm,
                horizon=self.horizon,
                seed=self.seed,
            ).backbone_profile
        if self.backbone == "DLinear":
            return {**DL_CONFIGS[self.dataset], "train_epochs": 30, "patience": 5, "lradj": "type3"}
        if self.backbone == "PatchTST":
            return {
                **PATCH_CONFIGS[self.dataset],
                "learning_rate": 0.0001,
                "train_epochs": 30,
                "patience": 5,
                "lradj": "TST",
                "pct_start": 0.3,
                "d_model": 128,
                "d_ff": 256,
                "e_layers": 2,
                "n_heads": 8,
                "dropout": 0.2,
                "fc_dropout": 0.2,
                "head_dropout": 0.0,
                "patch_len": 16,
                "stride": 8,
                "revin": 1,
            }
        if self.backbone == "iTransformer":
            return {
                **ITRANSFORMER_CONFIGS[self.dataset],
                "learning_rate": 0.0001,
                "train_epochs": 10,
                "patience": 3,
                "lradj": "type1",
                "dropout": 0.1,
                "n_heads": 8,
            }
        if self.backbone == "CycleNet":
            return {
                **CYCLENET_CONFIGS[self.dataset],
                "train_epochs": 30,
                "patience": 5,
                "lradj": "type3",
                "model_type": "linear",
            }
        return {
            **TSMIXER_CONFIGS[self.dataset],
            "train_epochs": 10,
            "patience": 3,
            "lradj": "type1",
            "e_layers": 2,
            "dropout": 0.1,
        }


def build_five_model_tasks():
    return [
        FiveModelTask(backbone, dataset, seed, arm, horizon=horizon)
        for backbone, dataset, horizon, seed, arm in product(
            RUN_ORDER, DATASETS, HORIZONS, SEEDS, ARMS
        )
    ]


def build_tasks():
    """Return the preserved five-backbone experiment matrix."""
    return build_five_model_tasks()
