"""Forecasting backbones used by the five-model SAER study."""
from model.backbones import DLinear, PatchTST, iTransformer, CycleNet, TSMixer

BACKBONE_MODELS = {
    'DLinear': DLinear,
    'PatchTST': PatchTST,
    'iTransformer': iTransformer,
    'CycleNet': CycleNet,
    'TSMixer': TSMixer,
}
