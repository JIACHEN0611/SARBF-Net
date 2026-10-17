"""Historical model names map to their canonical runtime modules."""
from model.components import orthogonal_backbone
from model.components import history_model
from model.components import feedback_model
from model.components import compact_frontend_model
from model.components import folded_head_model

LEGACY_MODELS = {
    'OLinear': orthogonal_backbone,
    'OLinear_SALCM': history_model,
    'OLinear_SALCM_BCIF': feedback_model,
    'OLinear_SALCM_BCIF_NoDiag': compact_frontend_model,
    'OLinear_SALCM_BCIF_OutputFold': folded_head_model,
}
