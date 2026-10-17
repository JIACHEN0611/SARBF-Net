"""iTransformer with the SAER input adapter."""
from copy import copy
from model.saer_adapters.wrapper import LongContextBackbone


class Model(LongContextBackbone):
    def __init__(self, configs):
        config = copy(configs)
        config.backbone = 'iTransformer'
        config.arm = getattr(configs, "arm", "sa_frontend")
        super().__init__(config)
