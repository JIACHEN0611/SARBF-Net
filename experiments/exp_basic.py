import os
import torch
from ablations.models import (OLinear_SALCM_BCIF_OutputFold_NoMass,
                              OLinear_SALCM_BCIF_OutputFold_Unbalanced)
from model import SARBF_Net, SA_LCM
from model.legacy.registry import LEGACY_MODELS


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'SARBF_Net': SARBF_Net,
            'SA_LCM': SA_LCM,
            'OLinear_SALCM_BCIF_NoDiag': LEGACY_MODELS['OLinear_SALCM_BCIF_NoDiag'],
            'OLinear_SALCM_BCIF_OutputFold': LEGACY_MODELS['OLinear_SALCM_BCIF_OutputFold'],
            'OLinear_SALCM_BCIF_OutputFold_NoMass':
                OLinear_SALCM_BCIF_OutputFold_NoMass,
            'OLinear_SALCM_BCIF_OutputFold_Unbalanced':
                OLinear_SALCM_BCIF_OutputFold_Unbalanced,
            # Kept for strict old-checkpoint replay; the runner exposes no BCIF arm.
            'OLinear_SALCM_BCIF': LEGACY_MODELS['OLinear_SALCM_BCIF'],
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            # CUDA_VISIBLE_DEVICES is supplied by the caller. args.gpu is a
            # logical index within that selection; keep training/replay aligned.
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
