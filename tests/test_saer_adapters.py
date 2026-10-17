from pathlib import Path
from types import SimpleNamespace
import importlib

import pytest
import torch

BACKBONES = ('DLinear', 'PatchTST', 'iTransformer', 'CycleNet', 'TSMixer')


def config(backbone='DLinear', arm='sa_frontend'):
    return SimpleNamespace(backbone=backbone, arm=arm, long_len=768, seq_len=96,
        pred_len=24, enc_in=3, individual=0, e_layers=1, n_heads=2,
        d_model=16, d_ff=32, dropout=0.0, fc_dropout=0.0, head_dropout=0.0,
        patch_len=16, stride=8, padding_patch='end', revin=0, affine=0,
        subtract_last=0, decomposition=0, kernel_size=25,
        output_attention=False, use_norm=1, embed='timeF', freq='h',
        class_strategy='projection', factor=1, activation='gelu', cycle=24,
        model_type='linear', use_revin=1, task_name='long_term_forecast',
        label_len=0, c_out=3)


def wrapper():
    assert (Path(__file__).resolve().parents[1]/'model/saer_adapters/wrapper.py').is_file(), 'SAER adapter missing'
    from model.saer_adapters.wrapper import LongContextBackbone
    return LongContextBackbone


@pytest.mark.parametrize('backbone', BACKBONES)
@pytest.mark.parametrize('arm', ['original','sa_frontend'])
def test_each_backbone_arm_shape_and_strict_loading(backbone, arm):
    cls=wrapper()
    torch.manual_seed(2021)
    model=cls(config(backbone,arm)).eval()
    clone=cls(config(backbone,arm)).eval()
    clone.load_state_dict(model.state_dict(),strict=True)
    x=torch.randn(2,768,3);marks=torch.randn(2,768,4);cycle=torch.tensor([0,3])
    result=model(x,x_mark_enc=marks,cycle_index=cycle)
    assert result.shape==(2,24,3) and torch.isfinite(result).all()
    torch.testing.assert_close(result,clone(x,x_mark_enc=marks,cycle_index=cycle),rtol=0,atol=0)
    assert (model.frontend is None)==(arm=='original')


@pytest.mark.parametrize('backbone', BACKBONES)
def test_optimizer_updates_saer_and_public_entry(backbone):
    cls=wrapper()
    cfg=config(backbone)
    model=cls(cfg)
    before=model.frontend.refiner.correction[-1].weight.detach().clone()
    loss=model(torch.randn(2,768,3),x_mark_enc=torch.randn(2,768,4),cycle_index=torch.tensor([0,3])).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.optim.Adam(model.parameters(),lr=1e-3).step()
    assert not torch.equal(before,model.frontend.refiner.correction[-1].weight)
    public=importlib.import_module('model.saer_adapters.'+backbone).Model(cfg)
    public.load_state_dict(model.state_dict(),strict=True)
    assert cfg.backbone==backbone


def test_marks_cycle_and_invalid_geometry():
    cls=wrapper()
    for backbone in ('iTransformer','CycleNet'):
        model=cls(config(backbone,'original')).eval()
        x=torch.randn(2,768,3);marks=torch.randn(2,768,4);cycle=torch.tensor([0,3])
        actual=model(x,x_mark_enc=marks,cycle_index=cycle)
        expected=(model.backbone(x[:,-96:],cycle) if backbone=='CycleNet'
                  else model.backbone(x[:,-96:],marks[:,-96:],None,None,mask=None))
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        with pytest.raises(ValueError): model(x[:,:767],x_mark_enc=marks,cycle_index=cycle)
    with pytest.raises(ValueError,match='cycle_index'): model(x)
    with pytest.raises(ValueError): cls(config('Unknown'))
    with pytest.raises(ValueError): cls(config('DLinear','unknown'))


def test_main_and_transfer_import_order_is_independent(tmp_path):
    cls=wrapper()
    from test_final_release import FinalReleaseNumericalTests
    from model.SARBF_Net import Model
    cfg=FinalReleaseNumericalTests()._configs(tmp_path,horizon=7)
    torch.manual_seed(2021);before=Model(cfg).eval()
    cls(config('PatchTST'))
    torch.manual_seed(2021);after=Model(cfg).eval()
    x=torch.randn(2,768,3)
    torch.testing.assert_close(before(x),after(x),atol=0,rtol=0)
