"""Synthetic CPU checks for the five SAER backbone adapters."""

from types import SimpleNamespace
import torch
from model.saer_adapters.registry import BACKBONE_MODELS
from model.saer_adapters.wrapper import LongContextBackbone


def main():
    torch.set_num_threads(1)
    for backbone in BACKBONE_MODELS:
        config=SimpleNamespace(backbone=backbone,arm='sa_frontend',long_len=768,
            seq_len=96,pred_len=24,enc_in=3,individual=0,e_layers=1,n_heads=2,
            d_model=16,d_ff=32,dropout=0.0,fc_dropout=0.0,head_dropout=0.0,
            patch_len=16,stride=8,padding_patch='end',revin=0,affine=0,
            subtract_last=0,decomposition=0,kernel_size=25,output_attention=False,
            use_norm=1,embed='timeF',freq='h',class_strategy='projection',factor=1,
            activation='gelu',cycle=24,model_type='linear',use_revin=1,
            task_name='long_term_forecast',label_len=0,c_out=3)
        torch.manual_seed(2021)
        model=LongContextBackbone(config)
        output=model(torch.randn(2,768,3),x_mark_enc=torch.randn(2,768,4),
                     cycle_index=torch.tensor([0,3]))
        assert output.shape==(2,24,3) and torch.isfinite(output).all()
        output.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        print(f'PASS: {backbone} + SAER CPU forward/backward')
    print('Synthetic fixtures only; no paper metrics were evaluated.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
