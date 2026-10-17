"""Compare independently imported source trees on synthetic CPU fixtures."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def compare_records(reference, candidate, path='record'):
    if type(reference) is not type(candidate):
        raise ValueError(f'{path}: type mismatch')
    if isinstance(reference, dict):
        if reference.keys() != candidate.keys():
            raise ValueError(f'{path}: keys mismatch')
        for key in reference:
            compare_records(reference[key], candidate[key], f'{path}.{key}')
    elif isinstance(reference, list):
        if len(reference) != len(candidate):
            raise ValueError(f'{path}: length mismatch')
        for i, (left, right) in enumerate(zip(reference, candidate)):
            compare_records(left, right, f'{path}[{i}]')
    elif reference != candidate:
        raise ValueError(f'{path}: value mismatch')


PROBE = r'''
import hashlib, importlib, json, runpy, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
torch.set_num_threads(1)
kind, output_dir, peer_dir = sys.argv[1:]
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
def tensor(t):
    value = t.detach().cpu().contiguous()
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest()}
def cfg_transfer(backbone, arm):
    return SimpleNamespace(backbone=backbone, arm=arm, long_len=768, seq_len=96,
        pred_len=24, enc_in=3, individual=0, e_layers=1, n_heads=2,
        d_model=16, d_ff=32, dropout=0.0, fc_dropout=0.0, head_dropout=0.0,
        patch_len=16, stride=8, padding_patch='end', revin=0, affine=0,
        subtract_last=0, decomposition=0, kernel_size=25,
        output_attention=False, use_norm=1, embed='timeF', freq='h',
        class_strategy='projection', factor=1, activation='gelu', cycle=24,
        model_type='linear', use_revin=1, task_name='long_term_forecast',
        label_len=0, c_out=3)
records = {}
for seed in (2021, 2022, 2023):
    cases = [('main','standard')] if kind == 'main' else [
        (b,a) for b in ('DLinear','PatchTST','iTransformer','CycleNet','TSMixer')
        for a in ('original','sa_frontend')]
    for backbone, arm in cases:
        key = f'{backbone}-{arm}-{seed}'
        with tempfile.TemporaryDirectory() as tmp:
            if kind == 'main':
                cfg = runpy.run_path('tests/test_final_release.py')[
                    'FinalReleaseNumericalTests']()._configs(Path(tmp), horizon=7)
                cls = importlib.import_module('model.SARBF_Net').Model
            else:
                cfg = cfg_transfer(backbone,arm)
                name = ('model.saer_adapters.wrapper' if Path('model/saer_adapters/wrapper.py').is_file()
                        else 'models.LongContextBackbone')
                cls = importlib.import_module(name).LongContextBackbone
            torch.manual_seed(seed)
            model = cls(cfg).cpu().eval()
            rng = tensor(torch.get_rng_state())
            if peer_dir:
                model.load_state_dict(torch.load(Path(peer_dir)/(key+'.pt'),
                    map_location='cpu',weights_only=True), strict=True)
            initial = {k:tensor(v) for k,v in model.state_dict().items()}
            torch.save(model.state_dict(),output_dir/(key+'.pt'))
            aliases={}
            for name, parameter in model.named_parameters(remove_duplicate=False):
                aliases.setdefault(id(parameter),[]).append(name)
            shared = sorted(sorted(names) for names in aliases.values() if len(names)>1)
            torch.manual_seed(99)
            x = torch.randn(2,768,3)
            if kind == 'main':
                prediction=model(x)
            else:
                marks=torch.randn(2,768,4)
                prediction=model(x,x_mark_enc=marks,cycle_index=torch.tensor([0,3]))
            prediction.square().mean().backward()
            gradients={k:tensor(p.grad) for k,p in model.named_parameters() if p.grad is not None}
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
            optimizer.step()
            records[key]={'rng':rng,'shared':shared,'state':initial,
                'prediction':tensor(prediction),'gradients':gradients,
                'updated_state':{k:tensor(v) for k,v in model.state_dict().items()}}
(output_dir/'records.json').write_text(json.dumps(records,sort_keys=True))
'''


def capture(root, kind, output_dir, peer_dir=None):
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    environment.pop('PYTHONPATH', None)
    subprocess.run([sys.executable, '-c', PROBE, kind, str(output_dir),
                    str(peer_dir) if peer_dir else ''], cwd=root, env=environment,
                   check=True, capture_output=True, text=True)
    return json.loads((output_dir / 'records.json').read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--kind', choices=['main', 'transfer'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            left = capture(args.reference_root.resolve(), args.kind, folder / 'reference')
            right = capture(args.candidate_root.resolve(), args.kind, folder / 'candidate')
            compare_records(left, right)
            cross_left = capture(args.reference_root.resolve(), args.kind,
                                 folder / 'cross_reference', folder / 'candidate')
            cross_right = capture(args.candidate_root.resolve(), args.kind,
                                  folder / 'cross_candidate', folder / 'reference')
            compare_records(left, cross_left)
            compare_records(left, cross_right)
            report = dict(status='PASS', kind=args.kind, cases=len(left),
                          seeds=[2021, 2022, 2023], exact=True,
                          strict_cross_loading=True, real_checkpoints_evaluated=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
        return 0
    except subprocess.CalledProcessError as error:
        print(error.stderr or str(error), file=sys.stderr)
        return 1
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
