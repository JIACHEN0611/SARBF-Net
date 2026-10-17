import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


def test_training_reads_only_train_and_val_and_preserves_device():
    assert (Path(__file__).resolve().parents[1]/'experiments/saer_transfer/exp_main.py').is_file(), 'Transfer training missing'
    from experiments.saer_transfer.exp_main import Exp_Main
    from experiments.saer_transfer.exp_basic import Exp_Basic
    experiment=Exp_Main.__new__(Exp_Main)
    flags=[]
    def load(flag):
        flags.append(flag)
        return flag,flag+'-loader'
    experiment._get_data=load
    assert experiment._get_training_data()==('train','train-loader','val','val-loader')
    assert flags==['train','val']
    experiment.args=SimpleNamespace(use_gpu=True,gpu=0,use_multi_gpu=False,devices='0')
    with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'3'}):
        assert str(Exp_Basic._acquire_device(experiment))=='cuda:0'
        assert os.environ['CUDA_VISIBLE_DEVICES']=='3'


def test_long_windows_align_targets_and_cycle_index(tmp_path):
    assert (Path(__file__).resolve().parents[1]/'experiments/saer_transfer/data_factory.py').is_file(), 'Transfer loader missing'
    from experiments.saer_transfer.data_factory import data_provider
    from experiments.saer_transfer.long_context_factory import long_context_data_provider,make_data_args
    rows=5000
    pd.DataFrame({'date':pd.date_range('2020-01-01',periods=rows,freq='h'),
                  'sensor':np.arange(rows),'OT':np.arange(rows)*2}).to_csv(tmp_path/'data.csv',index=False)
    args=SimpleNamespace(data='custom',root_path=str(tmp_path),data_path='data.csv',embed='timeF',
        batch_size=8,num_workers=0,freq='h',features='M',target='OT',seq_len=96,
        label_len=0,pred_len=24,cycle=24,long_len=768)
    assert make_data_args(args).seq_len==768 and args.seq_len==96
    short,_=data_provider(args,'val');long,_=long_context_data_provider(args,'val')
    assert len(short)==len(long)
    np.testing.assert_array_equal(short[0][1],long[0][1])
    np.testing.assert_array_equal(short[0][0],long[0][0][-96:])
    assert short[0][4].item()==long[0][4].item()


def test_schedulers_keep_official_learning_rate():
    assert (Path(__file__).resolve().parents[1]/'experiments/saer_transfer/exp_main.py').is_file()
    from experiments.saer_transfer.exp_main import Exp_Main
    experiment=Exp_Main.__new__(Exp_Main)
    experiment.args=SimpleNamespace(lradj='type1',learning_rate=1e-4,pct_start=.3,train_epochs=10)
    optimizer=torch.optim.Adam([torch.nn.Parameter(torch.ones(1))],lr=1e-4)
    assert experiment._build_scheduler(optimizer,5) is None
    assert optimizer.param_groups[0]['lr']==1e-4
    experiment.args.lradj='TST'
    assert isinstance(experiment._build_scheduler(optimizer,5),torch.optim.lr_scheduler.OneCycleLR)


def test_training_cli_completes_one_cpu_epoch(tmp_path):
    import subprocess
    import sys
    import json
    rows=5000
    values=np.sin(np.arange(rows)/20)
    pd.DataFrame({'date':pd.date_range('2020-01-01',periods=rows,freq='h'),
                  'sensor':values,'OT':values+.1}).to_csv(tmp_path/'data.csv',index=False)
    output=tmp_path/'outputs'
    record=output/'metrics.json'
    command=[sys.executable,'-m','experiments.saer_transfer.train',
        '--model','LongContextBackbone','--backbone','DLinear','--arm','original',
        '--is_training','1','--model_id','cpu_check','--data','custom',
        '--root_path',str(tmp_path),'--data_path','data.csv','--enc_in','2','--dec_in','2',
        '--c_out','2','--seq_len','96','--long_len','768','--pred_len','24','--num_workers','0',
        '--train_epochs','1','--batch_size','128','--patience','1','--use_gpu','false',
        '--checkpoints',str(output/'checkpoints'),'--output_root',str(output)]
    result=subprocess.run(command,cwd=Path(__file__).resolve().parents[1],
        env={**os.environ,'SA_FRONTEND_RESULT_JSON':str(record),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
        text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    metrics=json.loads(record.read_text())
    assert metrics['best_epoch']==1 and np.isfinite(metrics['mse']) and np.isfinite(metrics['mae'])
    assert list((output/'checkpoints').glob('*/checkpoint.pth'))
