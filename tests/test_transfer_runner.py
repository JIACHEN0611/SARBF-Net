import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


def modules():
    assert (ROOT/'tools/saer_transfer/run.py').is_file(), 'Transfer runner missing'
    from tools.saer_transfer import profiles,run
    return profiles,run


def test_matrix_filters_and_preview_have_no_side_effects(tmp_path):
    profiles,run=modules()
    tasks=profiles.build_tasks()
    assert len(tasks)==720 and len({x.task_id for x in tasks})==720
    assert {t.backbone for t in tasks}=={'DLinear','PatchTST','iTransformer','CycleNet','TSMixer'}
    root=tmp_path/'space directory'
    args=['--backbone','DLinear','--dataset','ETTm1','--horizon','96','--seed','2021',
          '--arm','sa_frontend','--run-root',str(root),'--dry-run']
    assert run.main(args)==0
    assert not root.exists()
    assert run.main(args[:-1]+['--list-tasks'])==0
    assert not root.exists()
    task=next(t for t in tasks if t.backbone=='DLinear')
    command=run.build_command(task,sys.executable,tmp_path/'data space')
    assert command[:4]==[sys.executable,'-u','-m','experiments.saer_transfer.train']
    assert command[command.index('--root_path')+1]==str((tmp_path/'data space').resolve())
    result=subprocess.run([sys.executable,'-m','tools.saer_transfer.run',*args],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.count('RUN ')==1


def test_completion_rejects_corrupt_nonfinite_and_failed(tmp_path):
    _,run=modules()
    record=dict(mse=.2,mae=.3,params=10,best_epoch=1)
    path=tmp_path/'metrics.json'
    for text in ['{',json.dumps({**record,'mse':float('nan')}),json.dumps({**record,'best_epoch':None})]:
        path.write_text(text)
        assert not run.is_complete(tmp_path)
    path.write_text(json.dumps(record));assert run.is_complete(tmp_path)
    (tmp_path/'failed.json').write_text('{}');assert not run.is_complete(tmp_path)


def test_existing_failed_directory_is_not_overwritten(tmp_path):
    profiles,run=modules()
    task=next(t for t in profiles.build_tasks() if t.backbone=='DLinear' and t.dataset=='ETTm1'
              and t.horizon==96 and t.seed==2021 and t.arm=='sa_frontend')
    folder=run.task_directory(tmp_path,task);folder.mkdir(parents=True)
    (folder/'failed.json').write_text('old failure')
    with pytest.raises(FileExistsError):
        run.main(['--backbone','DLinear','--dataset','ETTm1','--horizon','96','--seed','2021',
                  '--arm','sa_frontend','--run-root',str(tmp_path)])
    assert (folder/'failed.json').read_text()=='old failure'


def test_shell_entry_preserves_caller_relative_paths(tmp_path):
    caller=tmp_path/'external caller with spaces';caller.mkdir()
    environment=dict(os.environ,PYTHON=sys.executable,PYTHONPATH='')
    result=subprocess.run(['bash',str(ROOT/'scripts/run_saer_transfer_matrix.sh'),
        '--backbone','DLinear','--dataset','ETTm1','--horizon','96','--seed','2021',
        '--arm','sa_frontend','--data-root','data folder','--run-root','output folder',
        '--dry-run'],cwd=caller,env=environment,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    command=json.loads(result.stdout.strip().removeprefix('RUN '))
    assert command[command.index('--root_path')+1]==str(caller/'data folder')
    assert command[command.index('--output_root')+1].startswith(str(caller/'output folder'))
    assert not (caller/'output folder').exists()


def test_relative_python_executable_resolves_before_child_cwd(tmp_path,monkeypatch):
    _,run=modules()
    caller=tmp_path/'external caller with spaces';caller.mkdir()
    interpreter=caller/'bin folder'/'python stub';interpreter.parent.mkdir()
    interpreter.write_text('#!/bin/sh\nprintf \'%s\\n\' \'{"mse":0.2,"mae":0.3,"params":10,"best_epoch":1}\' > "$SA_FRONTEND_RESULT_JSON"\n')
    interpreter.chmod(0o755)
    data=caller/'data folder';data.mkdir();(data/'ETTm1.csv').write_text('placeholder')
    monkeypatch.chdir(caller)
    assert run.main(['--backbone','DLinear','--dataset','ETTm1','--horizon','96','--seed','2021',
        '--arm','original','--python','./bin folder/python stub','--data-root','data folder',
        '--run-root','output folder'])==0
    metadata=next((caller/'output folder').rglob('task.json'))
    command=json.loads(metadata.read_text())['command']
    assert command[0]==str(interpreter)
    assert command[command.index('--root_path')+1]==str(data)


@pytest.mark.parametrize('value',[None,42,['mse','mae','params','best_epoch']])
def test_non_object_metrics_are_incomplete(tmp_path,value):
    _,run=modules()
    (tmp_path/'metrics.json').write_text(json.dumps(value))
    assert not run.is_complete(tmp_path)
