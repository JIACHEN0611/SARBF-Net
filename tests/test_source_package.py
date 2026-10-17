"""Exercise source packaging through real files and a simulated baseline update."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest


def test_source_archive_and_update_reconstruct_exact_tree(tmp_path):
    from scripts.build_source_release import build_release
    repo, baseline, output = (tmp_path/n for n in ('repo','baseline','delivery'))
    def put(root, name, content):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
    put(repo,'run.py','print("current")\n')
    put(repo,'model/new.py','x=1\n')
    put(repo,'.github/workflows/tests.yml','name: tests\n')
    put(repo,'.gitignore','runs/\n')
    put(repo,'dataset/README.md','Data sources\n')
    for name in ['.git/config','.superpowers/ledger','docs/superpowers/plan.md','runs/seed/log.txt','artifacts/q/a.npy','dataset/raw.csv','dataset_pilot/raw.txt','weights.pth','part.7z.001','__pycache__/a.pyc']:
        put(repo,name,'excluded')
    put(baseline,'run.py','print("old")\n')
    put(baseline,'old.py','remove\n')
    result=build_release(repo,baseline,output)
    with zipfile.ZipFile(result['archive']) as z:
        assert all(n.startswith('SARBF-Net/') for n in z.namelist())
        files={n.removeprefix('SARBF-Net/'):z.read(n) for n in z.namelist()}
    assert {'run.py','model/new.py','.github/workflows/tests.yml','.gitignore','dataset/README.md'}<=files.keys()
    assert not any(n.startswith(('.git/','.superpowers/','runs/','artifacts/','docs/superpowers/')) for n in files)
    assert not any(n.endswith(('.csv','.pth','.npy','.001','.pyc')) for n in files)
    manifest=json.loads(files['RELEASE_MANIFEST.json'])
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in files.items() if name!='RELEASE_MANIFEST.json'}
    assert manifest['files']==hashes
    changes=json.loads(Path(result['changes']).read_text())
    reconstructed={'run.py':b'print("old")\n','old.py':b'remove\n'}
    for name in changes['deleted']:reconstructed.pop(name)
    for name in changes['added']+changes['modified']:reconstructed[name]=files[name]
    assert reconstructed==files
    assert json.loads(files['MODEL_LAYER_SHA256.json'])=={'model/new.py':hashes['model/new.py']}


def test_refuses_nested_output_or_symlinked_source(tmp_path):
    from scripts.build_source_release import build_release
    repo=tmp_path/'repo';repo.mkdir();(repo/'run.py').write_text('')
    baseline=tmp_path/'base';baseline.mkdir()
    with pytest.raises(ValueError,match='outside'):
        build_release(repo,baseline,repo/'delivery')
    target=tmp_path/'outside.py';target.write_text('secret')
    (repo/'linked.py').symlink_to(target)
    with pytest.raises(ValueError,match='symlink'):
        build_release(repo,baseline,tmp_path/'out')
