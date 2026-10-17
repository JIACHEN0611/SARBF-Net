"""Build a source-only release and an exact update list against an archived tree."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

SKIP_DIRS = {'.git', '.worktrees', '.superpowers', '.venv', 'venv', '__pycache__',
             '.pytest_cache', '.mypy_cache', '.idea', '.vscode', 'runs', 'results',
             'test_results', 'checkpoints', 'artifacts', 'dataset_template',
             'dataset_pilot', 'replay_output'}
BINARY_SUFFIXES = {'.pt', '.pth', '.ckpt', '.npy', '.npz', '.csv', '.tsv', '.parquet',
                   '.zip', '.7z', '.gz', '.tar', '.pyc', '.pyo', '.log'}
VERSION = '1.1.0'


def _excluded(path):
    parts=path.parts
    if any(p in SKIP_DIRS for p in parts):
        return True
    if parts[:2] == ('docs', 'superpowers'):
        return True
    if parts[0] == 'dataset' and path.as_posix() != 'dataset/README.md':
        return True
    if any(p.startswith('.') and p not in {'.github', '.gitignore', '.gitattributes'} for p in parts):
        return True
    return path.suffix.lower() in BINARY_SUFFIXES or '.7z.' in path.name.lower()


def source_files(root):
    """Return regular source files, rejecting links rather than dereferencing them."""
    files={}
    for parent, dirs, names in os.walk(root, followlinks=False):
        relative=Path(parent).relative_to(root)
        retained=[]
        for name in dirs:
            rel=relative/name
            # dataset itself is traversed only for its README.
            skip=_excluded(rel) and rel.as_posix()!='dataset'
            if skip:
                continue
            if (root/rel).is_symlink():
                raise ValueError(f'symlink in source: {rel}')
            retained.append(name)
        dirs[:]=retained
        for name in names:
            rel=relative/name
            if _excluded(rel):
                continue
            path=root/rel
            if path.is_symlink():
                raise ValueError(f'symlink in source: {rel}')
            if path.is_file():
                files[rel.as_posix()]=path
    return dict(sorted(files.items()))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def refresh_source_locks(root):
    files=source_files(root)
    pilot=root/'tools/sa_lcm_transport/run_pilot.py'
    if pilot.is_file():
        source=pilot.read_text(encoding='utf-8')
        assignment=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=='SOURCE_FILES' for t in n.targets))
        lines=source.splitlines(keepends=True)
        replacement='SOURCE_FILES = (\n'+''.join(f'    {name!r},\n' for name in files if name.endswith('.py'))+')\n'
        lines[assignment.lineno-1:assignment.end_lineno]=[replacement]
        pilot.write_text(''.join(lines),encoding='utf-8')
    model_hashes={name:_hash(path) for name,path in files.items()
                  if name.endswith('.py') and (name.startswith(('model/','layers/','ablations/models/')))}
    _write_json(root/'MODEL_LAYER_SHA256.json',model_hashes)


def build_release(repository_root:Path, baseline_root:Path, output_dir:Path)->dict:
    root,base,out=(Path(p).resolve() for p in (repository_root,baseline_root,output_dir))
    if not root.is_dir() or not base.is_dir():
        raise ValueError('repository and baseline must be existing directories')
    if out.is_relative_to(root) or out.is_relative_to(base):
        raise ValueError('output directory must be outside source and baseline trees')
    # Complete inspection before changing locks or writing outputs.
    source_files(root)
    previous={name:_hash(path) for name,path in source_files(base).items()}
    refresh_source_locks(root)
    files=source_files(root)
    manifest_path=root/'RELEASE_MANIFEST.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(release=f'SARBF-Net source v{VERSION}',assets_included=False,
                    asset_release='https://github.com/JIACHEN0611/SARBF-Net/releases/tag/v1.0.0',
                    transfer_preset_count=720,
                    files={name:_hash(path) for name,path in files.items() if name!='RELEASE_MANIFEST.json'})
    _write_json(manifest_path,manifest)
    files=source_files(root)
    current={name:_hash(path) for name,path in files.items()}
    changes={
        'version':VERSION,'baseline':'SARBF-Net_Assets_Prepare_v3 source',
        'added':sorted(current.keys()-previous.keys()),
        'modified':sorted(name for name in current.keys()&previous.keys() if current[name]!=previous[name]),
        'deleted':sorted(previous.keys()-current.keys()),
        'sha256':current,
        'asset_action':'Reuse v1.0.0 assets unchanged; do not replace or delete the four volumes.',
    }
    out.mkdir(parents=True,exist_ok=True)
    archive=out/f'SARBF-Net_Source_v{VERSION}.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,path in files.items():
            entry=zipfile.ZipInfo('SARBF-Net/'+name,date_time=(2026,10,17,0,0,0))
            entry.compress_type=zipfile.ZIP_DEFLATED
            entry.external_attr=0o100644<<16
            z.writestr(entry,path.read_bytes())
    changes_path=out/'SARBF-Net_Update_Changes.json'
    _write_json(changes_path,changes)
    guide=root/'docs/GITHUB_UPDATE_zh-CN.md'
    if guide.exists():
        shutil.copyfile(guide,out/'SARBF-Net_GitHub_Update_zh-CN.md')
    return {'archive':str(archive),'changes':str(changes_path),'file_count':len(files),
            'archive_sha256':_hash(archive),'added':len(changes['added']),
            'modified':len(changes['modified']),'deleted':len(changes['deleted'])}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(build_release(Path(__file__).resolve().parents[1],args.baseline_root,args.output_dir),indent=2))


if __name__=='__main__':
    main()
