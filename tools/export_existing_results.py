"""Export recorded per-run test metrics. Does not train or reconstruct seed values."""
import argparse,csv,json,re,hashlib
from pathlib import Path
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 rows=[];csvs=[];errors=[]
 for root_id,root in enumerate(a.root):
  if not root.is_dir():errors.append({'root_index':root_id,'error':'directory not found'});continue
  csvs += [{'root_index':root_id,'file':str(f.relative_to(root))} for f in sorted(root.rglob('*.csv'))]
  for f in sorted(root.rglob('result.json')):
   try:
    raw=f.read_bytes();d=json.loads(raw);test=d.get('test')
    if not isinstance(test,dict):test=d.get('metrics',{})
    mse=test.get('mse',d.get('test_mse'));mae=test.get('mae',d.get('test_mae'))
    rel=str(f.relative_to(root));cfg=d.get('args',d.get('config',{}));cfg=cfg if isinstance(cfg,dict) else {}
    seed=d.get('seed',cfg.get('seed'));match=re.search(r'(?:seed|random_seed)[_-]?(\d+)',rel)
    if seed is None and match:seed=int(match[1])
    if seed is None or mse is None or mae is None:continue
    if not isinstance(mse,(int,float)) or not isinstance(mae,(int,float)):continue
    rows.append({'root_index':root_id,'source_relative_path':rel,'source_sha256':hashlib.sha256(raw).hexdigest(),'seed':seed,'task':d.get('task',cfg.get('task','')),'model':d.get('model',cfg.get('model','')),'variant':d.get('arm',d.get('variant','')),'dataset':cfg.get('data_name',cfg.get('data','')),'horizon':cfg.get('pred_len',''),'test_mse':mse,'test_mae':mae})
   except (ValueError,OSError,TypeError) as e:errors.append({'root_index':root_id,'file':str(f.relative_to(root)),'error':str(e)})
 a.output.parent.mkdir(parents=True,exist_ok=True)
 fields=['root_index','source_relative_path','source_sha256','seed','task','model','variant','dataset','horizon','test_mse','test_mae']
 with a.output.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 report={'exported_seed_records':len(rows),'existing_csv_files':csvs,'errors':errors}
 a.output.with_suffix('.export_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'exported_seed_records':len(rows),'existing_csv_files':len(csvs),'errors':len(errors)}))
if __name__=='__main__':main()
