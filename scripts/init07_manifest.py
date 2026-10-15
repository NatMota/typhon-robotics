"""Make the final raw archive manifest after independent checks and visual review."""
import sys,time,json,shutil
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path[:0]=[str(W),str(W/'scripts')]
from init07_common import OUT,read,dump,sha
def main():
 assert read(OUT/'postprocess-complete.json')['passed']
 assert read(OUT/'visual-review.json')['passed']
 assert not (OUT/'raw-manifest.json').exists()
 frozen=read(OUT/'execution-freeze.json')['source_sha256']
 for name,value in frozen.items():assert sha(W/name)==value,name
 source=OUT/'source'
 for name in sorted(set(frozen)|{str(p.relative_to(W))for p in (W/'scripts').glob('init07*.py')}):
  dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True)
  assert not dest.exists();shutil.copy2(W/name,dest)
 deps=[]
 for e in read(OUT/'data/qualified-dataset.json')['episodes']:
  f=Path(e['absolute_path']);assert sha(f)==e['sha256']
  deps.append({'id':e['id'],'path':str(f),'bytes':f.stat().st_size,'sha256':e['sha256']})
 dump(OUT/'external-data-dependencies.json',{'training_episodes':deps,'historical_T_root':'/var/tmp/typhon-init02-20261015','Mint_historical_T_archive':'/media/nm/Fast Drive/Typhon/research-20261015/init02-matched-adaptation','note':'Historical T episodes are reused and remain separately archived. Training episode paths are original frozen dependencies, not duplicated into this new result.'})
 checkpoint={}
 for model in ['Q0','Q1']:
  m=read(OUT/f'checkpoint-archive-{model}-manifest.json')
  for row in m['files']:checkpoint[row['path']]=row
 files=[]
 for f in sorted(OUT.rglob('*')):
  if f.is_dir():continue
  assert f.is_file()and not f.is_symlink(),str(f)
  name=str(f.relative_to(OUT));digest=sha(f);size=f.stat().st_size
  if name in checkpoint:
   assert digest==checkpoint[name]['sha256']and size==checkpoint[name]['bytes'],name
  files.append({'path':name,'bytes':size,'sha256':digest,'checkpoint_archive':name in checkpoint})
 assert len(checkpoint)==12 and len([r for r in files if r['checkpoint_archive']])==12
 dump(OUT/'raw-manifest.json',{'schema':'init07-full-raw-v1','epoch':time.time(),'source_root':str(OUT),'files':files,'file_count':len(files),'total_bytes':sum(r['bytes']for r in files),'non_checkpoint_bytes':sum(r['bytes']for r in files if not r['checkpoint_archive'])})
 print(json.dumps({'files':len(files),'bytes':sum(r['bytes']for r in files),'stream_bytes':sum(r['bytes']for r in files if not r['checkpoint_archive'])}))
if __name__=='__main__':main()
