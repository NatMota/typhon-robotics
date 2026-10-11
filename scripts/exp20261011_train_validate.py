"""CPU-only postwrite integrity checks, separate from the running trainer."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import gc,hashlib,json,time,traceback,zipfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];OUT=Path('/var/tmp/typhon-research-20261011/broad-pt')
torch.set_num_threads(2)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
seen=set()
while True:
 for kind,path in [('recovery',OUT/'recovery-checkpoint.json')]+[('fixed',OUT/('checkpoint-'+str(s)+'.json')) for s in [3000,12000]]:
  if not path.exists():continue
  receipt=json.loads(path.read_text());key=(kind,int(receipt['step']),receipt['sha256'])
  if key in seen or receipt['step']<3000:continue
  target=Path(receipt.get('checkpoint',receipt.get('path')));start=time.time()
  try:
   assert target.is_file() and target.stat().st_size==receipt['bytes']
   assert zipfile.is_zipfile(target),'Truncated checkpoint container'
   with zipfile.ZipFile(target) as archive:
    count=len(archive.infolist());assert count>800
    assert archive.testzip() is None,'Checkpoint ZIP CRC failure'
   actual=sha(target);assert actual==receipt['sha256'],'Checkpoint digest differs from manifest'
   state=torch.load(target,map_location='cpu',weights_only=False)
   assert state['schema']=='m6-full-v1' and state['step']==receipt['step'] and state['domain_id']==29
   assert state['base_revision']=='c1c4a64a7e03ac5b95c468bf1578f3d03651b53b'
   assert state['source_manifest_sha256']==json.loads((OUT/'data-audit.json').read_text())['manifest_sha256']
   assert len(state['model'])>800
   assert all(torch.isfinite(t).all().item() for t in state['model'].values())
   result={'passed':True,'kind':kind,'step':receipt['step'],'path':str(target),'sha256':actual,'bytes':target.stat().st_size,'zip_crc_passed':True,'full_cpu_torch_load_passed':True,'all_model_tensors_finite':True,'model_keys':len(state['model']),'optimizer_states':len(state.get('optimizer',{}).get('state',{})),'cpu_only':True,'validated_unix_s':time.time(),'wall_seconds':time.time()-start}
   if kind=='recovery':
    assert result['optimizer_states']>800
    assert 'torch_rng' in state and 'cuda_rng' in state
   del state;gc.collect();seen.add(key)
  except Exception:result={'passed':False,'kind':kind,'step':receipt['step'],'path':str(target),'error':traceback.format_exc(),'validated_unix_s':time.time()};seen.add(key)
  dump(OUT/('checkpoint-validation-'+kind+'-'+str(receipt['step'])+'.json'),result);print(json.dumps(result),flush=True)
 if any(k[0]=='fixed' and k[1]==12000 for k in seen) and any(k[0]=='recovery' and k[1]==12000 for k in seen):break
 time.sleep(10)
