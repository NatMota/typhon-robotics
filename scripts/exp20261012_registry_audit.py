"""Evidence registry from existing checkpoint headers; no training or model mutation."""
import sys,json,hashlib,os,time
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];OUT=Path('/var/tmp/typhon-ost-sweep-20261012/audits');OUT.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def trim(x):
 if torch.is_tensor(x):return {'tensor_shape':list(x.shape),'dtype':str(x.dtype)}
 if isinstance(x,dict):return {str(k):trim(v)for k,v in x.items()if k not in ['model','optimizer','torch_rng','cuda_rng','rng','expected','inputs','prompt','lora_state']}
 if isinstance(x,(list,tuple)):return [trim(v)for v in x]
 if isinstance(x,Path):return str(x)
 return x
paths=sorted(Path('/var/tmp/humanoid-m7-checkpoints-20261009').glob('*.pt'))
paths+=sorted((R/'frozen_results/personal-m3-20261009/training').glob('*/prompt.pt'))
paths+=sorted((R/'frozen_results/personal-m4-20261009/training').glob('*/prompt.pt'))
b=json.loads(Path('/var/tmp/typhon-research-20261011/broad-pt/evaluation-protocol.json').read_text())
paths += [Path(v['checkpoint'])for v in b['methods'].values()]
rows=[]
for p in paths:
 if not p.exists():rows.append({'path':str(p),'status':'not_online'});continue
 try:
  c=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
  rows.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'header':trim(c),'model_tensor_count':len(c.get('model',{})),'optimizer_present':'optimizer'in c,'status':'header_and_hash_verified'});del c
 except Exception as e:rows.append({'path':str(p),'status':'read_error','error':repr(e)})
 (OUT/'model-registry.json').write_text(json.dumps({'base':json.loads((R/'.cache-xvla/checkpoint/hub_manifest.json').read_text()),'models':rows,'created_unix_s':time.time()},indent=2,default=str))
 print('AUDITED',p.name,rows[-1]['status'],flush=True)
