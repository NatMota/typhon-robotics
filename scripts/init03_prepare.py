import os,sys,json,gzip,collections,hashlib,time
from pathlib import Path
W=Path(__file__).resolve().parents[1];os.chdir(W)
sys.path[:0]=[str(W),str(W/'.m5-deps'),str(W/'.m6-deps')]
import numpy as np,torch
from transformers import AutoProcessor
from supervisor.adaptation.observations import views
from supervisor.adaptation.m6_contract import reorder
O=Path('/var/tmp/typhon-init03-20261015');D=Path('/var/tmp/typhon-init02-20261015')
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def ah(v):a=np.ascontiguousarray(v);return hashlib.sha256(str((a.shape,a.dtype)).encode()+a.tobytes()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 bank=json.loads((D/'data/qualified-dataset.json').read_text());wins=bank['windows']
 plan=json.load(gzip.open(D/'data/sampling-plan.json.gz','rt'))['plan']
 counts={k:collections.Counter(i for row in v for i in row)for k,v in plan.items()}
 proc=AutoProcessor.from_pretrained(W/'.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);proc.language_max_length=128
 packets={};audit=[];valid=True
 for parent in ['demo_0001','demo_0002','demo_0003','demo_0005']:
  row={'parent':parent,'tasks':{}};pair={}
  for task,suffix in [('loading','L-teacher'),('feeding','F-preserve')]:
   e=next(e for e in bank['episodes']if e['id']==parent+'-'+suffix)
   wi=min((i for i,w in enumerate(wins)if w['episode']==e['id']),key=lambda i:(wins[i]['start'],wins[i]['frame']))
   w=wins[wi];assert sha(e['absolute_path'])==e['sha256']
   with np.load(e['absolute_path'])as z:
    frame=w['frame'];start=w['start']
    b=dict(proc(images=views({k:z[v][frame]for k,v in e['views'].items()},'O2'),language_instruction=e['instruction']))
    b.update(proprio=reorder(torch.from_numpy(z['proprio'][frame].copy())).unsqueeze(0),action=reorder(torch.from_numpy(z['targets'][start:start+30].copy())).unsqueeze(0),domain_id=torch.tensor([29]))
    assert b['action'].shape==(1,30,20)
    row['tasks'][task]={'episode':e['id'],'source_sha256':e['sha256'],'window':w,'window_index':wi,'exposures':{k:counts[k][wi]for k in counts},'instruction':e['instruction'],'hashes':{k:ah(v.numpy())for k,v in b.items()},'initial_gripper_labels':b['action'][0,:,9].tolist(),'source_frame_action_index':int(z['frame_action_indices'][frame])}
   pair[task]=b;packets[parent+'-'+task]=b
  same={k:torch.equal(pair['loading'][k],pair['feeding'][k])for k in ['image_input','image_mask','proprio','domain_id']}
  row['effective_nontext_equal']=same;row['same_action_instant']=row['tasks']['loading']['window']['start']==row['tasks']['feeding']['window']['start']==0
  row['conflicting_inherited_grasp_verb_in_hold_instruction']='Grasp' in row['tasks']['feeding']['instruction']
  row['runtime_matches']={}
  for model in ['R','T']:
   for task in ['feeding','loading']:
    for seed in [912171,912172]:
     q=D/'episodes'/model/f'init-{parent}-{task}-{seed}'/'input-audit/initial-prepared.npz'
     with np.load(q)as z:
      row['runtime_matches'][f'{model}-{task}-{seed}']={k:bool(np.array_equal(v.numpy(),z[k])) for k,v in pair[task].items()if k!='action'}
  valid &= all(same.values()) and row['same_action_instant']
  audit.append(row)
 torch.save(packets,O/'packets.pt')
 result={'passed':bool(valid),'epoch':time.time(),'audit':audit,'no_training':True,'short_prefix':10,'model_dtype':'float32 weights; bf16 autocast','refinements':10,'pure_noise_time':1.0,'intermediate_times':[0.5,0.1]}
 dump(O/'packet-qualification.json',result)
 print(json.dumps({'passed':valid,'pairs':len(audit),'runtime_equal':all(all(v.values())for r in audit for v in r['runtime_matches'].values()),'epoch':time.time()}),flush=True)
 if not valid:raise RuntimeError('Aligned shared-observation contract failed; stop')
if __name__=='__main__':main()
