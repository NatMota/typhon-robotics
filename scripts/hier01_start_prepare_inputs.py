"""Prepare native actor tensors without loading a policy or generating actions."""
import os,sys,json,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
import numpy as np
import torch
from transformers import AutoProcessor
from supervisor.adaptation.observations import views,object_crop
from supervisor.adaptation.m6_contract import reorder,DOMAIN
from load04_common import array_sha
O=Path('/var/tmp/typhon-hier01-start-20261015')
p=json.loads((O/'protocol-draft.json').read_text());assert time.time()<p['clock']['preparation_deadline']
processor=AutoProcessor.from_pretrained('.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);processor.language_max_length=128
receipts=[]
for tag,c in p['cases'].items():
 z=np.load(c['initial_archive']);imgs=views({'primary':z['primary'],'secondary':z['secondary']},'O2');v=processor(images=imgs,language_instruction=c['instruction']);v['proprio']=reorder(torch.tensor(z['proprio'],dtype=torch.float32).unsqueeze(0));v['domain_id']=torch.tensor([DOMAIN],dtype=torch.long)
 v={k:a.cpu().contiguous().numpy() for k,a in v.items()};assert v['image_mask'].all();assert np.isfinite(v['image_input']).all()
 np.savez_compressed(c['source_prepared_archive'],**v);c['input_hashes']={k:array_sha(a) for k,a in v.items()};c['expected_initial_nontext']={k:x for k,x in c['input_hashes'].items() if k!='input_ids'}
 meta=object_crop(z['primary'])[1];assert meta['detected'] and not any(meta['padding_ltrb'])
 if c['cell']=='REF':
  old=Path('/var/tmp/typhon-hier01-dev-20261015/workers/demo_0012-loading-'+str(c['policy_seed'])+'-router/episode/input-audit/initial-prepared.npz')
  with np.load(old) as q:
   for k in v:np.testing.assert_array_equal(v[k],q[k])
 receipts.append({'id':tag,'input_hashes':c['input_hashes'],'crop_metadata':meta,'image_shape':list(v['image_input'].shape),'mask':v['image_mask'].tolist(),'standalone_actor_input_exact':c['cell']=='REF'})
assert len({c['instruction'] for c in p['cases'].values()})==1
for seed in [912171,912172]:
 ids=[c['input_hashes']['input_ids'] for c in p['cases'].values() if c['policy_seed']==seed];assert len(set(ids))==1
p['frozen_epoch']=time.time();assert p['frozen_epoch']<p['clock']['preparation_deadline']
(O/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
(O/'qualification.json').write_text(json.dumps({'passed':True,'scene_receipt':'scene-qualification.json','inputs':receipts,'policy_calls':0,'frozen_epoch':p['frozen_epoch']},indent=2)+'\n')
print('NATIVE_INPUTS_QUALIFIED_PROTOCOL_FROZEN')
