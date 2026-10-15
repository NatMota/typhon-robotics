"""Read-only audit of INIT01 actual first observations and candidate labels."""
import os,sys,json,gzip,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
import numpy as np
import torch
from transformers import AutoProcessor
from supervisor.adaptation.observations import views,object_crop
from supervisor.adaptation.m6_contract import reorder,DOMAIN
from load04_common import array_sha
O=Path('/var/tmp/typhon-init01-20261015');p=json.loads((O/'protocol.json').read_text())
assert (O/'complete.json').exists()
instructions=json.loads((O/'instruction-contract.json').read_text())
processor=AutoProcessor.from_pretrained('.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);processor.language_max_length=128
rows=[]
for parent in p['parents']:
 base=O/'continuations'/parent['id'];policy=base/'L-policy/episode'
 with np.load(policy/'input-audit/initial-prepared.npz') as z:reference={k:z[k].copy() for k in z.files}
 for branch in p['branches']:
  f=base/branch/'prefix/initial-live.npz';z=np.load(f);cfg=parent['entry']['config'] if branch=='F-preserve' else parent['physical_config']
  inp=processor(images=views({'primary':z['primary'],'secondary':z['secondary']},'O2'),language_instruction=instructions[parent['id']][branch])
  inp['proprio']=reorder(torch.tensor(z['proprio'],dtype=torch.float32).unsqueeze(0));inp['domain_id']=torch.tensor([DOMAIN],dtype=torch.long)
  v={k:x.cpu().contiguous().numpy() for k,x in inp.items()}
  for k in v:
   if branch!='F-preserve' or k!='input_ids':np.testing.assert_array_equal(v[k],reference[k],err_msg=parent['id']+'/'+branch+'/'+k)
  assert np.isfinite(v['image_input']).all()
  crops=object_crop(z['primary'])[1];candidate=None
  if branch!='L-policy':
   data=np.load(base/branch/'episode/candidate.npz')
   np.testing.assert_array_equal(data['primary_visible'][0],z['primary'])
   np.testing.assert_array_equal(data['secondary_visible'][0],z['secondary'])
   np.testing.assert_array_equal(data['proprio'][0].astype(np.float32),z['proprio'])
   cmds=[json.loads(x) for x in (base/branch/'episode/robot_actions.jsonl').read_text().splitlines()]
   ep=json.loads((base/branch/'episode/episode.json').read_text())
   post=np.asarray([x['post_time_s'] for x in cmds]);pre=np.r_[ep['initial_truth']['sim_time_s'],post[:-1]]
   assert np.allclose(post-pre,.05,atol=1e-10,rtol=0)
   assert np.allclose(pre,[x['pre_time_s'] for x in cmds],atol=1e-10,rtol=0)
   targets=data['targets'];assert len(targets)==len(cmds)
   for k,i in enumerate(data['frame_action_indices']):
    expected=np.r_[targets[i-1,:9],0.,np.zeros(10)] if i else z['proprio']
    np.testing.assert_array_equal(data['proprio'][k],np.asarray(expected,np.float32))
   complete=[int(i) for i in data['frame_action_indices'] if i+30<=len(targets)]
   candidate={'actions':len(targets),'complete_windows':len(complete),'real_timestamps_reconstructed_from_recorded_initial_and_post_time':True,'unrounded_dt_max_error':float(np.max(abs(post-pre-.05))),'all_command_history_exact':True}
  rows.append({'parent':parent['id'],'branch':branch,'actual_native_initial_tensors_exact':True,'task_text_matches':branch!='F-preserve','nontext_exact_for_preservation':branch=='F-preserve','mask':v['image_mask'].tolist(),'crop':crops,'input_hashes':{k:array_sha(x) for k,x in v.items()},'candidate':candidate})
(O/'input-and-label-audit.json').write_text(json.dumps({'passed':True,'new_model_calls':0,'physics_steps':0,'checks':rows,'epoch':time.time()},indent=2))
print('INPUT_AND_LABEL_AUDIT_PASSED',len(rows))
