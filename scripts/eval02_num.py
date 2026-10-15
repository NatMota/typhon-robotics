"""A frozen-image numeric intervention: four new chunks, four archived controls."""
import os,sys,json,time,hashlib,shutil,traceback,csv,difflib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
O=Path(os.environ.get('TYPHON_NUM_ROOT','/var/tmp/typhon-eval02-num-20261015'))
FIT=Path('/var/tmp/typhon-eval02-fit-20261015')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def equal(a,b,exclude=()):
 assert a.files==b.files
 for k in a.files:
  if k not in exclude:np.testing.assert_array_equal(a[k],b[k])
def limit():assert time.time()<read(O/'clock.json')['execution_deadline'],'Proposal block time cap'
def prepare():
 import torch
 from transformers import AutoProcessor
 from supervisor.adaptation.observations import views
 from supervisor.adaptation.m6_contract import reorder,DOMAIN
 from load04_common import array_sha
 p=read(FIT/'protocol.json');meta=p['packets']['postlift'];assert sha(meta['archive'])==meta['sha256']
 shutil.copy2(meta['archive'],O/'postlift.npz');z=np.load(O/'postlift.npz')
 original=meta['instruction'];assert original.count('y=0.030000')==1
 swapped=original.replace('y=0.030000','y=-0.120000')
 processor=AutoProcessor.from_pretrained('.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);processor.language_max_length=128
 inputs={};texts={};tokens={}
 for name,ins in [('original',original),('swapped',swapped)]:
  q=processor(images=views(dict(zip(['primary','secondary'],[z['primary'],z['secondary']])),'O2'),language_instruction=ins)
  q['proprio']=reorder(torch.as_tensor(z['proprio'],dtype=torch.float32).unsqueeze(0));q['domain_id']=torch.tensor([DOMAIN],dtype=torch.long)
  arrays={k:v.detach().cpu().contiguous().numpy() for k,v in q.items()}
  inputs[name]=arrays;np.savez_compressed(O/(name+'-prepared.npz'),**arrays)
  texts[name]=processor.tokenizer.decode(arrays['input_ids'][0],skip_special_tokens=True)
  tokens[name]={'ids':arrays['input_ids'][0].tolist(),'pieces':processor.tokenizer.convert_ids_to_tokens(arrays['input_ids'][0].tolist()),'decoded':texts[name]}
  assert 'x=0.125000' in texts[name] and 'world +y.' in texts[name],texts[name]
 assert texts['original'].replace('y=0.030000','y=-0.120000')==texts['swapped']
 for k in inputs['original']:
  if k!='input_ids':np.testing.assert_array_equal(inputs['original'][k],inputs['swapped'][k])
 assert not np.array_equal(inputs['original']['input_ids'],inputs['swapped']['input_ids'])
 ops=difflib.SequenceMatcher(a=tokens['original']['ids'],b=tokens['swapped']['ids'],autojunk=False).get_opcodes()
 tokenaudit={'passed':True,'original':tokens['original'],'swapped':tokens['swapped'],'opcodes':ops,'only_decoded_change':'y=0.030000 -> y=-0.120000','numeric_coordinates_complete':True,'all_nontext_arrays_exact':True}
 dump(O/'token-audit.json',tokenaudit)
 controls=[]
 for model in ['GOAL','B']:
  for seed in p['policy_seeds']:
   src=FIT/model/('postlift-'+str(seed));d=O/'reused'/model/str(seed);d.mkdir(parents=True)
   with np.load(src/'initial-prepared.npz') as a:
    assert set(a.files)==set(inputs['original'])
    for k in a.files:np.testing.assert_array_equal(a[k],inputs['original'][k])
   assert read(src/'input-000.json')['instruction']==original
   assert read(FIT/model/'reload-verification.json')['passed']
   for name in ['proposal.npz','initial-prepared.npz','noise-000.npz','input-000.json','metrics.json']:
    shutil.copy2(src/name,d/name)
   controls.append({'model':model,'seed':seed,'source':str(src),'copy':str(d),'hashes':{f.name:sha(f) for f in d.iterdir()}})
 sources=['scripts/load04_worker.py','scripts/eval02_goal_run.py','scripts/eval02_num.py','supervisor/adaptation/observations.py','supervisor/adaptation/m6_contract.py']
 assert sha(R/'scripts/load04_worker.py')==p['worker_sha256']
 protocol={'experiment':'EVAL02-NUM','ticket':11065,'frozen_epoch':time.time(),'clock':read(O/'clock.json'),'model_order':['GOAL','B'],'models':{k:p['models'][k] for k in ['GOAL','B']},'pair_selection':'Predefined RET-C mixed-result branch: GOAL-varied vs B','policy_seeds':p['policy_seeds'],'packet':meta,'packet_copy':str(O/'postlift.npz'),'instructions':{'original':original,'swapped':swapped},'expected_initial_nontext':{k:array_sha(v) for k,v in inputs['swapped'].items() if k!='input_ids'},'source_sha256':{f:sha(R/f) for f in sources},'controls':controls,'actual_noise_matching':'Regenerate native noise after exact seed reset; compare all CPU/CUDA RNG and actual draw arrays to saved FIT packets. No literal noise-file override.','inference':{'observation':'O2','domain':DOMAIN,'H':30,'denoise_steps':10,'dtype':'bf16','language_max_length':128},'evidence_chunks':8,'new_generations':4,'reused_generations':4,'physical_rollouts':0,'training':False,'holdouts':False,'interpretation':'Conflicting-cue numerical sensitivity; no swapped expert labels or task-success claims.'}
 dump(O/'protocol.json',protocol);print(json.dumps({'prepared':True,'decoded':texts,'token_changes':[x for x in ops if x[0]!='equal']}),flush=True)
def run():
 import eval02_goal_run as base
 from eval02_fit import matrices
 from scipy.spatial.transform import Rotation
 p=read(O/'protocol.json')
 for f,h in p['source_sha256'].items():assert sha(R/f)==h
 assert sha(p['packet_copy'])==p['packet']['sha256'];z=np.load(p['packet_copy']);expected=np.load(O/'swapped-prepared.npz');rows=[]
 for model in p['model_order']:
  limit();d=O/'generated'/model;d.mkdir(parents=True);base.OUT=d
  w=base.Worker({'model':p['models'][model],'expected_initial_nontext':p['expected_initial_nontext']})
  try:
   for seed in p['policy_seeds']:
    limit();dest=d/str(seed);ref=O/'reused'/model/str(seed);assert not dest.exists()
    w._request({'op':'reset','seed':seed})
    result=w._request({'op':'predict','images':[z['primary'],z['secondary']],'proprio':z['proprio'],'instruction':p['instructions']['swapped'],'audit_index':0,'audit_directory':str(dest)})
    pred=result.pop('actions')
    with np.load(dest/'initial-prepared.npz') as actual:equal(actual,expected)
    with np.load(dest/'noise-000.npz') as actual,np.load(ref/'noise-000.npz') as old:equal(actual,old)
    orig=np.load(ref/'proposal.npz')['predicted'];assert orig.shape==pred.shape==(30,20)
    np.savez_compressed(dest/'paired-proposals.npz',original=orig,swapped=pred)
    delta=(pred[:,:3]-orig[:,:3])*1000
    rot=np.degrees(Rotation.from_matrix(matrices(pred)@matrices(orig).transpose(0,2,1)).magnitude())
    opening=lambda a:(np.flatnonzero((a[:-1,9]>.5)&(a[1:,9]<=.5))+1).tolist()
    row={'model':model,'seed':seed,'new_directory':str(dest),'reused_directory':str(ref),'actual_input_and_noise_checks_passed':True,'inference':result,'original_final_xyz_m':orig[-1,:3].tolist(),'swapped_final_xyz_m':pred[-1,:3].tolist(),'delta_final_xyz_mm':delta[-1].tolist(),'delta_final_y_mm':float(delta[-1,1]),'mean_delta_y_mm':float(delta[:,1].mean()),'max_abs_delta_y_mm':float(np.abs(delta[:,1]).max()),'mean_xyz_displacement_mm':float(np.linalg.norm(delta,axis=1).mean()),'mean_rotation_displacement_deg':float(rot.mean()),'final_rotation_displacement_deg':float(rot[-1]),'original_close_probabilities':orig[:,9].tolist(),'swapped_close_probabilities':pred[:,9].tolist(),'original_opening_transitions':opening(orig),'swapped_opening_transitions':opening(pred),'delta_xyz_per_step_mm':delta.tolist(),'rotation_displacement_per_step_deg':rot.tolist()}
    rows.append(row);dump(dest/'comparison.json',row);dump(O/'comparisons.json',rows);print(json.dumps({k:row[k] for k in ['model','seed','delta_final_y_mm','mean_xyz_displacement_mm','swapped_opening_transitions']}),flush=True)
  finally:w.close()
 keys=['model','seed','delta_final_y_mm','mean_delta_y_mm','max_abs_delta_y_mm','mean_xyz_displacement_mm','mean_rotation_displacement_deg','final_rotation_displacement_deg']
 with (O/'comparisons.csv').open('w',newline='') as f:
  out=csv.DictWriter(f,fieldnames=keys);out.writeheader();out.writerows({k:r[k] for k in keys} for r in rows)
 dump(O/'validation_summary.json',{'passed':True,'evidence_chunks':8,'new_generations':4,'reused_generations':4,'fresh_checkpoint_probes':2,'physical_rollouts':0,'training':False,'actual_prepared_inputs_match_frozen_conditions':True,'actual_noise_and_rng_arrays_exact_to_archived_FIT':True,'native_noise_regenerated_not_injected':True,'no_swapped_imitation_error_scored':True,'all_results_finite':True,'finished_epoch':time.time()})
 dump(O/'complete.json',{'completed':True,'new_generations':4,'reused_generations':4,'epoch':time.time()})
if __name__=='__main__':
 try:{'prepare':prepare,'run':run}[sys.argv[1]]()
 except BaseException:dump(O/('STOPPED-'+sys.argv[1]+'.json'),{'epoch':time.time(),'error':traceback.format_exc()});raise
