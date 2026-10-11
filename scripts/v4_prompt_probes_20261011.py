"""Frozen paired-seed input probes. Execution waits for comparison and V4 search."""
import argparse, copy, gzip, hashlib, json, math, pickle, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SEEDS=[61601,61602,61603]
PRIMITIVES={
 'hold':'Hold the current end-effector position and orientation. Keep the gripper in its current state.',
 'open':'Hold the current end-effector position and orientation. Open the gripper.',
 'close':'Hold the current end-effector position and orientation. Close the gripper.',
 **{f'{axis}_{sign}':f'Move the end effector {distance:.2f} metres along world {symbol}{axis}. Hold its current orientation and gripper state.'
    for axis,distance in [('x',.03),('y',.03),('z',.02)] for sign,symbol in [('plus','+'),('minus','-')]},
 'yaw_plus':'Rotate the end effector 15 degrees about world +z. Hold its current position and gripper state.',
 'yaw_minus':'Rotate the end effector 15 degrees about world -z. Hold its current position and gripper state.'}

def read(p):return json.loads(p.read_text())
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load_payload(p):
 with gzip.open(p,'rb') as f:return pickle.load(f)
def matrix(a):
 x=a[3:6]/np.linalg.norm(a[3:6]);y=a[6:9]-np.dot(x,a[6:9])*x;y/=np.linalg.norm(y)
 return np.column_stack((x,y,np.cross(x,y)))
def cases():
 jobs=[]
 for model in ['v4-m6','vanilla-pt-domain3','v3-m6']:
  for slot in ['A_initial','B_pregrasp']:
   if model!='v3-m6':
    for condition in ['full_intent',*PRIMITIVES]:
     for seed in SEEDS:jobs.append(dict(family='simple_prompt',model=model,slot=slot,condition=condition,seed=seed))
    if model=='vanilla-pt-domain3':
     for seed in SEEDS:jobs.append(dict(family='native_prompt_reference',model=model,slot=slot,condition='native_task_prompt',seed=seed))
   for scope in ['intent','single']:
    for duration in [1.5,4.0]:
     for seed in SEEDS:jobs.append(dict(family='scope_duration_text',model=model,slot=slot,condition=f'{scope}_text_{duration:.1f}s',scope=scope,duration_text=duration,seed=seed))
  # Same input + seed repeated once per model to measure numerical repeatability.
  jobs.append(dict(family='repeatability',model=model,slot='A_initial',condition='intent_text_1.5s',scope='intent',duration_text=1.5,seed=SEEDS[0]))
 for i,j in enumerate(jobs):j['id']=f"probe-{i:04d}-{j['model']}-{j['slot']}-{j['condition']}-s{j['seed']}"
 assert len(jobs)==225 and len({j['id'] for j in jobs})==225
 return jobs

def variant(base,job):
 p=copy.deepcopy(base)
 if job['family'] in ['scope_duration_text','repeatability']:
  p['request']['scope']=job['scope'];p['request']['duration_s']=job['duration_text']
  if job['scope']=='single':p['request']['instruction']='Approach the block and align the gripper.'
  # Keep the supplied geometric goal context while removing a competing task imperative.
  p['instruction']='Target centre'+base['instruction'].split(' Target centre',1)[1]
 elif job['condition'] in PRIMITIVES:
  p['request']['scope']='single';p['request']['instruction']=PRIMITIVES[job['condition']]
  p['instruction']='Execute only the requested arm primitive in robot world coordinates. No other manipulation is requested in this action chunk.'
 assert p['request']['aligned_duration_s']==1.5 and p['request']['anchor_times_s']==base['request']['anchor_times_s']
 assert np.array_equal(p['proprio'],base['proprio'])
 assert all(np.array_equal(x,y) for x,y in zip(p['images'],base['images']))
 return p

def prepare(bank,out):
 b=read(bank/'manifest.json');assert not b['inference_run']
 manifest={'schema':'v4-paired-prompt-probes-v1','bank':str(bank),'bank_manifest_sha256':sha(bank/'manifest.json'),
  'jobs':cases(),'seeds':SEEDS,'controls':{'same_current_and_history_images':True,'same_proprio':True,'same_seed_within_model':True,
   'duration_factor':'wording only; aligned_duration_s, anchor timestamps, physical-time features and any later execution timing remain fixed at1.5s',
   'prompt_conflict':'single primitive replaces full-intent request and competing task imperative',
   'native_prompt_reference':'vanilla receives original task-only prompt in six extra reference requests',
   'native_model_has_history':False,'vanilla_domain':3,'m7_domain':29,'candidate_selection':False,'training':False},
  'scope':'Diagnostic chunk sensitivity and simple primitive control, not task-success evaluation or evidence of general instruction following.',
  'planned_physical_branches':{'slot':'A_initial','seed':SEEDS[0],'models':['v4-m6','vanilla-pt-domain3'],
   'conditions':['full_intent','hold','open','close','x_plus','x_minus','yaw_plus','yaw_minus'],'count':16,
   'required':'Qualify reset state and unchanged frozen full-intent replay before interpreting any branch. No expert-generated action may replace a model output.'},
  'planned':225,'script_sha256':sha(Path(__file__))}
 out.mkdir(parents=True,exist_ok=False);save(out/'protocol.json',manifest);(out/'probe-source.py').write_bytes(Path(__file__).read_bytes())
 for j in manifest['jobs']:variant(load_payload(bank/j['slot']/'payload.pkl.gz'),j)
 save(out/'preparation-check.json',{'passed':True,'jobs_validated':225,'model_or_simulation_executed':False})
 print(json.dumps({'prepared':225,'out':str(out)}))

def describe(actions,metadata,payload):
 actual=np.array(metadata['measured_eef_position_m']);actual_r=Rotation.from_quat(metadata['measured_eef_quat_xyzw']).as_matrix()
 rows=[]
 for i,a in enumerate(actions):
  r=matrix(a);rotvec=Rotation.from_matrix(r@actual_r.T).as_rotvec()
  rows.append({'waypoint':i,'delta_from_measured_xyz_mm':((a[:3]-actual)*1000).tolist(),
   'delta_from_input_proprio_xyz_mm':((a[:3]-payload['proprio'][:3])*1000).tolist(),
   'rotation_from_measured_world_rotvec_deg':np.degrees(rotvec).tolist(),
   'projected_wrist_x_yaw_deg':math.degrees(math.atan2(r[1,0],r[0,0])),
   'gripper_close_probability':float(a[9]),'would_close':bool(a[9]>.5)})
 return rows

def execute(out,root):
 assert (root/'comparison-complete.json').exists(),'Comparison must finish first'
 assert (root/'ordered-work-complete.json').exists(),'Fresh V4 search must finish first'
 protocol=read(out/'protocol.json');assert protocol['script_sha256']==sha(Path(__file__)),'Frozen probe script changed'
 bank=Path(protocol['bank']);assert protocol['bank_manifest_sha256']==sha(bank/'manifest.json')
 from m7_evaluate import M7Policy
 from m6_evaluate import M6Policy
 from supervisor.adaptation.m6_contract import reorder
 # request_text imports torch in the model worker module, so mirror only the exact formatter and assert it in CPU audit later.
 def text(task,req):return f'Requested {req["scope"]}: {req["instruction"]} Requested duration {req["duration_s"]:.2f} seconds. {task}'
 for model in ['v4-m6','vanilla-pt-domain3','v3-m6']:
  modeldir=out/model;modeldir.mkdir(exist_ok=True);policy=None
  try:
   if model=='vanilla-pt-domain3':policy=M6Policy('vanilla-domain3',None,modeldir/'worker.log')
   else:policy=M7Policy(Path('/var/tmp/humanoid-m7-checkpoints-20261009')/(model+'.pt'),modeldir/'worker.log',annotation_conditioned_v2=True)
   save(modeldir/'load-info.json',policy.load_info)
   for j in [j for j in protocol['jobs'] if j['model']==model]:
    dest=out/'responses'/j['id']
    if (dest/'result.json').exists():continue
    dest.mkdir(parents=True,exist_ok=False)
    base=load_payload(bank/j['slot']/'payload.pkl.gz');payload=variant(base,j);meta=read(bank/j['slot']/'metadata.json')
    exact_text=payload['instruction'] if j['family']=='native_prompt_reference' else text(payload['instruction'],payload['request'])
    wire=payload if model!='vanilla-pt-domain3' else {k:payload[k] for k in ['op','images','instruction','proprio']}
    if model=='vanilla-pt-domain3':wire['instruction']=exact_text
    policy._request({'op':'reset','seed':j['seed']});t=time.perf_counter();response=policy._request(wire)
    wall=time.perf_counter()-t
    with gzip.open(dest/'response.pkl.gz','wb',compresslevel=1) as f:pickle.dump(response,f,protocol=4)
    if model=='vanilla-pt-domain3':actions=np.asarray(response['actions'])
    else:actions=reorder(np.asarray([p['target_generic_interleaved'] for p in response['timed']['waypoints']]),True)
    assert actions.shape==(30,20) and np.isfinite(actions).all()
    np.savez_compressed(dest/'actions.npz',actions=actions)
    save(dest/'result.json',{'job':j,'wall_seconds':wall,'exact_language':exact_text,'request':payload['request'],
     'input_payload_sha256':sha(bank/j['slot']/'payload.pkl.gz'),'source_seed_not_replayed':True,
     'response_sha256':sha(dest/'response.pkl.gz'),'actual_input_shapes':response.get('input_shapes'),
     'image_mask':response.get('image_mask'),'action_metrics':describe(actions,meta,payload),
     'semantics':'M7 interleaved output converted once to concatenated-column controller format; native domain3 already concatenated.',
     'proprio_input_semantics':meta['policy_proprio_semantics'],'simulator_execution':False})
    print(json.dumps({'done':j['id'],'wall_seconds':wall}),flush=True)
  finally:
   if policy:policy.close()
 save(out/'inference-complete.json',{'responses':len(list((out/'responses').glob('*/result.json'))),'planned':225})
 assert read(out/'inference-complete.json')['responses']==225
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','execute']);p.add_argument('--bank',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--root',type=Path);a=p.parse_args()
 if a.command=='prepare':prepare(a.bank,a.output)
 else:execute(a.output,a.root)
