"""Matched ancestor retention diagnostic; native scenes and actions, no training."""
import os,sys,json,time,copy,hashlib,random,gzip,pickle,subprocess,traceback,csv
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
O=Path(os.environ.get('TYPHON_RETENTION_ROOT','/var/tmp/typhon-eval02-retention-20261015'))
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def limit():assert time.time()<read(O/'clock.json')['execution_deadline'],'Experiment review deadline'
def prepare_scenes():
 import numpy as np
 from supervisor.feeding_loading.environment import FeedingLoadingEnv
 from supervisor.adaptation.actions import proprio_vector
 from load04_common import array_sha
 import imageio.v2 as imageio
 root=Path('/var/tmp/typhon-ost-sweep-20261012/goal-grounding/data')
 manifest=read(root/'dataset-v2.json');cases={}
 for goal in [0,1]:
  e=next(x for x in manifest['episodes'] if x['source_demo_id']=='demo_0002' and x['goal_index']==goal)
  assert e['split']=='train'
  ep=read(Path(e['source_path'])/'episode.json');config=e['config'];seed=ep['seed']
  assert sha(root/e['path'])==e['sha256']
  env=None
  try:
   random.seed(seed);env=FeedingLoadingEnv(copy.deepcopy(config));obs=env.reset(seed);truth=env.truth()
   assert truth==ep['initial_truth'],('training initial truth mismatch',goal)
   arrays={k:getattr(env.env.sim.data,k).copy() for k in ['qpos','qvel','mocap_pos','mocap_quat']}
   arrays.update(primary=obs['rgb']['primary'],secondary=obs['rgb']['secondary'],proprio=proprio_vector(obs).astype(np.float32))
   z=np.load(root/e['path'])
   for k in ['primary','secondary','proprio']:np.testing.assert_array_equal(arrays[k],z[k][0])
   archive=O/('initial-G'+str(goal)+'.npz');np.savez_compressed(archive,**arrays)
   imageio.imwrite(O/('initial-G'+str(goal)+'.png'),np.concatenate([arrays['primary'],arrays['secondary']],axis=1))
   cases['G'+str(goal)]={'source_demo_id':'demo_0002','source_episode_id':e['id'],'simulation_seed':seed,'initial_truth':truth,'initial_archive':str(archive),'initial_archive_sha256':sha(archive),'config':config,'instruction':e['instruction'],'training_source_sha256':e['sha256'],'training_episode_sha256':sha(Path(e['source_path'])/'episode.json'),'training_native_initial_arrays_exact':True}
  finally:
   if env is not None:env.close()
 assert cases['G0']['config']['initial_state']==cases['G1']['config']['initial_state']
 dump(O/'cases.json',cases)
 print(json.dumps({'prepared_cases':list(cases),'initial_training_truth_RGB_proprio_exact':True}),flush=True)
def prepare_inputs():
 import numpy as np,torch
 from transformers import AutoProcessor
 from supervisor.adaptation.observations import views
 from supervisor.adaptation.m6_contract import reorder,DOMAIN
 from load04_common import array_sha
 cases=read(O/'cases.json')
 p=AutoProcessor.from_pretrained('.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);p.language_max_length=128
 for goal,c in cases.items():
  z=np.load(c['initial_archive']);x=p(images=views({k:z[k] for k in ['primary','secondary']},'O2'),language_instruction=c['instruction'])
  x['proprio']=reorder(torch.as_tensor(z['proprio'],dtype=torch.float32).unsqueeze(0));x['domain_id']=torch.tensor([DOMAIN],dtype=torch.long)
  arrays={k:v.detach().cpu().numpy() for k,v in x.items()}
  c['input_hashes']={k:array_sha(v) for k,v in arrays.items()}
  c['expected_initial_nontext']={k:v for k,v in c['input_hashes'].items() if k not in ['input_ids','attention_mask']}
  decoded=p.tokenizer.decode(arrays['input_ids'].reshape(-1).tolist(),skip_special_tokens=True)
  assert ''.join(c['instruction'].split()) in ''.join(decoded.split())
  c['decoded_instruction']=decoded;np.savez_compressed(O/('prepared-'+goal+'.npz'),**arrays)
 dump(O/'cases.json',cases);print('Input qualification passed',flush=True)
def run():
 import numpy as np
 from scipy.spatial.transform import Rotation
 import eval02_goal_run as base
 import live01_p_run as prior
 from supervisor.baseline.benchmark import run_episode
 from supervisor.baseline.xvla import to_action
 from m5_evaluate import supplemental
 p=read(O/'protocol.json')
 for f,h in p['source_sha256'].items():assert sha(R/f)==h,('source changed',f)
 class Policy:
  def __init__(self,w,case,dest,policy_seed,goal):
   self.w=w;self.case=case;self.dest=dest;self.policy_seed=policy_seed;self.goal=goal
   self.records=[];self.histories=[];self.chunks=[];self.target=np.zeros(20);self.index=0;self.proprio=None
  def reset(self,seed):
   assert seed==self.case['simulation_seed'];self.index=0;self.proprio=None
   self.w._request({'op':'reset','seed':self.policy_seed})
  def instrument(self,env,out):
   self.env=env;self.telemetry=gzip.open(out/'controller_trace.jsonl.gz','wt')
   assert env.truth()==self.case['initial_truth']
   z=np.load(self.case['initial_archive'])
   for k in ['qpos','qvel','mocap_pos','mocap_quat']:np.testing.assert_array_equal(getattr(env.env.sim.data,k),z[k])
   blob=pickle.dumps(prior.state(env),protocol=4);(out/'initial-state.pkl').write_bytes(blob)
   ref=O/('state-'+self.goal+'.pkl')
   if ref.exists():assert ref.read_bytes()==blob,'Initial available state mismatch across checkpoint/seed'
   else:ref.write_bytes(blob)
   dump(out/'initial-qualification.json',{'passed':True,'same_goal_available_state_exact':True,'state_sha256':hashlib.sha256(blob).hexdigest()})
   raw=env.step
   def step(action):
    limit();raw(action);d=base.native.b.rich(env,self.target,action);self.records.append(d)
    self.telemetry.write(json.dumps(d,allow_nan=False)+'\n')
   env.step=step
   return {'native_feeding_scene':True,'renderer_intervention':False,'training_initial_verified':True}
  def to_action(self,target,obs,config):
   self.target=np.asarray(target).copy();return to_action(target,obs,config)
  def predict(self,obs):
   limit()
   if self.proprio is None:
    prop=obs['proprio'];rot=Rotation.from_quat(prop['eef_quat_xyzw']).as_matrix()
    self.proprio=np.r_[prop['eef_position_m'],rot[:,0],rot[:,1],0.,np.zeros(10)].astype(np.float32)
   history=self.proprio.copy();self.histories.append(history)
   if self.index==0:
    z=np.load(self.case['initial_archive'])
    for k in ['primary','secondary']:np.testing.assert_array_equal(obs['rgb'][k],z[k])
    np.testing.assert_array_equal(history,z['proprio'].astype(np.float32))
   result=base.native.request(self.w,obs,history,self.case,self.index,self.dest/'input-audit')
   chunk=result.pop('actions');self.chunks.append(chunk.copy());self.proprio[:9]=chunk[-1,:9];self.index+=1
   np.savez_compressed(self.dest/'predictions.npz',chunks=np.asarray(self.chunks),histories=np.asarray(self.histories))
   dump(self.dest/'progress.json',{'queries':self.index,'control_steps':self.env.steps,'epoch':time.time()})
   return chunk
 rows=[];noise_refs={}
 dump(O/'execution-freeze.json',{'epoch':time.time(),'protocol_sha256':sha(O/'protocol.json'),'git_parent':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
 for goal in p['goal_order']:
  for model in p['model_order']:
   modelout=O/'workers'/(goal+'-'+model);modelout.mkdir(parents=True,exist_ok=False);base.OUT=modelout
   m=p['models'][model];wp={'model':m,'expected_initial_nontext':{}}
   limit();w=base.Worker(wp)
   try:
    for seed in p['policy_seeds']:
     limit();case=copy.deepcopy(p['cases'][goal]);wp['expected_initial_nontext']=case['expected_initial_nontext']
     tag=model+'-'+goal+'-'+str(seed);dest=modelout/(goal+'-'+str(seed))
     assert not dest.exists(),'No outcome retries'
     random.seed(case['simulation_seed']);policy=Policy(w,case,dest,seed,goal)
     scenario={'scenario_id':'eval02-retention-'+tag,'group':'original-training-combination','task':'feeding','seed':case['simulation_seed'],'config':copy.deepcopy(case['config'])}
     try:
      ep=run_episode(scenario,'pretrained',dest,policy,{'experiment':'EVAL02-RETENTION','model':model,'model_sha256':m['sha256'],'source_episode_id':case['source_episode_id'],'policy_seed':seed,'harness':False,'TTC':False,'new_training':False,'holdout':False,'protocol_sha256':sha(O/'protocol.json')},episode_setup=policy.instrument)
     finally:
      if hasattr(policy,'telemetry'):policy.telemetry.close()
     assert not any(ep[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors'])
     assert len(policy.chunks)==48 and len(policy.records)==1460
     noise=base.noise_gate(dest,seed,noise_refs.get(seed));noise_refs.setdefault(seed,dest)
     s=ep['evaluation'];row={'id':tag,'model':model,'goal':goal,'seed':seed,'directory':str(dest),'source_episode_id':case['source_episode_id'],'goal_xy_m':case['config']['task']['target_position_xy_m'],'final_xy_m':ep['final_truth']['object_position_m'][:2],'strict_success':s['success'],'position_error_mm':s['position_error_m']*1000,'yaw_error_deg':s['orientation_error_deg'],'criteria':s['criteria'],'physical_footprint':s['footprint_inside_target'],'wall_seconds':ep['wall_duration_s'],'noise_qualification':noise,'query_count':len(policy.chunks),'executed_records':len(policy.records),**supplemental(ep,dest)}
     dump(dest/'retention-summary.json',row);rows.append(row);dump(O/'summary.json',rows)
     print(json.dumps({k:row[k] for k in ['id','strict_success','position_error_mm','yaw_error_deg','final_xy_m','wall_seconds']}),flush=True)
   finally:w.close()
 dump(O/'complete.json',{'completed':True,'episodes':len(rows),'epoch':time.time(),'training':False,'holdout':False,'outcome_retries':0})
if __name__=='__main__':
 try:{'scenes':prepare_scenes,'inputs':prepare_inputs,'run':run}[sys.argv[1]]()
 except BaseException:
  dump(O/('STOPPED-'+sys.argv[1]+'.json'),{'error':traceback.format_exc(),'epoch':time.time()});raise
