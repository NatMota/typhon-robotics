"""LOAD02: instruction-only intervention, fixed physical scene and original scoring."""
import os,sys,json,time,copy,hashlib,gzip,subprocess
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
OUT=Path('/var/tmp/typhon-load02-20261013');OLD=Path('/var/tmp/typhon-load01-20261013')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def array_sha(a):
 import numpy as np
 a=np.asarray(a);return hashlib.sha256(str(a.dtype).encode()+json.dumps(list(a.shape)).encode()+a.tobytes()).hexdigest()
def endpoint(config,task):
 c=copy.deepcopy(config);w=c['workspace']
 if task=='feeding':
  c['task']={'type':'feeding','target_position_xy_m':w['staging_position_xy_m'],'target_yaw_rad':0.,'target_position_jitter_m':0.,'target_region_size_m':w['staging_size_m'],'instruction':read(OUT/'protocol.json')['feeding_template']}
 return c
def scores(row,config):
 from supervisor.feeding_loading.evaluation import score_episode
 return {task:score_episode(row['initial_truth'],row['final_truth'],row['stability_truths'],endpoint(config,task),row['attempts']) for task in ['feeding','loading']}
def prepare():
 import numpy as np
 from scipy.spatial.transform import Rotation
 from supervisor.baseline.xvla import task_prompt
 from supervisor.feeding_loading.environment import FeedingLoadingEnv
 old=read(OLD/'protocol.json');clock=read(OUT/'clock.json')
 assert time.time()<clock['qualification_deadline']
 feeding=next(s['config']['task']['instruction'] for s in old['scenarios'] if s['task']=='feeding')
 scenes=[s for s in old['scenarios'] if s['group']=='loading_development']
 methods={m:{'checkpoint':str(OLD/('train-'+m)/'step3000.pt'),'probe':str(OLD/('train-'+m)/'step3000-probe.pt'),'sha256':read(OLD/('train-'+m)/'complete.json')['checkpoint_sha256']} for m in ['F','FL']}
 for entry in methods.values():assert sha(entry['checkpoint'])==entry['sha256']
 protocol={'schema':'load02-fixed-scene-instruction-switch-v1','created_unix':time.time(),'models':methods,'feeding_template':feeding,'scenes':scenes,'seeds':[912171,912172],'instructions':['feeding','loading'],'total_rollouts':16,'physical_config':'unchanged loading-development config for both instructions','only_intervention':'full serialized model language_instruction; offline endpoint score changes correspondingly','H':30,'frequency_hz':20,'denoise_steps':10,'harness':False,'TTC':False,'training':False,'parent_result_commit':'4e8858a0af1cb0349f51c0c25130c1d72485a6df','statistical_status':'inspected development scenes, paired stochastic samples, not new families','stop':'input isolation failure, runtime error, qualification 10 min / execution 50 min / archive 60 min from clock','scorer':'original independent score_episode against both goals, including height/release/stability'}
 assert not (OUT/'protocol.json').exists();dump(OUT/'protocol.json',protocol)
 feasibility=[];cases=[];fixtures=[]
 for s in scenes:
  d=OLD/'evaluation/P/remainder'/s['scenario_id'];r=read(d/'episode.json');sc=scores(r,s['config'])
  ed=OLD/'expert-feasibility'/s['scenario_id']
  if not (ed/'episode.json').exists():
   found=[p for p in (OLD/'expert-feasibility').rglob('episode.json') if read(p.parent/'scenario.json')['source_demo_id']==s['source_demo_id']]
   assert len(found)==1;ed=found[0].parent
  er=read(ed/'episode.json')
  feasibility.append({'source_demo_id':s['source_demo_id'],'feeding_from_saved_parent':sc['feeding'],'feeding_evidence':str(d),'feeding_episode_sha256':sha(d/'episode.json'),'loading_from_saved_expert':er['evaluation'],'loading_evidence':str(ed),'loading_episode_sha256':sha(ed/'episode.json')})
  for seed in protocol['seeds']:
   fid=s['source_demo_id']+'-seed'+str(seed);fd=OUT/'fixtures'/fid;fd.mkdir(parents=True)
   env=FeedingLoadingEnv(s['config'])
   try:
    obs=env.reset(seed);truth=env.truth();sim=env.env.sim
    q=Rotation.from_quat(obs['proprio']['eef_quat_xyzw']).as_matrix()
    prop=np.r_[obs['proprio']['eef_position_m'],q[:,0],q[:,1],0.,np.zeros(10)].astype(np.float32)
    np.savez_compressed(fd/'input.npz',primary=obs['rgb']['primary'],secondary=obs['rgb']['secondary'],proprio=prop,qpos=sim.data.qpos.copy(),qvel=sim.data.qvel.copy(),mocap_pos=sim.data.mocap_pos.copy(),mocap_quat=sim.data.mocap_quat.copy())
    dump(fd/'truth.json',truth)
   finally:env.close()
   prompts={}
   for task in protocol['instructions']:
    prompts[task]=task_prompt(endpoint(s['config'],task)['task'])
    case=copy.deepcopy(s);case.update(scenario_id='load02-'+fid+'-'+task,seed=seed,group='fixed_scene_instruction',requested_task=task,fixture_id=fid,instruction_override=prompts[task])
    cases.append(case)
   dump(fd/'prompts.json',prompts);fixtures.append(fid)
 dump(OUT/'feasibility.json',{'cases':feasibility,'passed':all(x['feeding_from_saved_parent']['success'] and x['loading_from_saved_expert']['success'] for x in feasibility),'new_expert_executions':0})
 assert read(OUT/'feasibility.json')['passed'],'Existing endpoint feasibility gate failed'
 protocol['cases']=cases;protocol['fixtures']=fixtures;dump(OUT/'protocol.json',protocol)
 print(json.dumps({'cases_per_model':len(cases),'feasibility':True,'fixtures':len(fixtures),'elapsed_minutes':(time.time()-clock['t0_unix'])/60}),flush=True)
if __name__=='__main__':prepare()
