"""INIT01: training-family prefix replay and candidate-only handover supervision."""
import os,sys,json,time,copy,random,pickle,gzip,hashlib,traceback,shutil
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
import numpy as np
from scipy.spatial.transform import Rotation
import imageio.v2 as imageio
import hier01_task as h
import hier01_chain_amend as chain
from load04_data import CaptureEnv
from load04_common import endpoint
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.feeding_loading.runner import scripted_attempt
from supervisor.baseline.benchmark import Recorder
from supervisor.baseline.xvla import task_prompt
from supervisor.adaptation.actions import proprio_vector,absolute_target
O=Path('/var/tmp/typhon-return01-20261016');OLD=Path('/var/tmp/typhon-load04-20261014')
h.O=O;chain.O=O
read,dump,sha=h.read,h.dump,h.sha
def limit(stage='execution_deadline'):
 assert time.time()<read(O/'clock.json')[stage],'INIT01 frozen clock deadline'
def forbidden(*args,**kwargs):raise AssertionError('No reset after archived prefix')
def state(env):return chain.state_blob(env)
def numeric(a,b):
 if isinstance(a,dict):
  assert set(a)==set(b)
  return max([numeric(a[k],b[k]) for k in a] or [0.])
 if isinstance(a,(list,tuple,np.ndarray)):
  aa=np.asarray(a);bb=np.asarray(b)
  if aa.dtype.kind in 'biuf' and bb.dtype.kind in 'biuf':
   assert aa.shape==bb.shape;return float(np.max(np.abs(aa.astype(float)-bb.astype(float)))) if aa.size else 0.
  assert a==b;return 0.
 if isinstance(a,(bool,int,float,np.generic)):return float(abs(a-b))
 assert a==b;return 0.

def replay(parent,out,canonical=False):
 out.mkdir(parents=True,exist_ok=False)
 e=parent['entry'];orig=read(Path(e['source_path'])/'episode.json');physical=parent['physical_config']
 random.seed(orig['seed']);CaptureEnv.physical_config=copy.deepcopy(physical)
 env=CaptureEnv(copy.deepcopy(e['config']));env.reset(orig['seed']);env.enabled=False
 errors=[];joint_errors=[];post=[]
 try:
  assert numeric(env.truth(),orig['initial_truth'])<1e-8
  for line in Path(parent['actions_path']).read_text().splitlines():
   row=json.loads(line);env.step(row['normalized_command']);t=env.truth();post.append(t)
   errors.append(numeric(t,row['post_action_truth']))
   rob=env.env.robots[0];joint_errors.append(float(np.max(np.abs(env.env.sim.data.qpos[rob._ref_joint_pos_indexes]-row['joint_position_rad']))))
  assert max(errors)<1e-8 and max(joint_errors)<1e-8
  assert numeric(env.truth(),orig['final_truth'])<1e-8
  feed=score_episode(orig['initial_truth'],env.truth(),post[-24:],endpoint(physical,'feeding'),orig['attempts'])
  loading=score_episode(orig['initial_truth'],env.truth(),post[-24:],physical,orig['attempts'])
  assert loading['success'] and not feed['success'] and env.truth()['gripper_open']
  blob=state(env);ref=O/'qualification'/parent['id']/'terminal-state.pkl'
  if not canonical:assert blob==ref.read_bytes(),'Matched branch semantic state differs'
  (out/'terminal-state.pkl').write_bytes(blob)
  obs=env.observe();d=env.env.sim.data
  np.savez_compressed(out/'initial-live.npz',primary=obs['rgb']['primary'],secondary=obs['rgb']['secondary'],proprio=proprio_vector(obs).astype(np.float32),qpos=d.qpos.copy(),qvel=d.qvel.copy(),mocap_pos=d.mocap_pos.copy(),mocap_quat=d.mocap_quat.copy())
  imageio.imwrite(out/'initial.png',np.concatenate([obs['rgb']['primary'],obs['rgb']['secondary']],1))
  assert state(env)==blob
  rec={'passed':True,'parent':parent['id'],'source':e['source_path'],'seed':orig['seed'],'replayed_commands':len(errors),'max_truth_abs_error':max(errors),'max_joint_abs_error':max(joint_errors),'terminal_truth':env.truth(),'feeding_score':feed,'loading_score':loading,'matched_semantic_state_exact':not canonical,'semantic_state_sha256':hashlib.sha256(blob).hexdigest(),'no_reset_home_injection_after_prefix':True,'time':time.time()}
  dump(out/'prefix-qualification.json',rec)
  env.reset=forbidden;env.env.reset=forbidden
  return env,rec,post[-24:]
 except BaseException:
  env.close();dump(out/'FAILED.json',{'error':traceback.format_exc(),'epoch':time.time()});raise
def record_teacher(env,parent,branch,out,previous,initial_history):
 out.mkdir(parents=True,exist_ok=False);initial=env.truth();start=env.steps;cfg=copy.deepcopy(env.config)
 # Reuse the exact LOAD04 CaptureEnv recorder/absolute_target encoder from this boundary.
 env.counter=0;env.last_target=initial_history.copy();env.post=[];env.crop_audit=[];env.packet={k:[] for k in ['primary_visible','secondary_visible','primary_hidden','secondary_hidden','frame_action_indices','targets','proprio']};env.enabled=True
 execution_end=start+(60 if branch=='L-preserve' else 1460)
 rec=Recorder(env,out/'trajectory.jsonl.gz');raw=env.step;rich=[];commands=[];attempts=[];exception=None;parking=None;all_hold_observations=[]
 d=env.env.sim.data;rob=env.env.robots[0];joint_ids=[env.env.sim.model.joint_name2id(n) for n in rob.robot_model.joints]
 ranges=env.env.sim.model.jnt_range[joint_ids].copy()
 def step(command):
  limit();assert env.steps < execution_end, 'RETURN01 action budget';before=env.observe()
  if branch=='L-preserve':all_hold_observations.append({'primary':before['rgb']['primary'].copy(),'secondary':before['rgb']['secondary'].copy(),'measured_proprio':proprio_vector(before)})
  raw(command)
  commands.append({'action_index':len(commands),'pre_time_s':float(env.env.sim.data.time)-.05,'post_time_s':float(env.env.sim.data.time),'normalized_command':np.asarray(command).tolist(),'encoded_target':absolute_target(command,before,cfg).tolist(),'post_action_truth':env.truth(),'joint_position_rad':d.qpos[rob._ref_joint_pos_indexes].copy().tolist()})
  rich.append(h.base.native.b.rich(env,np.asarray(commands[-1]['encoded_target']),command))
 env.step=step
 writer=imageio.get_writer(out/'rollout.mp4',fps=5)
 def capture(scene):writer.append_data(np.concatenate([scene.frame('primary'),scene.frame('secondary')],1))
 capture(env);env.capture_callback=capture;imageio.imwrite(out/'initial.png',np.concatenate([env.frame(),env.frame('secondary')],1))
 try:
  if branch=='L-preserve':
   ob=env.observe();xyz=np.asarray(ob['proprio']['eef_position_m']);rot=Rotation.from_quat(ob['proprio']['eef_quat_xyzw']).as_matrix()
   for _ in range(60):
    rob=env.env.robots[0];now=env.env.sim.data.site_xmat[rob.eef_site_id].reshape(3,3)
    dr=Rotation.from_matrix(rot@now.T).as_rotvec()/cfg['control']['rotation_step_rad']
    dx=(xyz-env.env.sim.data.site_xpos[rob.eef_site_id])/cfg['control']['translation_step_m']
    env.step(np.r_[np.clip(dx,-1,1),np.clip(dr,-1,1),-1.])
  else:
   for index in range(2):
    a,retreat,yaw=scripted_attempt(env,cfg,index);attempts.append(a)
    if a['verified_lift']:break
   before=env.steps;parking={'converged':bool(env.move_to(np.array([-.25,-.25,1.08]),0.,-1.)),'steps':env.steps-before}
   for _ in range(12):env.tick(np.array([-.25,-.25,1.08]),0.,-1.,steps=2)
 except Exception:exception=traceback.format_exc()
 finally:
  rec.finish();env.capture_callback=None;capture(env);writer.close();imageio.imwrite(out/'final.png',np.concatenate([env.frame(),env.frame('secondary')],1))
 final=env.truth();stability=env.post[-24:];score=score_episode(initial,final,stability,cfg,attempts)
 if exception:score['success']=False
 arrays={k:np.asarray(v,np.float32 if k in ['targets','proprio'] else None) for k,v in env.packet.items()}
 np.savez_compressed(out/'candidate.npz',**arrays)
 if all_hold_observations:np.savez_compressed(out/'all60-hold-preaction-observations.npz',**{k:np.asarray([x[k]for x in all_hold_observations])for k in all_hold_observations[0]})
 dump(out/'crop-audit.json',env.crop_audit);dump(out/'controller-rich.json',rich)
 (out/'robot_actions.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in commands))
 ep={'parent':parent['id'],'branch':branch,'initial_truth':initial,'final_truth':final,'stability_truths':stability,'attempts':attempts,'evaluation':score,'success':score['success'],'controller_exception':exception,'parking':parking,'control_steps':env.steps-start,'candidate_only':True,'uses_simulator_truth_for_control':branch=='F-teacher','instruction':task_prompt(cfg['task']),'config':cfg}
 dump(out/'episode.json',ep)
 alignment=(len(commands)==len(arrays['targets']) and np.all(np.isfinite(arrays['targets'])) and np.array_equal(arrays['frame_action_indices'],np.arange(0,len(commands),4)) and np.allclose(arrays['targets'],np.asarray([x['encoded_target'] for x in commands],np.float32),atol=0,rtol=0))
 assert alignment
 full=[int(i) for i in arrays['frame_action_indices'] if i+30<=len(commands)]
 q=np.asarray([x['joint_position_rad'] for x in commands]);excursions=np.maximum(ranges[:,0]-q,q-ranges[:,1]);maxex=float(np.maximum(excursions,0).max())
 upright=[]
 for t in env.post:
  r=Rotation.from_quat(np.asarray(t['object_quat_wxyz'])[[1,2,3,0]]).as_matrix();tilt=float(np.arccos(np.clip(r[2,2],-1,1)))
  upright.append(bool(t['bilateral_grasp_contact'] and t['object_position_m'][2]-initial['object_position_m'][2]>=.07 and tilt<=cfg['evaluation']['upright_tolerance_rad']))
 qualifies=bool(branch=='F-teacher' and score['success'] and any(upright) and alignment and not exception)
 report={'parent':parent['id'],'branch':branch,'strict_success':score['success'],'upright_bilateral_lift_70mm':any(upright),'upright_bilateral_lift_samples':sum(upright),'action_alignment_qualified':bool(alignment),'complete_30_action_windows':full,'candidate_positive':qualifies,'supplied_preservation':branch=='L-preserve','object_displacement_m':float(np.linalg.norm(np.asarray(final['object_position_m'])-initial['object_position_m'])),'max_joint_limit_excursion_rad':maxex,'joint_limit_excursion_steps':int(np.any(excursions>0,axis=1).sum()),'phase_nonconvergence':sum(not z['converged'] for a in attempts for z in a['phases']),'contact_steps':sum(t['grasp_contact'] for t in env.post),'actions':len(commands),'directory':str(out),'controller_exception':exception,'generated_robot_labels_not_human_actions':True}
 dump(out/'qualification.json',report)
 return report

import subprocess
import init02_eval as ev
from load04_common import array_sha
PHASE='preparation'
def limit(*args):
 key='preparation_deadline' if PHASE=='preparation' else 'execution_deadline'
 assert time.time()<read(O/'clock.json')[key],('RETURN01 frozen deadline',PHASE)
h.limit=limit;h.O=O;h.base.native.OUT=O;chain.O=O
def worker(dest,expected=None):
 dest.mkdir(parents=True);h.base.OUT=dest
 return h.base.Worker({'model':read(O/'protocol.json')['model'],'expected_initial_nontext':expected or {}})
def prepare(env,parent,d,task,expected=None):
 d.mkdir(parents=True,exist_ok=True)
 before=state(env);cfg=endpoint(parent['physical_config'],task)
 source={'id':parent['id']+'-'+task,'config':parent['physical_config'],'source_demo_id':parent['id'],'simulation_seed':parent['seed'],'policy_seed':912171,'requested_task':task,'physical_scene_task':'loading','instruction':task_prompt(cfg['task']),'expected_initial_nontext':expected or {},'panel':'RETURN01-training'}
 env.config=copy.deepcopy(cfg);env.config['control']['max_steps']=env.steps+(60 if task=='loading' else 1460)
 assert state(env)==before,'Task switch changed physical/controller state'
 case=chain.request_case(env,source,source['id'],d,False)
 dump(d/'input-request.json',{'instruction':source['instruction'],'task':task,'expected':expected or {}})
 # Empty expected is only for first canonical preparation; subsequent comparisons are mandatory.
 pr=read(d/'input-request.json')
 if not expected:pr.pop('expected')
 dump(d/'input-request.json',pr)
 x=subprocess.run([str(R/'.venv-xvla/bin/python'),str(R/'scripts/return01_preprocess.py'),str(d)],capture_output=True,text=True,timeout=75)
 (d/'preprocess.log').write_text(x.stdout+x.stderr);assert x.returncode==0,x.stderr
 hashes=read(d/'input-qualification.json')['hashes'];source['expected_initial_nontext']={k:v for k,v in hashes.items()if k!='input_ids'}
 assert state(env)==before
 return source,hashes
def fresh(env,source,context,out,first):
 case=chain.request_case(env,source,context,out,False);case['expected_initial_nontext']=source['expected_initial_nontext'];return case
def main():
 global PHASE
 ev.request_case=fresh;ev.model_entry=lambda n:read(O/'protocol.json')['model']
 p=read(O/'protocol.json');qualified=[]
 for parent in p['parents']:
  limit();env=None
  try:
   env,receipt,previous=replay(parent,O/'qualification'/parent['id'],True)
   source,f=prepare(env,parent,O/'qualification'/parent['id']/'feeding','feeding')
   _,l=prepare(env,parent,O/'qualification'/parent['id']/'loading','loading',f)
   assert f['input_ids']!=l['input_ids']
   qualified.append({'parent':parent['id'],'feeding_nontext':source['expected_initial_nontext'],'prefix':receipt})
  finally:
   if env is not None:env.close()
 # One independently recreated prefix verifies branch equality before admitting continuation.
 for parent in p['parents']:
  env=None
  try:
   limit();env,_,_=replay(parent,O/'qualification'/parent['id']/'independent',False)
  finally:
   if env is not None:env.close()
 limit();w=worker(O/'qualification/worker-GOAL')
 try:probe=w.load_info
 finally:w.close()
 limit();dump(O/'qualification.json',{'passed':True,'parents':qualified,'probe':probe,'epoch':time.time(),'clock_not_reset':True})
 PHASE='execution';rows=[];failures=[]
 for parent in p['parents']:
  q=next(x for x in qualified if x['parent']==parent['id'])
  for branch in ['F-policy','F-teacher','L-preserve']:
   limit();assert shutil.disk_usage('/var/tmp').free>400_000_000
   d=O/'episodes'/parent['id']/branch;env=None;w=None
   try:
    env,pr,previous=replay(parent,d/'prefix',False)
    task='loading'if branch=='L-preserve'else'feeding'
    source,hashes=prepare(env,parent,d/'request',task,q['feeding_nontext'])
    if branch=='F-policy':
     w=worker(d/'worker',source['expected_initial_nontext'])
     row,_,_=ev.segment(env,w,source,parent['id']+'-'+branch,d/'rollout',False,previous,'GOAL')
     row.update(parent=parent['id'],branch=branch,strict_success=row['requested_success'])
     w.close();w=None
    else:
     prop=np.load(d/'request/initial-live.npz')['proprio']
     row=record_teacher(env,parent,branch,d/'rollout',previous,prop)
    row.update(experiment='RETURN01',epoch=time.time(),no_outcome_retry=True)
    dump(d/'summary.json',row);rows.append(row);dump(O/'episodes.json',rows)
    print(json.dumps({'parent':parent['id'],'branch':branch,'success':row['strict_success']}),flush=True)
   except Exception:
    x={'parent':parent['id'],'branch':branch,'error':traceback.format_exc(),'epoch':time.time(),'technical_failure_not_scored':True}
    failures.append(x);dump(O/'technical-failures.json',failures);print(json.dumps(x),flush=True)
   finally:
    if w is not None:w.close()
    if env is not None:env.close()
 limit();dump(O/'physical-complete.json',{'episodes':len(rows),'failures':len(failures),'epoch':time.time(),'training':False,'candidate_only':True})
if __name__=='__main__':
 try:main()
 except BaseException:
  dump(O/'STOPPED.json',{'phase':PHASE,'error':traceback.format_exc(),'epoch':time.time(),'clock_not_reset':True});raise
