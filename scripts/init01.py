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
O=Path('/var/tmp/typhon-init01-20261015');OLD=Path('/var/tmp/typhon-load04-20261014')
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
  feed=score_episode(orig['initial_truth'],env.truth(),post[-24:],e['config'],orig['attempts'])
  loading=score_episode(orig['initial_truth'],env.truth(),post[-24:],physical,orig['attempts'])
  assert feed['success'] and not loading['success'] and env.truth()['gripper_open']
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
def prepare():
 c=read(O/'clock.json');limit('preparation_deadline')
 data=read(OLD/'data/qualified-dataset.json')['episodes'];parents=[]
 for short in ['0001','0002','0003','0005']:
  identity='demo_'+short;e=next(x for x in data if x['source_demo_id']==identity and x['task']=='feeding');l=next(x for x in data if x['source_demo_id']==identity and x['task']=='loading')
  assert e['split']=='load04_train' and e['success'] and sha(e['absolute_path'])==e['sha256']
  a=Path(e['source_path'])/'robot_actions.jsonl';ep=Path(e['source_path'])/'episode.json'
  parents.append({'id':identity,'entry':e,'physical_config':l['config'],'actions_path':str(a),'actions_sha256':sha(a),'episode_sha256':sha(ep),'seed':read(ep)['seed']})
 model={'checkpoint':'/var/tmp/typhon-load01-20261013/train-FL/step3000.pt','probe':'/var/tmp/typhon-load01-20261013/train-FL/step3000-probe.pt','sha256':'76334b9a17ac5fc49f46c2cb850a3c39ef38e0f13b9cdadc324f760567c5f128','probe_sha256':'621318035b71ad85f1f7aff4d6636a5023aea889a78454a18212c29eac0727f9'}
 assert sha(model['checkpoint'])==model['sha256'] and sha(model['probe'])==model['probe_sha256']
 free=shutil.disk_usage(O).free
 assert free>1500*1024**2,('storage gate',free)
 p={'experiment':'INIT01','ticket':11072,'clock':c,'parents':parents,'model':model,'branches':['L-policy','L-teacher','F-preserve'],'policy_seed':912171,'native_actions':1436,'settle_ticks':24,'preserve_ticks':60,'teacher':'unchanged scripted_attempt up to2 attempts then original open-gripper park/settle; no preflight home at handover','prefix_tolerance':1e-8,'branch_comparator':'existing semantic state_blob, including warmstart, controller fields, contacts, RNG; excludes pointers and visualization statistic.meansize','positive_gate':'upright bilateral supported lift>=0.07m, strict loading success, valid recorder, no hidden intervention','candidate_only':True,'training':False,'quarantined_ids_used':False,'free_bytes_preflight':free,'estimated_new_bytes':1300000000,'source_sha256':{str(x.relative_to(R)):sha(x) for x in [R/'scripts/init01.py',R/'scripts/hier01_chain_amend.py',R/'scripts/hier01_task.py',R/'scripts/load04_data.py',R/'scripts/load04_worker.py',R/'supervisor/feeding_loading/environment.py',R/'supervisor/feeding_loading/runner.py',R/'supervisor/adaptation/actions.py']}}
 dump(O/'protocol.json',p);receipts=[]
 for parent in parents:
  limit('preparation_deadline');env=None
  try:env,receipt,_=replay(parent,O/'qualification'/parent['id'],True);receipts.append(receipt)
  finally:
   if env:env.close()
 limit('preparation_deadline')
 dump(O/'qualification.json',{'passed':True,'parents':receipts,'epoch':time.time(),'free_bytes':shutil.disk_usage(O).free})
 print('INIT01_PREFIXES_QUALIFIED',flush=True)
def set_request(env,parent,branch):
 before=state(env);cfg=copy.deepcopy(parent['physical_config'] if branch!='F-preserve' else parent['entry']['config'])
 cfg['control']['max_steps']=env.steps+(60 if branch=='F-preserve' else 1460)
 env.config=cfg;env.episode_config=copy.deepcopy(cfg)
 assert state(env)==before
 return cfg
def record_teacher(env,parent,branch,out,previous):
 out.mkdir(parents=True,exist_ok=False);initial=env.truth();start=env.steps;cfg=copy.deepcopy(env.config)
 # Reuse the exact LOAD04 CaptureEnv recorder/absolute_target encoder from this boundary.
 env.counter=0;env.last_target=None;env.post=[];env.crop_audit=[];env.packet={k:[] for k in ['primary_visible','secondary_visible','primary_hidden','secondary_hidden','frame_action_indices','targets','proprio']};env.enabled=True
 rec=Recorder(env,out/'trajectory.jsonl.gz');raw=env.step;rich=[];commands=[];attempts=[];exception=None;parking=None
 d=env.env.sim.data;rob=env.env.robots[0];joint_ids=[env.env.sim.model.joint_name2id(n) for n in rob.robot_model.joints]
 ranges=env.env.sim.model.jnt_range[joint_ids].copy()
 def step(command):
  limit();before=env.observe();raw(command)
  commands.append({'action_index':len(commands),'pre_time_s':float(env.env.sim.data.time)-.05,'post_time_s':float(env.env.sim.data.time),'normalized_command':np.asarray(command).tolist(),'encoded_target':absolute_target(command,before,cfg).tolist(),'post_action_truth':env.truth(),'joint_position_rad':d.qpos[rob._ref_joint_pos_indexes].copy().tolist()})
  rich.append(h.base.native.b.rich(env,np.asarray(commands[-1]['encoded_target']),command))
 env.step=step
 writer=imageio.get_writer(out/'rollout.mp4',fps=5)
 def capture(scene):writer.append_data(np.concatenate([scene.frame('primary'),scene.frame('secondary')],1))
 capture(env);env.capture_callback=capture;imageio.imwrite(out/'initial.png',np.concatenate([env.frame(),env.frame('secondary')],1))
 try:
  if branch=='F-preserve':
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
 dump(out/'crop-audit.json',env.crop_audit);dump(out/'controller-rich.json',rich)
 (out/'robot_actions.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in commands))
 ep={'parent':parent['id'],'branch':branch,'initial_truth':initial,'final_truth':final,'stability_truths':stability,'attempts':attempts,'evaluation':score,'success':score['success'],'controller_exception':exception,'parking':parking,'control_steps':env.steps-start,'candidate_only':True,'uses_simulator_truth_for_control':branch=='L-teacher','instruction':task_prompt(cfg['task']),'config':cfg}
 dump(out/'episode.json',ep)
 alignment=(len(commands)==len(arrays['targets']) and np.all(np.isfinite(arrays['targets'])) and np.array_equal(arrays['frame_action_indices'],np.arange(0,len(commands),4)) and np.allclose(arrays['targets'],np.asarray([x['encoded_target'] for x in commands],np.float32),atol=0,rtol=0))
 assert alignment
 full=[int(i) for i in arrays['frame_action_indices'] if i+30<=len(commands)]
 q=np.asarray([x['joint_position_rad'] for x in commands]);excursions=np.maximum(ranges[:,0]-q,q-ranges[:,1]);maxex=float(np.maximum(excursions,0).max())
 upright=[]
 for t in env.post:
  r=Rotation.from_quat(np.asarray(t['object_quat_wxyz'])[[1,2,3,0]]).as_matrix();tilt=float(np.arccos(np.clip(r[2,2],-1,1)))
  upright.append(bool(t['bilateral_grasp_contact'] and t['object_position_m'][2]-initial['object_position_m'][2]>=.07 and tilt<=cfg['evaluation']['upright_tolerance_rad']))
 qualifies=bool(branch=='L-teacher' and score['success'] and any(upright) and alignment and not exception)
 report={'parent':parent['id'],'branch':branch,'strict_success':score['success'],'upright_bilateral_lift_70mm':any(upright),'upright_bilateral_lift_samples':sum(upright),'action_alignment_qualified':bool(alignment),'complete_30_action_windows':full,'candidate_positive':qualifies,'supplied_preservation':branch=='F-preserve','object_displacement_m':float(np.linalg.norm(np.asarray(final['object_position_m'])-initial['object_position_m'])),'max_joint_limit_excursion_rad':maxex,'joint_limit_excursion_steps':int(np.any(excursions>0,axis=1).sum()),'phase_nonconvergence':sum(not z['converged'] for a in attempts for z in a['phases']),'contact_steps':sum(t['grasp_contact'] for t in env.post),'actions':len(commands),'directory':str(out),'controller_exception':exception,'generated_robot_labels_not_human_actions':True}
 dump(out/'qualification.json',report)
 return report
def run():
 p=read(O/'protocol.json');assert read(O/'qualification.json')['passed']
 for f,s in p['source_sha256'].items():assert sha(R/f)==s,f
 rows=[];noise_ref=None
 for parent in p['parents']:
  for branch in p['branches']:
   limit();env=None;w=None;folder=O/'continuations'/parent['id']/branch
   try:
    env,pre,prev=replay(parent,folder/'prefix',False);cfg=set_request(env,parent,branch)
    if branch=='L-policy':
     mo=folder/'worker';mo.mkdir();h.base.OUT=mo
     w=h.base.Worker({'model':p['model'],'expected_initial_nontext':{}})
     source={'config':parent['physical_config'],'simulation_seed':parent['seed'],'requested_task':'loading','physical_scene_task':'loading','instruction':task_prompt(cfg['task']),'expected_initial_nontext':{},'source_demo_id':parent['id']}
     row,_,_=chain.segment(env,w,source,parent['id']+'-'+branch,folder/'episode',False,prev)
     row.update(parent=parent['id'],branch=branch,strict_success=row['requested_success'],candidate_positive=False)
     ep=read(folder/'episode/episode.json');ep['seed']=parent['seed'];ep['policy_seed']=912171;dump(folder/'episode/episode.json',ep)
     noise=h.base.noise_gate(folder/'episode',912171,noise_ref);noise_ref=noise_ref or folder/'episode';row['noise_gate']=noise
    else:row=record_teacher(env,parent,branch,folder/'episode',prev)
    rows.append(row);dump(O/'summary.json',rows);print(json.dumps(row),flush=True)
   finally:
    if w:w.close()
    if env:env.close()
  dump(O/(parent['id']+'-SEALED.json'),{'sealed':True,'epoch':time.time(),'continuations':3})
 assert len(rows)==12
 dump(O/'complete.json',{'completed':True,'continuations':12,'training':False,'epoch':time.time()})
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  else:run()
 except BaseException:dump(O/('STOPPED-'+sys.argv[1]+'.json'),{'epoch':time.time(),'error':traceback.format_exc()});raise
