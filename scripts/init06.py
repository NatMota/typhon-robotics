"""INIT06: qualify two contrasting references from exact learner-visited q1 states."""
import os,sys,json,time,copy,gzip,hashlib,pickle,traceback,subprocess,shutil
from pathlib import Path
W=Path(__file__).resolve().parents[1];os.chdir(W);sys.path[:0]=[str(W),str(W/'scripts')]
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np
from scipy.spatial.transform import Rotation
import imageio.v2 as imageio
import init01 as initial
import init02_eval as ev
import hier01_task as h
import hier01_chain_amend as chain
from load04_common import endpoint
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.feeding_loading.runner import scripted_attempt
from supervisor.baseline.benchmark import Recorder
from supervisor.baseline.xvla import task_prompt
from supervisor.adaptation.actions import proprio_vector,absolute_target
O=Path('/var/tmp/typhon-init06-20261015');D=Path('/var/tmp/typhon-init02-20261015')
read,dump,sha=initial.read,initial.dump,initial.sha
state=chain.state_blob
def limit(stage='execution'):
 c=read(O/'clock.json'); mins={'prepare':15,'principal':35,'second':45,'execution':45,'archive':60}[stage]
 assert time.time()<c['t0']+mins*60,('INIT06 deadline',stage)
def contact(r):
 return bool(r['object_truth']['grasp_contact']) or any('gripper' in str(c)and'cube'in str(c)for c in r['contacts'])
def trace(path):
 with gzip.open(path,'rt')as f:return [json.loads(s)for s in f]
def spec(parent):
 cases=read(D/'evaluation-cases.json')['core']
 source=next(s for s in cases if s['id']==f'init-{parent}-feeding-912171')
 p=next(s for s in read(initial.O/'protocol.json')['parents']if s['id']==parent)
 f=D/'episodes/T'/source['id'];tr=trace(f/'controller_trace.jsonl.gz')
 ep=read(f/'episode.json')
 assert ep['model_sha256']=='4d7a4a2547f084b6026164f27a6b7d43b5cb4d3b83ee4875343d108908f35636'
 assert len(tr)==1460 and tr[29]['step']==ep['control_step_start']+30
 assert not any(contact(r)for r in tr[:30])
 disp=max(float(np.linalg.norm(np.asarray(r['object_truth']['object_position_m'])-source['initial_truth']['object_position_m']))for r in tr[:30])
 assert disp<=.001
 feeding=score_episode(source['initial_truth'],tr[29]['object_truth'],[r['object_truth']for r in tr[6:30]],endpoint(source['config'],'feeding'),[])
 loading=score_episode(source['initial_truth'],tr[29]['object_truth'],[r['object_truth']for r in tr[6:30]],endpoint(source['config'],'loading'),[])
 assert feeding['success'] and not loading['success']
 paths=[f/x for x in ['episode.json','controller_trace.jsonl.gz','trajectory.jsonl.gz','policy_io.npz','predictions.npz','input-audit/input-001.json','boundary-before.pkl','initial-live.npz']]
 paths += [Path(p['actions_path']),Path(p['entry']['source_path'])/'episode.json',initial.O/'qualification'/parent/'terminal-state.pkl']
 return {'parent':parent,'source':source,'parent_entry':p,'folder':str(f),'source_sha256':{str(x):sha(x)for x in paths},'q1_step':tr[29]['step'],'policy_seed':912171,'q':1,'prefix_commands':30,'archived_first30_contact':False,'maximum_archived_block_displacement_m':disp,'feeding_initial_score':feeding,'loading_initial_score':loading}
def prepared(folder,sp,task):
 ins=read(Path(sp['folder'])/'input-audit/input-001.json')['instruction']if task=='feeding'else read(D/'episodes/T'/f"init-{sp['parent']}-loading-912171"/'input-audit/input-000.json')['instruction']
 a=read(Path(sp['folder'])/'input-audit/input-001.json')
 dump(folder/'input-request.json',{'instruction':ins,'task':task,'archived_input_hashes':a['input_hashes']})
 p=subprocess.run([str(W/'.venv-xvla/bin/python'),'scripts/init06_preprocess.py',str(folder)],cwd=W,capture_output=True,text=True)
 (folder/'input-qualification.log').write_text(p.stdout+p.stderr)
 assert p.returncode==0,('Actual q1 prepared input qualification failed',str(folder))
 return read(folder/'input-qualification.json')
def reconstruct(sp,folder,task,canonical=False):
 folder.mkdir(parents=True,exist_ok=False)
 env=None
 try:
  # Qualified original feeding prefix; only its established semantic exclusions.
  env,previous=ev.initialise(sp['source'],folder/'prefix')
  f=Path(sp['folder']);old=trace(f/'controller_trace.jsonl.gz')[:30]
  before=state(env);assert before==(f/'boundary-before.pkl').read_bytes(),'Historical q0 semantic state mismatch'
  diffs=[];new=[]
  for row in old:
   env.step(row['command']);cur=h.base.native.b.rich(env,np.asarray(row['target'],np.float32),row['command'])
   e=initial.numeric(cur,row);diffs.append(e);new.append(cur);assert e<1e-8,('Archived control boundary mismatch',e)
  assert env.steps==sp['q1_step']
  blob=state(env);(folder/'q1-state.pkl').write_bytes(blob)
  ref=O/'qualification'/sp['parent']/'q1-state.pkl'
  if not canonical:assert blob==ref.read_bytes(),'New q1 semantic state mismatch'
  ob=env.observe();assert state(env)==blob
  with np.load(f/'predictions.npz')as z:hist=z['histories'][1].copy();chunk=z['chunks'][0].copy()
  np.testing.assert_array_equal(hist[:9],chunk[-1,:9])
  np.testing.assert_array_equal(hist[9:],np.zeros(11,dtype=hist.dtype))
  with np.load(f/'policy_io.npz')as z:
   np.testing.assert_allclose(ob['proprio']['eef_position_m'],z['eef_position_m'][1],atol=1e-8,rtol=0)
   np.testing.assert_allclose(ob['proprio']['eef_quat_xyzw'],z['eef_quat_xyzw'][1],atol=1e-8,rtol=0)
   rawdiff={k:{'changed_values':int((ob['rgb'][k]!=z[k][1]).sum()),'maximum_level_delta':int(np.abs(ob['rgb'][k].astype(int)-z[k][1].astype(int)).max())}for k in ['primary','secondary']}
  np.savez_compressed(folder/'q1-observation.npz',primary=ob['rgb']['primary'],secondary=ob['rgb']['secondary'],proprio=hist,measured_proprio=proprio_vector(ob))
  with gzip.open(folder/'replayed-q0-controls.jsonl.gz','wt')as f1:
   for r in new:f1.write(json.dumps(r,allow_nan=False)+'\n')
  inp=prepared(folder,sp,task)
  before=state(env);cfg=copy.deepcopy(endpoint(sp['source']['config'],task));cfg['control']['max_steps']=env.steps+(60 if task=='feeding'else 1430)
  env.config=cfg;env.episode_config=copy.deepcopy(cfg);assert state(env)==before
  rec={'passed':True,'parent':sp['parent'],'task':task,'canonical':canonical,'historical_q0_complete_semantic_state_exact':True,'historical_first30_rich_trace_max_abs_difference':max(diffs),'historical_q1_compared_fields':list(old[-1]),'historical_q1_full_state_archived':False,'historical_unarchived_fields':'qvel/full warmstart and every internal controller field at q1 were not separately archived; deterministic exact-q0 action replay plus all available rich trace and actual prepared inputs qualify reconstruction, not direct full-state historical comparison','paired_full_semantic_state_exact':not canonical,'q1_semantic_sha256':hashlib.sha256(blob).hexdigest(),'raw_image_difference':rawdiff,'actual_input_qualification':inp,'maximum_suffix_steps':60 if task=='feeding'else 1430,'no_reset_home_pose_injection':True,'epoch':time.time()}
  dump(folder/'reconstruction.json',rec)
  return env,hist,[r['object_truth']for r in new[-24:]],rec
 except BaseException:
  if env:env.close()
  dump(folder/'FAILED.json',{'error':traceback.format_exc(),'epoch':time.time()});raise
def prepare():
 limit('prepare')
 p={'experiment':'INIT06','ticket':11077,'clock':read(O/'clock.json'),'candidate_only':True,'training':False,'new_policy_queries':0,'parents':[spec('demo_0001'),spec('demo_0002')],'max_continuations':4,'conditional_second':'complete first hold/loading pair qualifies','hold_ticks':60,'loading_remaining_ticks_including_settle':1430,'hold_target':'measured full pose captured once, gripper open','teacher':'unchanged INIT01 scripted_attempt, max2 attempts, original open park and24 settle','pair_semantics':'same q1 complete physical state and actual processed nontext input, original F/L instructions','historical_qualification':'exact original q0 semantic state, exact recorded30 commands and rich trace tolerance1e-8, exact prepared q1 inputs','source_code_sha256':{str(x.relative_to(W)):sha(x)for x in [W/'scripts/init01.py',W/'scripts/init02_eval.py',W/'scripts/load04_data.py',W/'scripts/hier01_chain_amend.py',W/'supervisor/feeding_loading/runner.py',W/'supervisor/feeding_loading/environment.py',W/'supervisor/adaptation/actions.py']}}
 assert shutil.disk_usage(O).free>1500*1024**2
 dump(O/'protocol.json',p)
 for s in p['parents']:
  limit('prepare');env=None
  try:env,_,_,_=reconstruct(s,O/'qualification'/s['parent'],'feeding',True)
  finally:
   if env:env.close()
 limit('prepare');dump(O/'prepared.json',{'passed':True,'epoch':time.time(),'parents':2,'physics_suffixes':0});print('BOTH_Q1_STATES_QUALIFIED',flush=True)
def qualify_suffix(out,sp,hist,branch):
 ep=read(out/'episode.json');r=read(out/'qualification.json');rich=read(out/'controller-rich.json')
 commands=[json.loads(s)for s in (out/'robot_actions.jsonl').read_text().splitlines()];states=[x['post_action_truth']for x in commands]
 initial_xyz=np.asarray(ep['initial_truth']['object_position_m'])
 disp=max([float(np.linalg.norm(np.asarray(t['object_position_m'])-initial_xyz))for t in states]or[0.])
 anycontact=any(contact(t)for t in rich)
 good=np.array([t['object_truth']['bilateral_grasp_contact']and t['object_truth']['object_position_m'][2]-initial_xyz[2]>=.07 and t['object_tilt_deg']<=np.degrees(ep['config']['evaluation']['upright_tolerance_rad'])for t in rich],bool)
 runs=[];start=None
 for i,v in enumerate(np.r_[good,False]):
  if v and start is None:start=i
  elif not v and start is not None:runs.append({'first_action':start,'last_action':i-1,'samples':i-start,'start_time_s':states[start]['sim_time_s'],'end_time_s':states[i-1]['sim_time_s']});start=None
 lift=any(x['samples']>=3 for x in runs)
 with np.load(out/'candidate.npz')as z:
  np.testing.assert_array_equal(z['proprio'][0],hist)
  np.savez_compressed(out/'q1-observation.npz',primary=z['primary_visible'][0],secondary=z['secondary_visible'][0],proprio=z['proprio'][0])
  full=[int(i)for i in z['frame_action_indices']if i+30<=len(z['targets'])]
  assert full==r['complete_30_action_windows'] and full
  assert np.array_equal(z['frame_action_indices'],np.arange(0,len(commands),4))
 inp=prepared(out,sp,'feeding'if branch=='F-preserve'else'loading')
 keep=bool(ep['evaluation']['success']and not anycontact and disp<=.001 and r['actions']==60)
 clean=bool(ep['evaluation']['success']and lift and r['phase_nonconvergence']==0 and ep['parking']['converged']and r['max_joint_limit_excursion_rad']==0 and not ep['controller_exception'])if branch=='L-teacher'else True
 qualified=bool(r['action_alignment_qualified']and inp['passed']and not ep['controller_exception']and(keep if branch=='F-preserve'else clean))
 r.update(qualified=qualified,preservation_qualified=keep if branch=='F-preserve'else None,maximum_object_displacement_mm=disp*1000,any_gripper_object_contact=anycontact,upright_supported_lift_70mm_3samples=lift,lift_intervals=runs,actual_q1_previous_target_proprio_preserved=True,initial_prepared_input=inp,labels_not_terminal_padded=True,phase_parking_converged=ep['parking']['converged']if ep['parking']else None,epoch=time.time())
 dump(out/'qualification-final.json',r);return r
def run():
 assert read(O/'prepared.json')['passed']
 p=read(O/'protocol.json');rows=[]
 for name,digest in p['source_code_sha256'].items():assert sha(W/name)==digest
 for i,s in enumerate(p['parents']):
  limit('principal'if i==0 else'second');pair=[]
  for branch in ['F-preserve','L-teacher']:
   limit('principal'if i==0 else'second');env=None;folder=O/'continuations'/s['parent']/branch
   assert not folder.exists(),'No outcome retries'
   try:
    env,hist,prev,reconstruction=reconstruct(s,folder/'reconstruction','feeding'if branch=='F-preserve'else'loading')
    before=state(env)
    r=record_teacher(env,s['parent_entry'],branch,folder/'episode',prev,hist)
    r=qualify_suffix(folder/'episode',s,hist,branch)
    rows.append(r);pair.append(r);dump(O/'summary.json',rows);print(json.dumps(r),flush=True)
   finally:
    if env:env.close()
  a,b=[O/'continuations'/s['parent']/z/'reconstruction'for z in ['F-preserve','L-teacher']]
  assert(a/'q1-state.pkl').read_bytes()==(b/'q1-state.pkl').read_bytes()
  with np.load(a/'actual-prepared.npz')as x,np.load(b/'actual-prepared.npz')as y:
   for k in x.files:
    if k!='input_ids':np.testing.assert_array_equal(x[k],y[k])
  ok=all(x['qualified']for x in pair)
  dump(O/(s['parent']+'-SEALED.json'),{'qualified_pair':ok,'epoch':time.time(),'continuations':2,'complete_semantic_state_pair_equal':True,'nontext_inputs_pair_equal':True})
  if not ok:
   dump(O/'complete.json',{'completed':True,'qualified_pairs':i,'continuations':len(rows),'second_parent_unexecuted':i==0,'gate_failure':s['parent'],'epoch':time.time()});return
 dump(O/'complete.json',{'completed':True,'qualified_pairs':2,'continuations':4,'training':False,'epoch':time.time()})
def record_teacher(env,parent,branch,out,previous,initial_history):
 out.mkdir(parents=True,exist_ok=False);initial=env.truth();start=env.steps;cfg=copy.deepcopy(env.config)
 # Reuse the exact LOAD04 CaptureEnv recorder/absolute_target encoder from this boundary.
 env.counter=0;env.last_target=initial_history.copy();env.post=[];env.crop_audit=[];env.packet={k:[] for k in ['primary_visible','secondary_visible','primary_hidden','secondary_hidden','frame_action_indices','targets','proprio']};env.enabled=True
 rec=Recorder(env,out/'trajectory.jsonl.gz');raw=env.step;rich=[];commands=[];attempts=[];exception=None;parking=None;all_hold_observations=[]
 d=env.env.sim.data;rob=env.env.robots[0];joint_ids=[env.env.sim.model.joint_name2id(n) for n in rob.robot_model.joints]
 ranges=env.env.sim.model.jnt_range[joint_ids].copy()
 def step(command):
  limit();before=env.observe()
  if branch=='F-preserve':all_hold_observations.append({'primary':before['rgb']['primary'].copy(),'secondary':before['rgb']['secondary'].copy(),'measured_proprio':proprio_vector(before)})
  raw(command)
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
 if all_hold_observations:np.savez_compressed(out/'all60-hold-preaction-observations.npz',**{k:np.asarray([x[k]for x in all_hold_observations])for k in all_hold_observations[0]})
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
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  else:run()
 except BaseException:dump(O/('STOPPED-'+sys.argv[1]+'.json'),{'epoch':time.time(),'error':traceback.format_exc()});raise
