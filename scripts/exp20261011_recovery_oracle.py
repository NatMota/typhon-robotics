"""Diagnostic-only recovery oracle from exactly replayed V4 failures; never reset after handoff."""
import argparse, copy, gzip, hashlib, json, math, os, sys, time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np
from scipy.spatial.transform import Rotation
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.runner import scripted_attempt
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.adaptation.actions import absolute_target,proprio_vector

CASES=[
 ('offaxis_held','benchmark-native/v4-m6/film-demo_0006',60),
 ('sideface_held','v4-search/attempt-0014-demo_0006',60),
 ('otherfamily_held','v4-search/attempt-0095-demo_0003',600),
 ('released_wrongyaw','benchmark-native/v4-m6/film-demo_0006',120),
 ('released_wrongface','v4-search/attempt-0027-demo_0006',160),
 ('missed_grasp','benchmark-native/v4-m6/film-demo_0002',90),
]
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def objrot(t):return Rotation.from_quat(np.asarray(t['object_quat_wxyz'])[[1,2,3,0]]).as_matrix()
def difference(actual,expected):
 p=max(float(np.linalg.norm(np.asarray(actual[k])-expected[k])) for k in ['eef_position_m','object_position_m'])
 r=math.degrees(Rotation.from_matrix(objrot(actual)@objrot(expected).T).magnitude())
 assert p<1e-6 and r<1e-4,('unchanged prefix diverged',p,r)
 assert actual['bilateral_grasp_contact']==expected['bilateral_grasp_contact']
 return p,r

def plan(root,out):
 out.mkdir(parents=True,exist_ok=False)
 entries=[]
 for name,rel,n in CASES:
  source=root/rel;sc=read(source/'scenario.json');ep=read(source/'episode.json');tr=[json.loads(s) for s in gzip.open(source/'trajectory.jsonl.gz','rt')]
  t=tr[n-1]['truth'];assert n<=ep['policy_actions_executed']
  entries.append({'id':name,'source':str(source),'source_demo_id':sc['source_demo_id'],'prefix_steps':n,'expected_start':t,
   'training_use':'quarantined_diagnostic_only; P_validation and already-inspected evaluation states; never merge into P_train',
   'source_sha256':{f:sha(source/f) for f in ['scenario.json','episode.json','trajectory.jsonl.gz']}})
 p={'version':'recovery-oracle-20261011-v1','cases':entries,'variants':['nominal_expert','grasp_transform_expert'],
  'qualification':'exact unchanged prefix replay, original scenario and seven-dimensional requested commands; no state reset/teleport after handoff',
  'controls':'same state per variant; deterministic; max 800 extra control steps; 24-sample settling window; original physics, scene, scoring unchanged',
  'object_geometry':'original M3 50x35x30 mm,80g provisional geometry preserved, not measured physical DUPLO',
  'information':'both experts use simulator truth; correction inputs contain camera RGB and robot proprio only; truth separate for diagnosis',
  'training':False,'checkpoint_updates':0,'code_sha256':sha(Path(__file__))}
 save(out/'protocol.json',p);(out/'source.py').write_bytes(Path(__file__).read_bytes());return p

class Recorder:
 def __init__(self,env,out):
  import imageio.v2 as imageio
  self.env=env;self.out=out;self.phase='handoff';self.rows=[];self.actions=[];self.targets=[];self.proprio=[];self.images=[];self.secondary=[];self.indices=[]
  self.writer=imageio.get_writer(str(out/'recovery.mp4'),fps=5,codec='libx264',quality=8)
  self.original_step=env.step;self.lowstep=env.env.step;self.applied=[]
  def capture(c,*a,**kw):self.applied.append(np.asarray(c).copy());return self.lowstep(c,*a,**kw)
  env.env.step=capture;env.step=self.step
 def step(self,command):
  env=self.env;i=len(self.rows);robot=env.env.robots[0];sim=env.env.sim
  r=sim.data.site_xmat[robot.eef_site_id].reshape(3,3).copy()
  obs={'proprio':{'eef_position_m':sim.data.site_xpos[robot.eef_site_id].copy(),'eef_quat_xyzw':Rotation.from_matrix(r).as_quat()}}
  p=proprio_vector(obs);p[9]=float(env.last_grip>0)
  target=absolute_target(command,obs,env.config)
  self.proprio.append(p);self.actions.append(np.asarray(command));self.targets.append(target)
  before=env.truth();joint=sim.data.qpos[robot._ref_joint_pos_indexes].copy()
  if i%4==0:
   a=env.frame();b=env.frame('secondary');self.images.append(a);self.secondary.append(b);self.indices.append(i);self.writer.append_data(np.concatenate([a,b],axis=1))
  self.original_step(command)
  self.rows.append({'index':i,'phase':self.phase,'pre_action_truth':before,'post_action_truth':env.truth(),
   'requested_command':np.asarray(command).tolist(),'applied_command':self.applied[-1].tolist(),
   'pre_action_joint_position_rad':joint.tolist(),'pre_action_eef_rotation_matrix':r.tolist()})
 def close(self):
  self.writer.close();self.env.step=self.original_step;self.env.env.step=self.lowstep
  np.savez_compressed(self.out/'correction.npz',primary=np.asarray(self.images,np.uint8),secondary=np.asarray(self.secondary,np.uint8),
   frame_action_indices=np.asarray(self.indices,np.int32),actions=np.asarray(self.actions,np.float32),targets=np.asarray(self.targets,np.float32),
   proprio=np.asarray(self.proprio,np.float32),phase=np.asarray([r['phase'] for r in self.rows]))
  with gzip.open(self.out/'privileged-trace.jsonl.gz','wt') as f:
   for row in self.rows:f.write(json.dumps(row)+'\n')

def graspaware(env,rec,cfg):
 phases=[];attempts=[];table=cfg['workspace']['table_z_m'];height=cfg['object']['height_m'];initial=env.truth()
 def phase(name,fn):
  rec.phase=name;before=env.steps;result=fn();phases.append({'phase':name,'steps':env.steps-before,'result':result,'truth':env.truth()});return result
 def moving(name,target,yaw,grip):return phase(name,lambda:bool(env.move_to(target,yaw,grip,limit=120)))
 if not initial['bilateral_grasp_contact']:
  source=np.asarray(initial['object_position_m']);yaw=initial['object_yaw_rad'];above=source.copy();above[2]=max(above[2]+.16,table+.17)
  grasp=source.copy();grasp[2]+=.005
  if not moving('approach',above,yaw,-1):return {'phases':phases,'reason':'approach_nonconvergence','attempts':attempts}
  if not moving('descend',grasp,yaw,-1):return {'phases':phases,'reason':'descend_nonconvergence','attempts':attempts}
  phase('close',lambda:env.tick(grasp,yaw,1,18));moving('lift',above,yaw,1)
  t=env.truth();verified=t['bilateral_grasp_contact'] and t['object_position_m'][2]-source[2]>=.07
  attempts.append({'verified_lift':bool(verified),'lift_truth':t})
  if not verified:return {'phases':phases,'reason':'regrasp_not_verified','attempts':attempts}
 else:attempts.append({'verified_lift':True,'inherited_supported_grasp':True,'lift_truth':initial})
 robot=env.env.robots[0];sim=env.env.sim;ctl=cfg['control'];bounds=np.asarray(ctl['workspace_bounds_m'])
 def object_move(name,xyz,desired,limit=160,grip=1):
  rec.phase=name;before=env.steps;stable=0;max_change=0.;firstg=None;reason='nonconvergence'
  for _ in range(limit):
   t=env.truth();curR=sim.data.site_xmat[robot.eef_site_id].reshape(3,3);curP=sim.data.site_xpos[robot.eef_site_id];objR=objrot(t);objP=np.asarray(t['object_position_m'])
   G=curR.T@objR;d=curR.T@(objP-curP)
   if firstg is None:firstg=G.copy()
   max_change=max(max_change,math.degrees(Rotation.from_matrix(G@firstg.T).magnitude()))
   goalR=desired@G.T;goalP=np.asarray(xyz)-goalR@d
   if not t['bilateral_grasp_contact']:reason='lost_bilateral_contact';break
   if (goalP<bounds[:,0]).any() or (goalP>bounds[:,1]).any():reason='eef_target_outside_workspace';break
   pos=np.linalg.norm(objP-xyz);err=Rotation.from_matrix(desired@objR.T).magnitude()
   stable=stable+1 if pos<.004 and err<math.radians(3.5) else 0
   if stable>=4:reason='converged';break
   dp=(goalP-curP)/ctl['translation_step_m'];dr=Rotation.from_matrix(goalR@curR.T).as_rotvec()/ctl['rotation_step_rad']
   env.step(np.r_[np.clip(dp,-.4,.4),np.clip(dr,-.35,.35),grip])
  phases.append({'phase':name,'steps':env.steps-before,'result':reason,'max_grasp_change_deg':max_change,'truth':env.truth()})
  return reason=='converged'
 current=env.truth();p=np.asarray(current['object_position_m']);lift=p.copy();lift[2]=max(p[2],table+.18)
 if not object_move('stabilize_lift',lift,objrot(current)):return {'phases':phases,'reason':'lift_nonconvergence_or_contact_loss','attempts':attempts}
 goalR=Rotation.from_euler('z',cfg['task']['target_yaw_rad']).as_matrix()
 if not object_move('orient',lift,goalR):return {'phases':phases,'reason':'orientation_nonconvergence_or_contact_loss','attempts':attempts}
 above=np.r_[cfg['task']['target_position_xy_m'],lift[2]]
 if not object_move('transfer',above,goalR):return {'phases':phases,'reason':'transfer_nonconvergence_or_contact_loss','attempts':attempts}
 dest=np.r_[cfg['task']['target_position_xy_m'],table+height/2+.008]
 if not object_move('place',dest,goalR):return {'phases':phases,'reason':'placement_nonconvergence_or_contact_loss','attempts':attempts}
 # Retain the measured wrist goal while opening, never rederive a grasp after release.
 wristR=sim.data.site_xmat[robot.eef_site_id].reshape(3,3).copy();wristP=sim.data.site_xpos[robot.eef_site_id].copy()
 def wrist(name,xyz,steps):
  rec.phase=name
  for _ in range(steps):
   curR=sim.data.site_xmat[robot.eef_site_id].reshape(3,3);curP=sim.data.site_xpos[robot.eef_site_id]
   env.step(np.r_[np.clip((xyz-curP)/ctl['translation_step_m'],-.4,.4),np.clip(Rotation.from_matrix(wristR@curR.T).as_rotvec()/ctl['rotation_step_rad'],-.35,.35),-1])
 wrist('release',wristP,20);retreat=wristP.copy();retreat[2]=min(1.12,retreat[2]+.15);wrist('retreat',retreat,35)
 return {'phases':phases,'reason':'completed_controller_sequence','attempts':attempts}

def run(root,out,case_id=None,variant=None):
 protocol=read(out/'protocol.json');selected=[c for c in protocol['cases'] if case_id is None or c['id']==case_id]
 for c in selected:
  for v in ([variant] if variant else protocol['variants']):
   dest=out/c['id']/v
   if (dest/'summary.json').exists():continue
   dest.mkdir(parents=True,exist_ok=False);sc=read(Path(c['source'])/'scenario.json');ep=read(Path(c['source'])/'episode.json')
   tr=[json.loads(s) for s in gzip.open(Path(c['source'])/'trajectory.jsonl.gz','rt')];cfg=copy.deepcopy(sc['config'])
   env=None;rec=None;started=time.perf_counter();qualification=[];result={};exception=None;stability=[]
   try:
    env=FeedingLoadingEnv(cfg);env.reset(sc['seed']);qualification.append(difference(env.truth(),ep['initial_truth']))
    for row in tr[:c['prefix_steps']]:env.step(row['requested_action']);qualification.append(difference(env.truth(),row['truth']))
    start=env.truth();sim=env.env.sim
    state={'truth':start,**{k:getattr(sim.data,k).tolist() for k in ['qpos','qvel','act','ctrl','qacc_warmstart']}}
    common=dest.parent/'qualified-start-state.json'
    if common.exists():assert state==read(common),'branches do not start identically'
    else:save(common,state)
    save(dest/'qualification.json',{'passed':True,'max_position_error_m':max(x[0] for x in qualification),'max_rotation_error_deg':max(x[1] for x in qualification),'prefix_steps':c['prefix_steps'],'state_sha256':sha(common)})
    import imageio.v2 as imageio
    imageio.imwrite(dest/'initial-primary.png',env.frame());imageio.imwrite(dest/'initial-secondary.png',env.frame('secondary'))
    env.config['control']['max_steps']=env.steps+800 # only diagnostic continuation budget changes; replay already qualified
    rec=Recorder(env,dest)
    try:
     if v=='nominal_expert':
      rec.phase='nominal_scripted_attempt';attempt,retreat,yaw=scripted_attempt(env,cfg);result={'attempts':[attempt],'reason':'completed_nominal_script','phases':attempt['phases']}
     else:result=graspaware(env,rec,cfg)
     rec.phase='settle';last=env.last_grip
     for _ in range(24):env.step(np.r_[np.zeros(6),last]);stability.append(env.truth())
    except Exception as e:exception={'type':type(e).__name__,'message':str(e)}
    final=env.truth();scores=score_episode(start,final,stability,cfg,result.get('attempts',[]))
    if exception:scores['success']=False
    opening=[r for r in rec.rows if r['requested_command'][-1]<0 and r['pre_action_truth']['bilateral_grasp_contact']]
    release_errors=[{'index':r['index'],'phase':r['phase'],'yaw_error_deg':math.degrees(abs(math.atan2(math.sin(r['pre_action_truth']['object_yaw_rad']-cfg['task']['target_yaw_rad']),math.cos(r['pre_action_truth']['object_yaw_rad']-cfg['task']['target_yaw_rad'])))),'object_height_m':r['pre_action_truth']['object_position_m'][2]} for r in opening]
    rec.close();rec=None
    imageio.imwrite(dest/'final-primary.png',env.frame());imageio.imwrite(dest/'final-secondary.png',env.frame('secondary'))
    summary={'case':c['id'],'variant':v,'source_demo_id':c['source_demo_id'],'source':c['source'],'prefix_steps':c['prefix_steps'],
     'qualified_replay':True,'same_start_state_sha256':sha(common),'initial_truth':start,'final_truth':final,'evaluation':scores,
     'controller':result,'exception':exception,'wall_time_s':time.perf_counter()-started,'release_commands_while_held':release_errors,
     'uses_privileged_truth':True,'reset_after_handoff':False,'training_allowed':False,'quarantine_reason':c['training_use'],
     'correction_archive_sha256':sha(dest/'correction.npz')}
    save(dest/'summary.json',summary);print(json.dumps({'case':c['id'],'variant':v,'success':scores['success'],'position_mm':scores['position_error_m']*1000,'yaw':scores['orientation_error_deg'],'criteria':scores['criteria'],'reason':result.get('reason'),'exception':exception}),flush=True)
   except Exception as e:
    save(dest/'failure.json',{'type':type(e).__name__,'message':str(e),'qualification_checks_completed':len(qualification)});raise
   finally:
    if rec:rec.close()
    if env:env.close()

def report(out):
 p=read(out/'protocol.json');rows=[];manifest=[]
 for c in p['cases']:
  for v in p['variants']:
   dest=out/c['id']/v
   if not (dest/'summary.json').exists():continue
   s=read(dest/'summary.json');a=np.load(dest/'correction.npz');n=len(a['actions']);assert n==len(a['targets'])==len(a['proprio'])
   assert len(a['primary'])==len(a['frame_action_indices']);assert a['frame_action_indices'].max()<n
   assert np.isfinite(a['targets']).all();rows.append(s)
   manifest.append({'id':c['id']+'/'+v,'source_demo_id':c['source_demo_id'],'source_prefix_steps':c['prefix_steps'],
    'path':str(dest/'correction.npz'),'sha256':sha(dest/'correction.npz'),'actions':n,'frames':len(a['primary']),
    'successful_correction':s['evaluation']['success'],'split':'quarantined_diagnostic_only','allowed_for_training':False,
    'labels':'generated privileged-recovery expert, not human motor actions','inputs':'RGB before action, robot proprio every action; no truth state in NPZ',
    'proprio_schema':'20D interleaved xyz/6D rotation/gripper-command/zero pad, measured pre-action EEF; unlike M3 previous-target history',
    'target_schema':'20D absolute_target from pre-action measured EEF and normalized OSC command;20Hz; frame_action_indices5Hz',
    'phases':sorted(set(a['phase'].tolist()))})
 save(out/'correction-dataset-manifest.json',{'episodes':manifest,'training_performed':False,'holdouts_changed':False})
 totals={v:{'success':sum(r['evaluation']['success'] for r in rows if r['variant']==v),'attempts':sum(r['variant']==v for r in rows)} for v in p['variants']}
 save(out/'validation-summary.json',{'totals':totals,'completed':len(rows),'expected':len(p['cases'])*len(p['variants']),
  'qualified_all':all(r['qualified_replay'] for r in rows),'correction_actions':sum(m['actions'] for m in manifest),
  'training_eligible_actions':0,'independent_sampling':False,'interpretation':'selected V4 failure-state oracle feasibility, not overall robot success rate'})
 lines=['# Recovery oracle from V4 failure states','',f"Completed {len(rows)} / {len(p['cases'])*2} deterministic branches; no model training.",'',
 '| State | Expert | Task success | Position mm | Yaw ° | Face up | Released | Stable | Reason |','|---|---|---:|---:|---:|---|---|---|---|']
 for r in rows:
  e=r['evaluation'];cr=e['criteria'];lines.append(f"| {r['case']} | {r['variant']} | {e['success']} | {1000*e['position_error_m']:.2f} | {e['orientation_error_deg']:.2f} | {cr['marked_face_up']} | {cr['released']} | {cr['stable']} | {r['controller'].get('reason')} |")
 lines+=['','Both variants start from the exact same replayed physical state, including robot/object velocities and controller history. No object teleportation, reattachment, or environment reset occurs after handoff. The nominal expert is unchanged. The grasp-transform variant is a new diagnostic privileged controller: it measures the current full object-to-gripper transform, retains the grasp while orienting the object, and opens only after object-goal convergence.','',
 'These are selected failure states, not independent random trials. The suite includes previously inspected validation takes; every correction is quarantined and forbidden from the existing train split. A new future train/holdout protocol would be necessary before using these labels. Failed traces are preserved and excluded from any successful-correction subset.','',
 'Correction NPZ contains pre-action RGB, measured robot proprioception, normalized OSC commands and encoded absolute robot targets. Truth remains in separate diagnostic traces. This observation convention differs from the old M3 previous-command proprio convention and requires explicit training adapter treatment; it must not be silently mixed.','',
 'Reproduction: `.venv-sim/bin/python scripts/exp20261011_recovery_oracle.py --root /var/tmp/typhon-film-feeding-20261011-v3 --output FRESH_DIR plan`, then same command with `run`, then `report`. Use the saved source.py to reproduce the exact version.']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(totals),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','run','report']);p.add_argument('--root',type=Path,default=Path('/var/tmp/typhon-film-feeding-20261011-v3'));p.add_argument('--output',type=Path,required=True);p.add_argument('--case');p.add_argument('--variant',choices=['nominal_expert','grasp_transform_expert']);a=p.parse_args()
 if a.mode=='plan':plan(a.root,a.output)
 elif a.mode=='run':run(a.root,a.output,a.case,a.variant)
 else:report(a.output)
