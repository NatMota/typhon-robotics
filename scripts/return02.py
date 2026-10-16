
"""RETURN02 fixed inward acquisition corridor. No inference or training."""
import os,sys,json,time,copy,gzip,traceback,hashlib,shutil
from pathlib import Path
W=Path(__file__).resolve().parents[1];os.chdir(W);sys.path[:0]=[str(W),str(W/'scripts')]
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np
import return01 as r
import return02_geometry as geo
from return02_expert import scripted_attempt
from load04_common import endpoint
O=Path('/var/tmp/typhon-return02-20261016');OLD=Path('/var/tmp/typhon-return01-20261016')
read,dump,sha,state=r.read,r.dump,r.sha,r.state
PHASE='preparation'
def limit(*args):
 key='preparation_deadline'if PHASE=='preparation'else'execution_deadline'
 assert time.time()<read(O/'clock.json')[key],('RETURN02 frozen deadline',PHASE)
r.O=O;r.limit=limit;r.h.O=O;r.h.limit=limit;r.chain.O=O;r.scripted_attempt=scripted_attempt
def geometry(env,parent):
 m,d=geo.rawmodel(env);sim=env.env.sim;rob=env.env.robots[0]
 names={sim.model.geom_id2name(i):i for i in range(m.ngeom)}
 wall=names['fixture_wall_1'];centre=np.asarray(d.xpos[int(m.geom_bodyid[wall])]).copy();wp=np.asarray(d.geom_xpos[wall]).copy()
 inward=centre-wp;inward[2]=0;inward/=np.linalg.norm(inward);delta=.002*inward
 assert abs(delta[2])<1e-15
 rm=np.asarray(d.geom_xmat[wall]).reshape(3,3);assert max(abs(rm.T@inward))>1-1e-10
 local={};world={};g=np.asarray(d.site_xpos[rob.eef_site_id]).copy();G=np.asarray(d.site_xmat[rob.eef_site_id]).reshape(3,3)
 for name,i in names.items():
  if not name or not(m.geom_contype[i]or m.geom_conaffinity[i]):continue
  if not any(s in name for s in ['gripper','fixture','table','cube']):continue
  Rg=np.asarray(d.geom_xmat[i]).reshape(3,3);v=geo.vertices(m,i)@Rg.T+d.geom_xpos[i]
  world[name]=v
  if name.startswith('gripper'):local[name]=(v-g)@G
 assert local and 'gripper0_finger2_collision'in local
 obstacles={n:v for n,v in world.items()if not n.startswith('gripper')}
 t=env.truth();source=np.asarray(t['object_position_m']);yaw=t['object_yaw_rad'];desired=env.wrist_orientation(yaw)
 contact=read(OLD/'analysis-v2/first-contact-diagnostic.json')
 old=next(x for x in contact['rows']if x['parent']==parent['id'])
 heights=sorted(set([float(source[2]+.005),float(source[2]+.16)]+[z['action']['post_action_truth']['eef_position_m'][2]for z in old['contacts']]))
 refs=[];intersections=[];bounds=np.asarray(env.config['control']['workspace_bounds_m']);wall_inner=float(np.max(world['fixture_wall_1']@inward))
 for h in heights:
  nom=source.copy();nom[2]=h;shifted=nom+delta
  assert np.all(shifted>=bounds[:,0])and np.all(shifted<=bounds[:,1])
  points={n:v@desired.T+shifted for n,v in local.items()}
  nominal={n:v@desired.T+nom for n,v in local.items()}
  shell0=float(np.min(nominal['gripper0_finger2_collision']@inward)-wall_inner)
  shell2=float(np.min(points['gripper0_finger2_collision']@inward)-wall_inner)
  assert abs((shell2-shell0)-.002)<1e-10
  hit=[]
  for name,v in points.items():
   lo,hi=v.min(0),v.max(0)
   for on,ov in obstacles.items():
    ol,oh=ov.min(0),ov.max(0)
    if np.all(lo<=oh)and np.all(hi>=ol):hit.append({'gripper':name,'obstacle':on,'overlap_xyz_m':(np.minimum(hi,oh)-np.maximum(lo,ol)).tolist()})
  intersections +=[{'reference_z_m':h,**x}for x in hit]
  refs.append({'nominal_xyz':nom.tolist(),'shifted_xyz':shifted.tolist(),'wrist_rotation':desired.tolist(),'finger2_wall_plane_clearance_E0_m':shell0,'finger2_wall_plane_clearance_E2_m':shell2,'intersections':hit})
 # Open-aperture containment at shifted grasp reference, with compiled finger shells.
 grasp=source+delta;grasp[2]+=.005
 fingers=[]
 for token in ['finger1_collision','finger2_collision']:
  key=next(n for n in local if n.endswith(token));fingers.append(local[key]@desired.T+grasp)
 axis=fingers[1].mean(0)-fingers[0].mean(0);axis/=np.linalg.norm(axis)
 left=float(np.max(fingers[0]@axis));right=float(np.min(fingers[1]@axis))
 cube=np.concatenate([v for n,v in obstacles.items()if'cube'in n]);clo,chi=float(np.min(cube@axis)),float(np.max(cube@axis))
 aperture=left<clo and chi<right
 result={'parent':parent['id'],'wall':'fixture_wall_1','wall_geom_id':wall,'fixture_body_position':centre.tolist(),'wall_position':wp.tolist(),'inward_normal_world':inward.tolist(),'delta_xyz_m':delta.tolist(),'sampled_references':refs,'all_references_within_workspace':True,'finger_shell_away_by_exactly_2mm':True,'shifted_intersections':intersections,'open_aperture':{'axis':axis.tolist(),'left_inner':left,'right_inner':right,'brick_min':clo,'brick_max':chi,'encloses_block':bool(aperture)},'collision_geometry':{n:{'world_min':v.min(0).tolist(),'world_max':v.max(0).tolist()}for n,v in world.items()},'method':'Existing detached compiled-vertex conservative AABB screen at nominal acquisition references and archived observed-contact heights; fixed open aperture; no continuous-collision guarantee','passed':not intersections and bool(aperture)}
 return result
def qualified_prefix(parent,d,canonical=False):
 env,rec,previous=r.replay(parent,d,canonical)
 assert(d/'terminal-state.pkl').read_bytes()==(OLD/'qualification'/parent['id']/'terminal-state.pkl').read_bytes()
 return env,rec,previous
def run_teacher(env,parent,condition,out,previous,prop,delta):
 env.return02_offset=np.asarray(delta if condition=='E2'else[0.,0.,0.]);env.return02_phase='initial'
 original_tick=env.tick;original_move=env.move_to;original_step=env.step
 mapping=geo.mapped(env);logs=[];current={}
 def tick(target_xyz,yaw,grip,steps=1):
  nonlocal current
  phase=env.return02_phase
  if np.array_equal(np.asarray(target_xyz),np.array([-.25,-.25,1.08])):phase='parking_or_settling'
  shifted=phase in ['approach','descend','close','lift','failed_grasp_retreat','failed_grasp_release']
  target=np.asarray(target_xyz,float)
  current={'phase':phase,'used_reference_xyz_m':target.tolist(),'nominal_reference_xyz_m':(target-env.return02_offset if shifted else target).tolist(),'applied_offset_xyz_m':env.return02_offset.tolist()if shifted else[0.,0.,0.],'yaw_rad':float(yaw),'full_wrist_rotation':env.wrist_orientation(yaw).tolist(),'gripper_command':float(grip)}
  return original_tick(target_xyz,yaw,grip,steps=steps)
 def step(command):
  original_step(command);before=state(env);f=geo.forces(env,mapping);assert state(env)==before
  logs.append({'index':len(logs),'raw_reference':copy.deepcopy(current),'bounded_command':np.asarray(command).tolist(),'actual_truth':env.truth(),'force_telemetry':f})
 env.tick=tick;env.step=step
 try:
  row=r.record_teacher(env,parent,'F-teacher',out,previous,prop)
 finally:
  env.tick=original_tick;env.move_to=original_move
 dump(out/'reference-force-trace.json',logs);dump(out/'actuation-mapping.json',mapping)
 assert len(logs)==row['actions']
 commands=[json.loads(s)for s in(out/'robot_actions.jsonl').read_text().splitlines()]
 for f,c in zip(logs,commands):assert f['bounded_command']==c['normalized_command']
 row.update(condition=condition,offset_xyz_m=env.return02_offset.tolist(),force_samples=len(logs))
 dump(out/'return02-summary.json',row)
 return row
def main():
 global PHASE
 p=read(O/'protocol.json');parents=p['parents'];qualified=[];delta=None
 for parent in parents:
  limit();env=None
  try:
   env,rec,previous=qualified_prefix(parent,O/'qualification'/parent['id'],True)
   q=next(x for x in read(OLD/'qualification.json')['parents']if x['parent']==parent['id'])
   source,hashes=r.prepare(env,parent,O/'qualification'/parent['id']/'input','feeding',q['feeding_nontext'])
   before=state(env);screen=geometry(env,parent);assert state(env)==before
   dump(O/'qualification'/parent['id']/'geometry-screen.json',screen)
   if delta is None:delta=screen['delta_xyz_m']
   else:np.testing.assert_array_equal(delta,screen['delta_xyz_m'])
   qualified.append(screen)
  finally:
   if env is not None:env.close()
 limit();passed=all(x['passed']for x in qualified)
 dump(O/'qualification.json',{'passed':passed,'screens':qualified,'offset_xyz_m':delta,'epoch':time.time(),'no_inference':True,'fixed_candidate_not_swept':True})
 if not passed:
  dump(O/'physical-complete.json',{'episodes':0,'eligibility_failed':True,'condition':'2mm rejected by predeclared screen','epoch':time.time(),'training':False});return
 PHASE='execution';rows=[];skipped=[];failures=[]
 for parent in parents:
  nominal_ok=False
  q=next(x for x in read(OLD/'qualification.json')['parents']if x['parent']==parent['id'])
  for condition in ['E0','E2']:
   if condition=='E2'and not nominal_ok:
    skipped.append({'parent':parent['id'],'condition':condition,'reason':'E0 not trace-qualified'});continue
   limit();assert shutil.disk_usage('/var/tmp').free>400_000_000
   d=O/'episodes'/parent['id']/condition;env=None
   try:
    env,rec,previous=qualified_prefix(parent,d/'prefix',False)
    source,hashes=r.prepare(env,parent,d/'input','feeding',q['feeding_nontext'])
    prop=np.load(d/'input/initial-live.npz')['proprio'];row=run_teacher(env,parent,condition,d/'rollout',previous,prop,delta)
    if condition=='E0':
     original=[json.loads(x)for x in(OLD/'episodes'/parent['id']/'F-teacher/rollout/robot_actions.jsonl').read_text().splitlines()]
     now=[json.loads(x)for x in(d/'rollout/robot_actions.jsonl').read_text().splitlines()]
     exact=len(original)==len(now)and all(r.numeric(a,b)<1e-8 for a,b in zip(original,now))
     old_rich=read(OLD/'episodes'/parent['id']/'F-teacher/rollout/controller-rich.json');new_rich=read(d/'rollout/controller-rich.json')
     exact=bool(exact and len(old_rich)==len(new_rich)and all(r.numeric(a,b)<1e-8 for a,b in zip(old_rich,new_rich)))
     nominal_ok=exact
     dump(d/'nominal-replay-audit.json',{'passed':exact,'original_actions':len(original),'new_actions':len(now),'max_allowed_abs_error':1e-8,'no_outcome_retry':True})
    rows.append({'parent':parent['id'],'condition':condition,**row});dump(O/'episodes.json',rows)
    print(json.dumps({'parent':parent['id'],'condition':condition,'strict_success':row['strict_success'],'nominal_qualified':nominal_ok}),flush=True)
   except Exception:
    x={'parent':parent['id'],'condition':condition,'error':traceback.format_exc(),'epoch':time.time(),'no_retry':True};failures.append(x);dump(O/'technical-failures.json',failures);print(json.dumps(x),flush=True)
   finally:
    if env is not None:env.close()
 dump(O/'skipped.json',skipped);limit()
 dump(O/'physical-complete.json',{'episodes':len(rows),'technical_failures':len(failures),'skipped':len(skipped),'epoch':time.time(),'training':False,'model_inference':False,'outcome_retries':0})
if __name__=='__main__':
 try:main()
 except BaseException:
  dump(O/'STOPPED.json',{'phase':PHASE,'error':traceback.format_exc(),'epoch':time.time(),'clock_not_reset':True});raise
