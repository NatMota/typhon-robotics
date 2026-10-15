"""INIT10: predeclared 5 mm vertical-target replay, no model calls or training."""
import os,sys,time,json,gzip,hashlib,traceback,copy,pickle
from pathlib import Path
W=Path(__file__).resolve().parents[1];os.chdir(W);sys.path[:0]=[str(W),str(W/'scripts')]
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np
import mujoco
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation
import init02_eval as ev
import init01 as initial
import hier01_task as h
import hier01_chain_amend as chain
from supervisor.baseline.benchmark import Recorder
from supervisor.baseline.xvla import to_action
O=Path('/var/tmp/typhon-init10-20261015');D=Path('/var/tmp/typhon-init07-20261015')
CASES=['init-demo_0003-loading-912172','init-demo_0005-loading-912171']
Q1='26d573eb90f791d9af86bbff30e138f51c5aa658ac7abefe76787ba352363a7a'
read,dump,sha=initial.read,initial.dump,initial.sha
def limit(stage='physical'):
 assert time.time()<read(O/'clock.json')[stage+'_deadline'],('INIT10 deadline',stage)
h.limit=lambda:limit();initial.limit=lambda *a:limit()
def traces(path):
 with gzip.open(path,'rt')as f:return [json.loads(x)for x in f]
def rot(a):
 x=np.asarray(a[3:6],float);x=x/np.linalg.norm(x);y=np.asarray(a[6:9],float);y=y-x*np.dot(x,y);y=y/np.linalg.norm(y)
 return np.column_stack([x,y,np.cross(x,y)])
def rawmodel(env):return env.env.sim.model._model,env.env.sim.data._data
def mapped(env):
 m,d=rawmodel(env);g=env.env.robots[0].gripper
 joints=[int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n))for n in g.joints]
 acts=[int(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n))for n in g.actuators]
 assert joints and acts and min(joints+acts)>=0
 assert all(int(m.actuator_trntype[a])==int(mujoco.mjtTrn.mjTRN_JOINT)and int(m.actuator_trnid[a,0])in joints for a in acts)
 assert m.na==0,'Activation mapping needs explicit qualification if nonempty'
 return {'joint_names':list(g.joints),'joint_ids':joints,'actuator_names':list(g.actuators),'actuator_ids':acts,'qpos_ids':[int(m.jnt_qposadr[j])for j in joints],'dof_ids':[int(m.jnt_dofadr[j])for j in joints],'activation_count':int(m.na),'actuator_trnid':m.actuator_trnid[acts].tolist(),'actuator_gear':m.actuator_gear[acts].tolist(),'mujoco_version':mujoco.__version__}
def forces(env,mapping):
 m,d=rawmodel(env);sim=env.env.sim;aid=mapping['actuator_ids'];qi=mapping['qpos_ids'];vi=mapping['dof_ids'];rows=[]
 for i,c in enumerate(d.contact[:d.ncon]):
  f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);frame=np.asarray(c.frame).reshape(3,3).copy()
  rows.append({'index':i,'geom1':sim.model.geom_id2name(c.geom1),'geom2':sim.model.geom_id2name(c.geom2),'geom_ids':[int(c.geom1),int(c.geom2)],'distance':float(c.dist),'position_world':c.pos.tolist(),'frame_rows_world':frame.tolist(),'wrench_contact':f.tolist(),'force_world_contact_basis':(frame.T@f[:3]).tolist(),'torque_world_contact_basis':(frame.T@f[3:]).tolist(),'dim':int(c.dim),'efc_address':int(c.efc_address),'friction':c.friction.tolist()})
 return {'ctrl':d.ctrl[aid].tolist(),'actuator_force':d.actuator_force[aid].tolist(),'activation':d.act.tolist(),'finger_generalized_actuator_force':d.qfrc_actuator[vi].tolist(),'finger_qpos':d.qpos[qi].tolist(),'finger_qvel':d.qvel[vi].tolist(),'contacts':rows,'sample_time_s':float(d.time),'timing':'Natural post-control-step derived arrays; no forward call.20Hz, not physics-substep peak data.','wrench_convention':'contact-frame force:torque retained; world vectors rotate components only, not an asserted object-side net force'}
def vertices(m,i):
 typ=int(m.geom_type[i]);size=np.array(m.geom_size[i])
 if typ==int(mujoco.mjtGeom.mjGEOM_MESH):
  mid=int(m.geom_dataid[i]);ad=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);return np.array(m.mesh_vert[ad:ad+n])
 if typ==int(mujoco.mjtGeom.mjGEOM_BOX):
  return np.array([[x,y,z]for x in [-1,1]for y in [-1,1]for z in [-1,1]])*size
 if typ==int(mujoco.mjtGeom.mjGEOM_CAPSULE):size=np.array([size[0],size[0],size[0]+size[1]])
 elif typ==int(mujoco.mjtGeom.mjGEOM_CYLINDER):size=np.array([size[0],size[0],size[1]])
 elif typ==int(mujoco.mjtGeom.mjGEOM_SPHERE):size=np.full(3,size[0])
 else:raise ValueError(('Unsupported collision screen geometry',i,typ))
 return np.array([[x,y,z]for x in [-1,1]for y in [-1,1]for z in [-1,1]])*size
def screen(env,targets):
 m,d=rawmodel(env);sim=env.env.sim;rob=env.env.robots[0];G=np.array(d.site_xmat[rob.eef_site_id]).reshape(3,3);g=np.array(d.site_xpos[rob.eef_site_id])
 gr=[];obstacles=[];geom=[]
 for i in range(m.ngeom):
  name=sim.model.geom_id2name(i)or''
  if not(m.geom_contype[i]or m.geom_conaffinity[i]):continue
  if not(name.startswith('gripper')or'table'in name or'fixture'in name):continue
  R=np.array(d.geom_xmat[i]).reshape(3,3);v=vertices(m,i)@R.T+d.geom_xpos[i]
  geom.append({'id':i,'name':name,'type':int(m.geom_type[i]),'size':m.geom_size[i].tolist(),'world_min':v.min(0).tolist(),'world_max':v.max(0).tolist()})
  if name.startswith('gripper'):gr.append((name,(v-g)@G))
  else:obstacles.append((name,v.min(0),v.max(0)))
 assert gr and any('table'in n for n,_,_ in obstacles)and any('fixture'in n for n,_,_ in obstacles)
 collisions=[];table_clearance=[]
 for k,a in enumerate(targets):
  for name,v in gr:
   z=v@rot(a).T+a[:3];lo=z.min(0);hi=z.max(0)
   for on,ol,oh in obstacles:
    if np.all(lo<=oh)and np.all(hi>=ol):collisions.append({'target':k,'gripper':name,'obstacle':on})
    if'table'in on:table_clearance.append(float(lo[2]-oh[2]))
 return {'passed':not collisions,'method':'Discrete conservative collision-geometry AABB screen at fixed branch aperture; meshes use compiled vertices. Actual dynamics and closing aperture may differ. Runtime contact abort remains active.','geometry':geom,'intersections':collisions,'minimum_gripper_table_vertical_clearance_m':min(table_clearance)}
def make(case,condition):
 limit();out=O/'branches'/case['id']/condition;out.mkdir(parents=True,exist_ok=False);folder=D/'episodes/Q1'/case['id'];old=traces(folder/'controller_trace.jsonl.gz');ep=read(folder/'episode.json')
 assert ep['model_sha256']==Q1
 z=np.load(folder/'predictions.npz');targets=z['chunks'].reshape(-1,20)[:120].copy();assert targets.shape==(120,20)
 env,previous=ev.initialise(case,out/'prefix')
 assert chain.state_blob(env)==(folder/'boundary-before.pkl').read_bytes(),'q0 semantic state mismatch'
 rec=Recorder(env,out/'trajectory.jsonl.gz')
 writer=imageio.get_writer(out/'rollout.mp4',fps=5)
 def capture(scene):writer.append_data(np.concatenate([scene.frame('primary'),scene.frame('secondary')],axis=1))
 capture(env);env.capture_callback=capture;imageio.imwrite(out/'initial.png',np.concatenate([env.frame(),env.frame('secondary')],axis=1))
 diffs=[]
 for i in range(60):
  limit()
  if i%30==0:env.observe()
  action=to_action(targets[i],env.observe(),case['config']);env.step(action);cur=h.base.native.b.rich(env,targets[i],action)
  error=initial.numeric(cur,old[i]);diffs.append(error);assert error<1e-8,('prefix divergence',case['id'],i,error)
 blob=chain.state_blob(env);(out/'branch-state.pkl').write_bytes(blob)
 ref=O/'branches'/case['id']/'N'/'branch-state.pkl'
 if condition=='D5':assert blob==ref.read_bytes(),'Paired branch state mismatch'
 assert env.steps==old[59]['step']
 dump(out/'prefix-validation.json',{'passed':True,'maximum_rich_error':max(diffs),'q0_semantic_exact':True,'branch_tick':env.steps,'paired_semantic_exact':condition=='D5','branch_sha256':hashlib.sha256(blob).hexdigest(),'epoch':time.time()})
 return dict(env=env,rec=rec,writer=writer,capture=capture,out=out,case=case,targets=targets,old=old,z0=ep['initial_truth']['object_position_m'][2])
def finish(ctx):
 env=ctx['env'];ctx['rec'].finish();ctx['capture'](env);ctx['writer'].close();env.capture_callback=None
 imageio.imwrite(ctx['out']/'final.png',np.concatenate([env.frame(),env.frame('secondary')],axis=1));env.close()
def execute(ctx,condition):
 env=ctx['env'];out=ctx['out'];mapping=mapped(env);rows=[];error=[];abort=None
 try:
  for i in range(60,120):
   limit()
   if i%30==0:env.observe()
   original=ctx['targets'][i];target=original.copy()if condition=='N'else original.astype(np.float64)
   if condition=='D5':target[2]-=.005
   action=to_action(target,env.observe(),ctx['case']['config']);env.step(action)
   row=h.base.native.b.rich(env,target,action)
   if condition=='N':
    err=initial.numeric(row,ctx['old'][i]);error.append(err)
    if err>=1e-8:dump(out/'first-divergence.json',{'index':i,'error':err,'new':row,'archived':ctx['old'][i]});raise AssertionError(('native divergence',i,err))
   before=chain.state_blob(env);f=forces(env,mapping);assert before==chain.state_blob(env),'Force logger changes semantic state'
   row.update(original_target=original.tolist(),applied_target=target.tolist(),force_telemetry=f);rows.append(row)
   bad=[c for c in f['contacts']if('gripper'in c['geom1']and any(s in c['geom2']for s in ['table','fixture']))or('gripper'in c['geom2']and any(s in c['geom1']for s in ['table','fixture']))]
   if condition=='D5'and bad:abort={'reason':'forbidden gripper table/fixture contact','tick':env.steps,'contacts':bad};break
  dump(out/'actuator-map.json',mapping)
  flags=[r['object_truth']['bilateral_grasp_contact']and r['object_truth']['object_position_m'][2]-ctx['z0']>=.01 for r in rows]
  strong=[r['object_truth']['bilateral_grasp_contact']and r['object_truth']['object_position_m'][2]-ctx['z0']>=.07 and r['object_tilt_deg']<=10 for r in rows]
  consecutive=lambda a:any(all(a[i:i+3])for i in range(len(a)-2))
  end=rows[-1];held=end['object_truth']['bilateral_grasp_contact']and end['object_tilt_deg']<=10
  summary={'case':ctx['case']['id'],'condition':condition,'steps':len(rows),'native_replay_passed':len(rows)==60 and max(error,default=0)<1e-8 if condition=='N'else None,'native_max_error':max(error,default=0)if condition=='N'else None,'supported_10mm_3samples':bool(consecutive(flags)),'upright_supported_70mm_3samples':bool(consecutive(strong)),'end_upright_bilateral':bool(held),'retention_gate':bool(consecutive(strong)and held and not abort and len(rows)==60),'maximum_brick_rise_mm':max((r['object_truth']['object_position_m'][2]-ctx['z0'])*1000 for r in rows),'abort':abort,'full_task_success':None,'new_model_calls':0,'new_training':0,'epoch':time.time()}
  dump(out/'summary.json',summary);return summary
 finally:
  with gzip.open(out/'controller-force-trace.jsonl.gz','wt')as f:
   for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
  finish(ctx)
def main():
 limit('feasibility');cases=[next(c for c in read(D/'evaluation-cases.json')['core']if c['id']==name)for name in CASES]
 manifest={'clock':read(O/'clock.json'),'model_sha256':Q1,'cases':cases,'active_z_offset_m':-.005,'segment_indices':[60,120],'native_tolerance':1e-8,'new_model_calls':0,'training':False,'diagnostic_quarantine':True,'script_sha256':sha(Path(__file__)),'source_sha256':{str(D/'episodes/Q1'/c['id']/n):sha(D/'episodes/Q1'/c['id']/n)for c in cases for n in ['episode.json','predictions.npz','controller_trace.jsonl.gz','boundary-before.pkl']}}
 dump(O/'execution-freeze.json',manifest)
 contexts=[];gates=[]
 try:
  for case in cases:
   ctx=make(case,'N');contexts.append(ctx);targets=ctx['targets'][60:120].astype(float);shift=targets.copy();shift[:,2]-=.005
   np.testing.assert_array_equal(shift[:,[j for j in range(20)if j!=2]],targets[:,[j for j in range(20)if j!=2]])
   np.testing.assert_allclose(shift[:,2]-targets[:,2],-.005,rtol=0,atol=1e-15)
   bounds=np.array(case['config']['control']['workspace_bounds_m']);valid=bool(np.all(shift[:,:3]>=bounds[:,0])and np.all(shift[:,:3]<=bounds[:,1]))
   # Segment ends immediately before original opening, but includes pre-close open targets as recorded.
   assert ctx['old'][119]['step']+1 in [464,457] and ctx['old'][120]['target'][9]<=.5 and ctx['old'][119]['target'][9]>.5
   geometry=screen(ctx['env'],shift);mapping=mapped(ctx['env']);before=chain.state_blob(ctx['env']);sample=forces(ctx['env'],mapping);assert before==chain.state_blob(ctx['env'])
   gate={'case':case['id'],'targets_in_bounds':valid,'geometry':geometry,'actuators':mapping,'read_only_force_sample':sample,'passed':bool(valid and geometry['passed']),'epoch':time.time()}
   dump(ctx['out']/'feasibility.json',gate);gates.append(gate)
   assert gate['passed'],'Fixed5mm intervention failed screen; no alternative depth'
  limit('feasibility');dump(O/'feasibility-gate.json',{'passed':True,'cases':gates,'epoch':time.time()})
  results=[]
  for ctx in contexts:
   s=execute(ctx,'N');ctx['closed']=True;results.append(s);dump(O/'results.json',results)
   assert s['native_replay_passed'],'Native replay failed; no causal comparison'
   altered=make(ctx['case'],'D5')
   s=execute(altered,'D5');results.append(s);dump(O/'results.json',results)
  for path,hsh in manifest['source_sha256'].items():assert sha(Path(path))==hsh
  dump(O/'physical-complete.json',{'passed':True,'branches':len(results),'epoch':time.time(),'source_unchanged':True})
 except BaseException:
  for ctx in contexts:
   if not ctx.get('closed'):
    try:finish(ctx);ctx['closed']=True
    except Exception:pass
  raise
if __name__=='__main__':
 try:main()
 except BaseException:dump(O/'STOPPED.json',{'epoch':time.time(),'error':traceback.format_exc()});raise
