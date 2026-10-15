"""HIER01-START: native scene construction and explicit diagnostic initialization."""
import os,sys,json,time,copy,random,pickle,traceback,hashlib,subprocess,shutil
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
import numpy as np
from scipy.spatial.transform import Rotation
import imageio.v2 as imageio
import hier01_task as h
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from load04_common import array_sha
O=Path('/var/tmp/typhon-hier01-start-20261015');h.O=O
read,dump,sha=h.read,h.dump,h.sha
def initialize_object(env,case):
 if case['cell']=='REF':return {'object_intervention':False}
 d=env.env.sim.data;robot=env.env.robots[0];before=h.prior.state(env)
 pose=case['diagnostic_object_pose_wxyz']
 d.set_joint_qpos(env.env.cube.joints[0],np.asarray(pose));d.set_joint_qvel(env.env.cube.joints[0],np.zeros(6));env.env.sim.forward()
 after=h.prior.state(env)
 for field in ['qpos','qvel']:
  idx=robot._ref_joint_pos_indexes if field=='qpos' else robot._ref_joint_vel_indexes
  np.testing.assert_array_equal(before[field][idx],after[field][idx])
 np.testing.assert_array_equal(before['qpos'][robot._ref_gripper_joint_pos_indexes],after['qpos'][robot._ref_gripper_joint_pos_indexes])
 assert pickle.dumps(before['controller'],protocol=4)==pickle.dumps(after['controller'],protocol=4)
 assert before['time']==after['time'] and env.steps==40
 np.testing.assert_allclose(env.truth()['object_position_m'],pose[:3],atol=1e-14,rtol=0)
 np.testing.assert_allclose(env.truth()['object_quat_wxyz'],pose[3:],atol=1e-14,rtol=0)
 return {'object_intervention':True,'when':'after ordinary standalone reset, before first observation/action','robot_and_controller_unchanged':True,'ticks_added':0,'object_pose_wxyz':pose,'object_qvel_zero':True,'no_live_handover_restoration':True}
def archive(env,dest):
 obs=env.observe();rot=Rotation.from_quat(obs['proprio']['eef_quat_xyzw']).as_matrix()
 prop=np.r_[obs['proprio']['eef_position_m'],rot[:,0],rot[:,1],0.,np.zeros(10)].astype(np.float32);d=env.env.sim.data
 np.savez_compressed(dest,primary=obs['rgb']['primary'],secondary=obs['rgb']['secondary'],proprio=prop,qpos=d.qpos.copy(),qvel=d.qvel.copy(),mocap_pos=d.mocap_pos.copy(),mocap_quat=d.mocap_quat.copy())
 return obs
def geometry_receipt(env,case):
 m=env.env.sim.model;d=env.env.sim.data;ws=case['config']['workspace'];truth=env.truth()
 stage=m.body_name2id('m1_stage');ids=np.where(m.geom_bodyid==stage)[0]
 assert len(ids)==5 and np.all(m.geom_contype[ids]==0) and np.all(m.geom_conaffinity[ids]==0)
 # Table is axis aligned; keep full rotated block footprint within its edges.
 half=np.array([case['config']['object'][k]/2 for k in ['width_m','depth_m','height_m']])
 q=np.array(truth['object_quat_wxyz']);rot=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
 corners=np.array([[x,y,z] for x in [-half[0],half[0]] for y in [-half[1],half[1]] for z in [-half[2],half[2]]])@rot.T+truth['object_position_m']
 assert np.max(abs(corners[:,0]))<ws['table_width_m']/2 and np.max(abs(corners[:,1]))<ws['table_depth_m']/2
 center=np.array(ws['staging_position_xy_m']);sh=ws['staging_size_m']/2
 assert abs(center[0])+sh+.002<ws['table_width_m']/2 and abs(center[1])+sh+.01<ws['table_depth_m']/2
 contacts=[(m.geom_id2name(c.geom1),m.geom_id2name(c.geom2),float(c.dist)) for c in d.contact[:d.ncon]]
 objectids=set(env.env.cube.contact_geoms)
 objectcontacts=[c for c in contacts if c[0] in objectids or c[1] in objectids]
 assert all(('robot' not in a and 'gripper' not in a and 'fixture' not in a and 'robot' not in b and 'gripper' not in b and 'fixture' not in b) for a,b,_ in objectcontacts),objectcontacts
 overlap=bool(np.all(corners[:,0]>=center[0]-sh) and np.all(corners[:,0]<=center[0]+sh) and np.all(corners[:,1]>=center[1]-sh) and np.all(corners[:,1]<=center[1]+sh))
 disjoint=bool(corners[:,0].min()>center[0]+sh or corners[:,0].max()<center[0]-sh or corners[:,1].min()>center[1]+sh or corners[:,1].max()<center[1]-sh)
 if case['cell']!='REF':
  assert overlap==(case['cell'][0]==case['cell'][1])
  assert disjoint==(case['cell'][0]!=case['cell'][1])
 return {'stage_geoms_noncolliding':True,'object_contacts':objectcontacts,'within_table':True,'object_inside_marker':overlap,'object_disjoint_marker':disjoint,'stage_position_m':d.mocap_pos[m.body_mocapid[stage]].tolist(),'fixture_position_m':d.mocap_pos[m.body_mocapid[m.body_name2id('m1_fixture')]].tolist(),'object_corners_m':corners.tolist()}
def prepare():
 clock=read(O/'clock.json');old=read('/var/tmp/typhon-hier01-dev-20261015/protocol.json');assert time.time()<clock['preparation_deadline']
 hashes={k:sha(R/k) for k in old['source_sha256']};assert hashes==old['source_sha256']
 model=old['models']['FL'];assert sha(model['checkpoint'])==model['sha256']
 truth=read('/var/tmp/typhon-hier01-chain-20261015/workers/F-L-stage2-loading/episode/episode.json')['initial_truth']
 pose=np.r_[truth['object_position_m'],truth['object_quat_wxyz']];delta=np.array([0.,-.12,0.])
 cases={};receipts=[];base_models={}
 for seed in [912171,912172]:
  src=old['cases'][f'demo_0012-loading-{seed}'];standard=pickle.load(open('/var/tmp/typhon-hier01-dev-20261015/state-demo_0012-loading-'+str(seed)+'.pkl','rb'))
  for cell in ['REF','AA','AB','BA','BB']:
   assert time.time()<clock['preparation_deadline'];c=copy.deepcopy(src);c.update(cell=cell,expected_initial_nontext={});c['id']=cell+'-'+str(seed)
   c['diagnostic_object_pose_wxyz']=None if cell=='REF' else (pose+np.r_[delta if cell[0]=='B' else np.zeros(3),np.zeros(4)]).tolist()
   if cell!='REF' and cell[1]=='B':c['config']['workspace']['staging_position_xy_m']=(np.array(src['config']['workspace']['staging_position_xy_m'])+delta[:2]).tolist()
   conf=copy.deepcopy(c['config']);conf['workspace']['staging_position_xy_m']=src['config']['workspace']['staging_position_xy_m'];assert conf==src['config']
   # Object initialization remains the original reset until the explicit pre-observation diagnostic setter.
   random.seed(seed);env=FeedingLoadingEnv(c['config'])
   try:
    env.reset(seed);state=h.prior.state(env);robot=env.env.robots[0]
    for k in ['qpos','qvel']:np.testing.assert_array_equal(state[k],standard[k])
    assert pickle.dumps(state['controller'],protocol=4)==pickle.dumps(standard['controller'],protocol=4)
    # All compiled geom properties are unchanged; only stage mocap/body placement changes.
    m=env.env.sim.model
    props={k:np.array(getattr(m,k)).copy() for k in ['geom_type','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity','geom_friction','geom_rgba','body_mass','body_inertia']}
    if seed not in base_models:base_models[seed]=props
    else:
     for k,v in props.items():np.testing.assert_array_equal(v,base_models[seed][k])
    init=initialize_object(env,c);rec=geometry_receipt(env,c)
    dest=O/'fixtures'/c['id'];dest.mkdir(parents=True)
    obs=archive(env,dest/'initial.npz');c['initial_archive']=str(dest/'initial.npz');c['initial_archive_sha256']=sha(dest/'initial.npz');c['initial_truth']=env.truth()
    c['source_prepared_archive']=str(dest/'prepared.npz')
    (dest/'state.pkl').write_bytes(pickle.dumps(h.prior.state(env),protocol=4));imageio.imwrite(dest/'initial.png',np.concatenate([obs['rgb']['primary'],obs['rgb']['secondary']],axis=1))
    dump(dest/'scene-receipt.json',{'cell':cell,'seed':seed,'native_scene_construction':True,'only_config_change':'workspace.staging_position_xy_m' if cell.endswith('B') else None,'geometry_properties_unchanged':True,'robot_reset_matches_standalone_exact':True,'initialization':init,**rec})
    receipts.append({'id':c['id'],**rec});cases[c['id']]=c
   finally:env.close()
 for seed in [912171,912172]:
  for a,b in [('AA','AB'),('BA','BB')]:
   x=cases[a+'-'+str(seed)];y=cases[b+'-'+str(seed)];assert x['initial_truth']==y['initial_truth']
   with np.load(x['initial_archive']) as ax,np.load(y['initial_archive']) as ay:
    for k in ['qpos','qvel','proprio']:np.testing.assert_array_equal(ax[k],ay[k])
 for rel in ['scripts/hier01_start.py','scripts/hier01_start_prepare_inputs.py']:hashes[rel]=sha(R/rel)
 p={'experiment':'HIER01-START','ticket':11071,'clock':clock,'models':{'FL':model},'runtime':old['runtime'],'cases':cases,'source_sha256':hashes,'source_handover_truth':truth,'order':[f'REF-{s}' for s in [912171,912172]]+[f'{c}-{s}' for c in ['AA','AB','BA','BB'] for s in [912171,912172]],'marker_delta_world_m':delta.tolist(),'actor_boundary':'Only current RGB/native proprio/full unchanged loading instruction; no diagnostic label or object truth','data_boundary':'All new starts are quarantined development diagnostics; no training, zero EVAL02 intervention trials','initialization':'Ordinary standalone robot/controller reset in every cell, then explicit object-only initial pose setter, zero added steps; not continuous-chain repair'}
 dump(O/'protocol-draft.json',p);dump(O/'scene-qualification.json',{'passed':True,'resets':10,'policy_calls':0,'receipts':receipts,'epoch':time.time()});print('SCENES_QUALIFIED',flush=True)
class Policy(h.Policy):
 def instrument(self,env,out):
  info=initialize_object(env,self.case);meta=super().instrument(env,out)
  return {**meta,'native_scene_unchanged':self.case['cell']=='REF','training_initial_verified':False,'development_initial_verified':True,'explicit_object_initialization':info,'marker_from_native_config':True}
ACTIVE_DEADLINE=0.
def phase_limit():
 assert time.time()<ACTIVE_DEADLINE,'Frozen phase deadline reached'
def run():
 global ACTIVE_DEADLINE
 h.limit=phase_limit
 p=read(O/'protocol.json');assert read(O/'qualification.json')['passed']
 for k,v in p['source_sha256'].items():assert sha(R/k)==v,k
 assert time.time()<p['clock']['controls_deadline'];out=O/'worker';out.mkdir();h.base.OUT=out
 w=h.base.Worker({'model':p['models']['FL'],'expected_initial_nontext':{}});rows=[];refs={}
 try:
  for ix,tag in enumerate(p['order']):
   deadline=p['clock']['controls_deadline'] if ix<2 else p['clock']['execution_deadline'];ACTIVE_DEADLINE=deadline;assert time.time()<deadline
   assert shutil.disk_usage(O).free>512*1024**2,'Less than512MiB output headroom'
   if ix==2:
    assert time.time()<p['clock']['controls_deadline'],'Control gate missed30minute limit'
    gate=all(x['requested_success'] and x['supported_lift_10mm_3steps'] for x in rows)
    dump(O/'control-gate.json',{'passed':gate,'rows':rows,'epoch':time.time()})
    if not gate:break
   case=copy.deepcopy(p['cases'][tag]);dest=O/'episodes'/tag;w.protocol['expected_initial_nontext']=case['expected_initial_nontext'];random.seed(case['simulation_seed'])
   policy=Policy(w,case,dest,case['policy_seed'],tag)
   scenario={'scenario_id':'hier01-start-'+tag,'source_demo_id':'demo_0012','group':'quarantined-start-context-development','task':'loading','seed':case['simulation_seed'],'config':copy.deepcopy(case['config'])}
   try:
    ep=h.run_episode(scenario,'pretrained',dest,policy,{'experiment':'HIER01-START','cell':case['cell'],'model_sha256':p['models']['FL']['sha256'],'policy_seed':case['policy_seed'],'new_training':False,'continuous_workflow':False,'harness':False,'TTC':False,'protocol_sha256':sha(O/'protocol.json')},episode_setup=policy.instrument)
   finally:
    if hasattr(policy,'telemetry'):policy.telemetry.close()
   assert not any(ep[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors']),{k:ep[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors']}
   assert len(policy.chunks)==48 and len(policy.records)==1460
   seed=case['policy_seed'];noise=h.base.noise_gate(dest,seed,refs.get(seed));refs.setdefault(seed,dest)
   with np.load(dest/'input-audit/initial-prepared.npz') as a,np.load(case['source_prepared_archive']) as b:
    assert a.files==b.files
    for k in a.files:np.testing.assert_array_equal(a[k],b[k])
   s=ep['evaluation'];row={'id':tag,'cell':case['cell'],'seed':seed,'directory':str(dest),'requested_success':s['success'],'position_error_mm':s['position_error_m']*1000,'yaw_error_deg':s['orientation_error_deg'],'criteria':s['criteria'],'noise':noise,'queries':len(policy.chunks),'wall_seconds':ep['wall_duration_s'],**h.supplemental(ep,dest)}
   rows.append(row);dump(O/'summary.json',rows);print(json.dumps({k:row[k] for k in ['id','requested_success','position_error_mm','supported_lift_10mm_3steps']}),flush=True)
 finally:w.close()
 dump(O/'complete.json',{'episodes':len(rows),'complete':len(rows)==10,'control_gate_passed':len(rows)>2,'epoch':time.time(),'training':False})
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  else:run()
 except BaseException:dump(O/('STOPPED-'+sys.argv[1]+'.json'),{'epoch':time.time(),'error':traceback.format_exc()});raise
