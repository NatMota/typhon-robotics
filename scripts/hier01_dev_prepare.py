"""Freeze the two-family development panel; reuse qualified task dispatcher."""
import os,sys,time,json,copy,pickle,hashlib,shutil,random,traceback
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
import numpy as np
from scipy.spatial.transform import Rotation
import hier01_task as h
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from load04_common import array_sha,endpoint
O=Path('/var/tmp/typhon-hier01-dev-20261015');OLD=Path('/var/tmp/typhon-hier01-task-20261015');L2=Path('/var/tmp/typhon-load02-20261013');L3=Path('/var/tmp/typhon-load03-20261013')
read,dump,sha=h.read,h.dump,h.sha
def prepare():
 clock=read(O/'clock.json');old=read(OLD/'protocol.json');q=read(OLD/'qualification.json');assert q['passed']
 assert not (O/'protocol.json').exists()
 hashes={}
 for rel,digest in old['source_sha256'].items():
  assert sha(R/rel)==digest,rel
  hashes[rel]=digest
 # Include the unchanged physical environment/controller tree, in addition to actor modules.
 baseline=Path('/var/tmp/typhon-worktrees-20261015/hier01-task')
 for p in (R/'supervisor').rglob('*.py'):
  rel=str(p.relative_to(R));assert sha(p)==sha(baseline/rel);hashes[rel]=sha(p)
 assert old['runtime']=={'O':'O2','domain':29,'H':30,'denoise_steps':10,'control_hz':20,'max_ticks':1500,'reset_ticks':40,'settle_ticks':24}
 for model,entry in old['models'].items():
  assert sha(entry['checkpoint'])==entry['sha256'];assert sha(entry['probe'])==entry['probe_sha256']
 for task,model in old['routing'].items():assert h.select_checkpoint(task)==model
 try:h.select_checkpoint('unsupported')
 except ValueError:pass
 else:raise AssertionError('Unsupported task accepted')
 p2=read(L2/'protocol.json');p3=read(L3/'protocol.json');cases={};order=[];receipts=[]
 for family in ['demo_0012','demo_0013']:
  for seed in [912171,912172]:
   assert time.time()<clock['preparation_deadline']
   source=next(c for c in p2['cases'] if c['source_demo_id']==family and c['seed']==seed and c['requested_task']=='feeding')
   fd=L2/'fixtures'/source['fixture_id'];z=np.load(fd/'input.npz');truth=read(fd/'truth.json');iso=read(fd/'isolation.json')
   src3=next(c for c in p3['cases'] if c['source_demo_id']==family and c['seed']==seed and c['requested_task']=='feeding' and c['visibility']=='visible')
   assert source['config']==src3['config']
   with np.load(L3/'fixtures'/src3['fixture_id']/'input.npz') as zz:
    assert z.files==zz.files
    for k in z.files:np.testing.assert_array_equal(z[k],zz[k])
   random.seed(seed);env=FeedingLoadingEnv(source['config'])
   try:
    obs=env.reset(seed);assert env.truth()==truth
    for k in ['qpos','qvel','mocap_pos','mocap_quat']:np.testing.assert_array_equal(getattr(env.env.sim.data,k),z[k])
    for k in ['primary','secondary']:np.testing.assert_array_equal(obs['rgb'][k],z[k])
    rotation=Rotation.from_quat(obs['proprio']['eef_quat_xyzw']).as_matrix()
    prop=np.r_[obs['proprio']['eef_position_m'],rotation[:,0],rotation[:,1],0.,np.zeros(10)].astype(np.float32)
    np.testing.assert_array_equal(prop,z['proprio'].astype(np.float32))
    state=pickle.dumps(h.prior.state(env),protocol=4)
   finally:env.close()
   for task in ['feeding','loading']:
    source=next(c for c in p2['cases'] if c['source_demo_id']==family and c['seed']==seed and c['requested_task']==task)
    context=f'{family}-{task}-{seed}';dest=O/('initial-'+context+'.npz');shutil.copy2(fd/'input.npz',dest)
    (O/('state-'+context+'.pkl')).write_bytes(state)
    prepared=fd/('prepared-'+task+'.npz')
    with np.load(prepared) as a:assert {k:array_sha(a[k]) for k in a.files}==iso['hashes'][task]
    assert source['instruction_override']==read(fd/'prompts.json')[task]
    cases[context]={'id':source['scenario_id'],'source_demo_id':family,'historical_fixture':str(fd),'historical_source_case':source,'config':source['config'],'simulation_seed':seed,'policy_seed':seed,'requested_task':task,'physical_scene_task':'loading','instruction':source['instruction_override'],'initial_archive':str(dest),'initial_archive_sha256':sha(dest),'initial_truth':truth,'input_hashes':iso['hashes'][task],'expected_initial_nontext':iso['expected_nontext'],'source_prepared_archive':str(prepared),'source_prepared_sha256':sha(prepared)}
    assert endpoint(source['config'],'feeding')['task']['type']=='feeding'
    assert endpoint(source['config'],'loading')['task']['type']=='loading'
   receipts.append({'family':family,'seed':seed,'LOAD02_LOAD03_visible_arrays_exact':True,'current_reset_matches_archive_exact':True,'full_state_sha256':hashlib.sha256(state).hexdigest()})
 for family in ['demo_0012','demo_0013']:
  for task in ['feeding','loading']:
   for seed in [912171,912172]:
    for route in ['single','router']:order.append({'context':f'{family}-{task}-{seed}','route':route})
 for rel in ['scripts/hier01_dev_prepare.py','scripts/hier01_dev.py']:hashes[rel]=sha(R/rel)
 protocol={'experiment':'HIER01-DEV','ticket':11067,'frozen_epoch':time.time(),'clock':clock,'models':old['models'],'routing':old['routing'],'control_model':'B','runtime':old['runtime'],'cases':cases,'execution_order':order,'source_sha256':hashes,'reuse_qualification':{'source':str(OLD/'qualification.json'),'source_sha256':sha(OLD/'qualification.json'),'original_protocol_sha256':sha(OLD/'protocol.json'),'executed_actor_dispatcher_physics_unchanged':True,'model_and_native_configuration_hashes_unchanged':True,'new_model_calls':0},'panel_status':'Already inspected development families seen in feeding training; loading combinations withheld from specified loading supervision. Not independent new-family generalisation.','data_boundary':'Zero new training, unchanged holdouts/quarantines; outputs may not enter training.','conditional_branch':{'all_eight_router_success_no_losses':'HIER01-CHAIN, separate45min','otherwise':'20min read-only first router failure by family/task/seed; no corrective run'}}
 assert len(cases)==8 and len(order)==16 and time.time()<clock['preparation_deadline']
 dump(O/'protocol.json',protocol);dump(O/'qualification.json',{'passed':True,'receipts':receipts,'dispatcher_reuse':protocol['reuse_qualification'],'fresh_model_calls':0,'physics_resets':4,'epoch':time.time()})
 print('DEV_QUALIFIED_16_FROZEN',flush=True)
if __name__=='__main__':
 try:prepare()
 except BaseException:dump(O/'STOPPED-prepare.json',{'error':traceback.format_exc(),'epoch':time.time()});raise
