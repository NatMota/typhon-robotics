"""LIVE06 fixed execution-contract diagnostic; unchanged V4 and expert."""
import sys,os,json,time,copy,subprocess,hashlib,collections
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));os.chdir(R)
import numpy as np
import live05_isolation as live
from supervisor.planning.execution_guard import ExecutionGuard,execution_violation
from supervisor.baseline.benchmark import run_episode,save
OUT=Path('/var/tmp/typhon-live06-execution-guard-20261012')
PREVIOUS=Path('/var/tmp/typhon-live05-isolation-20261012-v2')
EXPERT=Path('/var/tmp/typhon-research-20261012/loading/factorial/evaluation/expert')
base_guard_for=live.guard_for
def prepare():
 OUT.mkdir(exist_ok=False);prior=live.read(PREVIOUS/'protocol.json')
 expert=[]
 for p in sorted(EXPERT.glob('*/scenario.json')):
  assert live.read(p.parent/'episode.json')['success']
  expert.append({'source':str(p.parent),'scenario':live.read(p)})
 assert len(expert)==8
 protocol={'schema':'live06-v1','frozen_unix':time.time(),'checkpoint':prior['checkpoint'],'checkpoint_sha256':prior['checkpoint_sha256'],
  'scenes':prior['scenes'],'routes':[{'id':name,'noise':'continuous','wait_ticks':0,'guard':'shadow' if name=='native_reference' else 'enforce','candidates':1} for name in ['native_reference','original_guard','execution_guard']],
  'episodes':9,'expert_shadow_trials':expert,'invalid_command_tests':11,
  'intervention':'Only per-tick raw target-jump veto changes: evaluate the unchanged clipped OSC translation. Complete prospective chunks retain original admission, raw workspace/no-go ray and measured limits remain. No target modification, training or threshold change.',
  'scope':'Already admitted model chunks or independently verified expert primitives only. No physical deployment or full-arm future collision claim.',
  'reference_qualification':'Compare known-grasp raw inputs/noise and commands until first guard; retain known fresh-render repeatability limitation on stationary0002 fromLIVE05. No outcome tolerance changes.',
  'core_limits':prior['limits'],'code_sha256':{n:live.sha(R/n) for n in ['scripts/live06_execution_guard.py','supervisor/planning/execution_guard.py','tests/test_execution_guard.py','scripts/live05_isolation.py','scripts/live04_candidates_worker.py','supervisor/planning/m7_service.py']}}
 save(OUT/'protocol.json',protocol);(OUT/'code').mkdir()
 for n in protocol['code_sha256']:(OUT/'code'/Path(n).name).write_bytes((R/n).read_bytes())
 print(json.dumps({'prepared':True,'protocol_sha256':live.sha(OUT/'protocol.json'),'v4_trials':9,'expert_trials':8}),flush=True)
def expert():
 protocol=live.read(OUT/'protocol.json')
 for case in protocol['expert_shadow_trials']:
  scene=case['scenario'];folder=OUT/'expert-shadow'/scene['scenario_id'];events=[]
  def setup(env,folder):
   guard=base_guard_for(env);original=env.tick
   def tick(xyz,yaw,grip,steps=1):
    for _ in range(steps):
     obs=env.observe();rot=env.wrist_orientation(yaw);a=np.r_[xyz,rot[:,0],rot[:,1],float(grip>0),np.zeros(10)]
     old=guard.candidate([a],obs);new=execution_violation(guard,a,obs,env.config['control']['translation_step_m'])
     events.append({'step':env.steps,'old_reason':old,'new_reason':new,'target':a.tolist(),'eef_position_m':obs['proprio']['eef_position_m'].tolist(),
      'joint_position_rad':obs['proprio']['joint_position_rad'].tolist(),'joint_velocity_rad_s':obs['proprio']['joint_velocity_rad_s'].tolist()})
     original(xyz,yaw,grip,steps=1)
   env.tick=tick
   return {'scope':'shadow-only per-tick checks; expert commands unchanged; expert uses privileged truth by design'}
  ep=run_episode(scene,'expert',folder,episode_setup=setup)
  save(folder/'guard-events.json',events)
  result={'scenario_id':scene['scenario_id'],'strict_success':ep['success'],'runtime_errors':ep['controller_exceptions']+ep['artifact_errors']+ep['evaluation_errors'],
   'old_reasons':dict(collections.Counter(e['old_reason'] or 'admissible' for e in events)),
   'new_reasons':dict(collections.Counter(e['new_reason'] or 'admissible' for e in events)),
   'same_criteria_as_prior':ep['evaluation']['criteria']==live.read(Path(case['source'])/'episode.json')['evaluation']['criteria']}
  save(folder/'summary.json',result);print(json.dumps(result),flush=True)
  assert not result['runtime_errors']
 save(OUT/'expert-complete.json',{'completed':True,'episodes':8})
def run(shard):
 protocol=live.read(OUT/'protocol.json')
 for n,v in protocol['code_sha256'].items():assert live.sha(R/n)==v
 live.OUT=OUT;folder=OUT/f'worker-{shard}';folder.mkdir(exist_ok=False);policy=live.old.Policy(folder)
 save(folder/'model-load.json',policy.load_info)
 try:
  live.old.OUT=folder;live.old.qualify(policy)
  jobs=[(r,s) for r in protocol['routes'] for s in protocol['scenes']]
  for i,(route,scene) in enumerate(jobs):
   if i%2!=shard:continue
   def guard_for(env):
    old=base_guard_for(env)
    return ExecutionGuard(old.config,env.config['control']['translation_step_m']) if route['id']=='execution_guard' else old
   live.guard_for=guard_for
   live.episode(route,scene,policy)
 finally:policy.close()
 save(folder/'complete.json',{'completed':True})
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='expert':expert()
 else:run(int(sys.argv[2]))
