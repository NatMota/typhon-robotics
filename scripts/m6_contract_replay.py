"""Qualify M6's exact model-output permutation through the existing OSC controller."""
import os,sys,json
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from supervisor.adaptation.m6_contract import reorder,self_check
from supervisor.adaptation.actions import absolute_target,robot_observation
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.baseline.xvla import to_action
from supervisor.baseline.benchmark import run_episode
out=Path('results/personal-m6-joint-20261009/contract-replay');out.mkdir(parents=True,exist_ok=False)
suite=json.loads(Path('frozen_results/personal-m3-20261009/protocol/scenarios.json').read_text());rows=[];errors=[];original_step=FeedingLoadingEnv.step

def step(env,command):
    obs=robot_observation(env)
    target=reorder(absolute_target(command,obs,env.config))
    decoded=to_action(reorder(target,True),obs,env.config)
    errors.append(float(np.max(np.abs(decoded-command))))
    return original_step(env,decoded)
FeedingLoadingEnv.step=step
try:
    for task in ['feeding','loading']:
        s=next(x for x in suite['primary'] if x['task']==task)
        row=run_episode(s,'expert',out/s['scenario_id'],provenance={'diagnostic':'M6 generic-interleaved action output decoded through frozen OSC adapter'})
        rows.append({'id':s['scenario_id'],'success':row['success'],'position_error_mm':row['position_error_mm'],'exceptions':row['controller_exceptions']});print(rows[-1],flush=True)
finally:FeedingLoadingEnv.step=original_step
report={'rotation_permutation':self_check(),'episodes':rows,'maximum_command_error':max(errors),'passed':max(errors)<1e-6 and all(r['success'] for r in rows)}
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');assert report['passed'];print(report,flush=True)
