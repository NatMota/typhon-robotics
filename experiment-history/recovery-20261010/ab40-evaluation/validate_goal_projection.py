import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import copy,json,math,sys,time
from pathlib import Path
import imageio.v2 as imageio
sys.path.insert(0,str(Path.cwd()))
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.planning.simple_goal_classifier import GoalSuccessClassifier
from supervisor.planning.typhon_service import calibrated_goal_context
root=Path('/var/tmp/typhon-ab40-checker-preflight-20261010')
scenarios=json.loads(Path('frozen_results/personal-m3-20261009/protocol/scenarios.json').read_text())['primary']
classifier=GoalSuccessClassifier(); rows=[]
for task in ('feeding','loading'):
    source=next(s for s in scenarios if s['scenario_id']==task+'-standard-01')
    for goal_deg,offset,yaw_offset in [(45,0,0),(90,0,0),(45,0,180),(0,.12,0)]:
        cfg=copy.deepcopy(source['config']); yaw=math.radians(goal_deg)
        cfg['task'].update(target_yaw_rad=yaw,target_position_jitter_m=0.)
        target=cfg['task']['target_position_xy_m']
        obj_yaw=yaw+math.radians(yaw_offset)
        cfg['initial_state'].update(position_xy_m=[target[0]+offset,target[1]],position_jitter_m=0.,yaw_rad=obj_yaw,yaw_choices_rad=[obj_yaw])
        env=FeedingLoadingEnv(cfg)
        try:
            obs=env.reset(source['seed']); frame=obs['rgb']['primary']
            result=classifier(frame,calibrated_goal_context(cfg))
            name=f'{task}-goal{goal_deg}-offset{offset}-yawoffset{yaw_offset}'
            imageio.imwrite(root/(name+'.png'),frame)
            expected='complete' if offset==0 and yaw_offset==0 else 'incomplete'
            rows.append(dict(case=name,expected=expected,observed=result['status'],reason=result['reason'],passed=result['status']==expected))
            print(rows[-1],flush=True)
        finally:env.close()
(root/'goal-projection-validation.json').write_text(json.dumps(rows,indent=2)+'\n')
assert all(r['passed'] for r in rows),rows
