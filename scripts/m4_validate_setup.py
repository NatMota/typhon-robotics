"""Validate frozen setup/interventions without a learned-policy outcome."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import imageio.v2 as imageio
import numpy as np
from supervisor.baseline.protocol import load_suite
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.adaptation.recovery import empty_grasp_setup,displacement_hook


def main(protocol,out):
    _,suite=load_suite(protocol);out.mkdir(parents=True,exist_ok=False);rows=[]
    for s in suite['primary']:
        folder=out/s['scenario_id'];folder.mkdir();env=FeedingLoadingEnv(s['config'])
        try:
            env.reset(s['seed']);initial=env.truth();imageio.imwrite(folder/'initial.png',env.frame())
            row={'id':s['scenario_id'],'category':s['group']}
            if s.get('setup'):
                setup=empty_grasp_setup(env,folder);row.update(prefix_contact=setup['prefix_contact'],prefix_steps=setup['prefix_steps'],handover_step=env.steps,
                    gripper_closed_after_prefix=not env.truth()['gripper_open'],last_command_finite=bool(np.isfinite(setup['last_command']).all()))
                assert not setup['prefix_contact'] and setup['prefix_steps']==98 and row['gripper_closed_after_prefix']
                imageio.imwrite(folder/'handover.png',env.frame())
            elif s.get('intervention'):
                for _ in range(120):env.step(np.r_[np.zeros(6),-1.])
                before=env.observe();hook=displacement_hook(s['intervention']);event=hook(env,4);after=env.observe()
                assert event and event['step']==160 and hook(env,4) is None
                assert np.allclose(event['after']['object_position_m'][:2],s['intervention']['position_xy_m'])
                row.update(step=event['step'],primary_changed=bool(np.any(before['rgb']['primary']!=after['rgb']['primary'])),secondary_changed=bool(np.any(before['rgb']['secondary']!=after['rgb']['secondary'])))
                assert row['primary_changed'] and row['secondary_changed']
                imageio.imwrite(folder/'before-relocation.png',before['rgb']['primary']);imageio.imwrite(folder/'after-relocation.png',after['rgb']['primary'])
            else:
                states=[]
                for i in range(24):
                    env.step(np.r_[np.zeros(6),-1.])
                    if i%2:states.append(env.truth())
                score=score_episode(initial,env.truth(),states,env.episode_config,[]);row['initial_goal_criteria']=score['criteria']
                if s['group']=='already_completed':assert score['success']
                elif s['group']=='wrong_orientation':assert not score['criteria']['orientation']
                elif s['group']=='misplacement':assert not score['criteria']['position']
            rows.append(row)
        finally:env.close()
    report={'passed':True,'scenarios_checked':len(rows),'no_learned_policy_outcomes_used':True,'checks':rows}
    (out/'validation_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.protocol,a.output)
