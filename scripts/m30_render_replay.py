"""M3.0 development images and exact action round-trip replay, without training."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import math
import numpy as np
from PIL import Image
from supervisor.feeding_loading.config import load_config
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.adaptation.actions import absolute_target, proprio_vector
from supervisor.baseline.xvla import to_action


def main(out, m16, canonical=False, fixtures_only=False):
    out.mkdir(parents=True, exist_ok=False)
    cfg = load_config('configs/feeding_loading/feeding_duplo_v2.json')
    if canonical:cfg['object'].update(width_m=.05,depth_m=.035,height_m=.03,mass_kg=.08,dimensions_status='M2_provisional_canonical_geometry')
    cfg['initial_state'].update(position_jitter_m=0., yaw_choices_rad=[0.])
    records = []
    env = FeedingLoadingEnv(cfg)
    try:
        env.reset(301)
        # Development fixtures span left start, middle transit, right target.
        for pi, xy in enumerate([[-.30, .08], [0., 0.], [.125, -.12]]):
            for angle in [0, 90, 180, 270]:
                q = math.radians(angle)/2
                env.env.sim.data.set_joint_qpos(env.env.cube.joints[0], [*xy, .8+cfg['object']['height_m']/2, math.cos(q), 0., 0., math.sin(q)])
                env.env.sim.data.set_joint_qvel(env.env.cube.joints[0], np.zeros(6)); env.env.sim.forward()
                name = f'position{pi}-yaw{angle:03d}'
                for view in ['primary', 'secondary']:
                    Image.fromarray(env.frame(view)).save(out/f'{name}-{view}.png')
                records.append({'id': name, 'known_fixture_yaw': angle, 'xy': xy, 'type': 'orientation_fixture'})
    finally:
        env.close()
    replay = []
    for demo in ([] if fixtures_only else ['personal_demo_0001', 'personal_demo_0007']):
        source = m16/'episode_artifacts'/demo
        if not source.exists():
            continue
        config = load_config(source/'resolved_config.json')
        result = json.loads((source/'episode.json').read_text())
        actions = [json.loads(x) for x in (source/'robot_actions.jsonl').read_text().splitlines()]
        env = FeedingLoadingEnv(config); errors=[]; state_errors=[]; samples=[]
        try:
            env.reset(result['seed'])
            for i, row in enumerate(actions):
                obs = env.observe()
                target = absolute_target(row['normalized_command'], obs, config)
                decoded = to_action(target, obs, config)
                errors.append(float(np.max(abs(decoded-row['normalized_command']))))
                if i % 20 == 0:
                    name=f'{demo}-step{env.steps:04d}'
                    for view in ['primary', 'secondary']:
                        Image.fromarray(obs['rgb'][view]).save(out/f'{name}-{view}.png')
                    records.append({'id':name, 'type':'expert_trajectory', 'step':env.steps, 'demo_id':demo})
                samples.append({'step':env.steps, 'absolute_target':target.tolist(), 'measured_proprio':proprio_vector(obs).tolist()})
                env.step(decoded)
                truth=env.truth()
                state_errors.append(max(float(np.max(abs(np.asarray(truth[k])-row['post_action_truth'][k])))
                    for k in ['object_position_m','object_quat_wxyz','eef_position_m']))
            replay.append({'demo_id':demo, 'steps':len(actions), 'max_command_error':max(errors),
                           'max_state_error':max(state_errors), 'original_success':result['success'],
                           'passed':max(errors)<1e-10 and max(state_errors)<1e-8})
            (out/f'{demo}-absolute-targets.json').write_text(json.dumps(samples)+'\n')
        finally:
            env.close()
    (out/'manifest.json').write_text(json.dumps({'records':records, 'replay':replay, 'fixture_config':cfg},indent=2)+'\n')
    print(json.dumps(replay,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True);p.add_argument('--m16',type=Path,required=True)
    p.add_argument('--canonical',action='store_true');p.add_argument('--fixtures-only',action='store_true')
    a=p.parse_args(); main(a.output,a.m16,a.canonical,a.fixtures_only)
