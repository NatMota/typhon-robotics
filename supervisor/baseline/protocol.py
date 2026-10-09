"""Freeze scenarios before observing manipulation outcomes. No outcome-based replacement."""
import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from supervisor.feeding_loading.config import load_config, validate_config

VERSION = 'm2-primary-v1'
M1_COMMIT = 'dce6ffbd03ac4c1c689b12962b5e9c4fe1b0f6ec'
ROOT = Path('results/personal-m2-20261009')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def environment_hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path('supervisor/feeding_loading').glob('*.py'))}


def validate_layout(config):
    validate_config(config)
    c = config
    source = np.array(c['initial_state']['position_xy_m'])
    target = np.array(c['task']['target_position_xy_m'])
    if c['task']['type'] == 'loading':
        # Every yaw's full footprint stays on the staging square before pickup.
        radius = math.hypot(c['object']['width_m'], c['object']['depth_m']) / 2
        assert np.max(np.abs(source - c['workspace']['staging_position_xy_m'])) + radius < c['workspace']['staging_size_m'] / 2
    assert np.linalg.norm(source-target) > .09, 'Primary source and destination must be distinct'
    # Analytic reach envelope deliberately stays near M1's validated corridor.
    assert -.11 <= source[0] <= .15 and -.24 <= source[1] <= -.10
    assert -.10 <= target[0] <= .16 and -.20 <= target[1] <= .19
    grasp_z = c['workspace']['table_z_m'] + c['object']['height_m']/2 + .005
    max_z = c['workspace']['table_z_m'] + c['object']['height_m']/2 + .16 + c['workspace']['fixture_floor_height_m']
    assert c['control']['workspace_bounds_m'][2][0] <= grasp_z < max_z <= c['control']['workspace_bounds_m'][2][1]
    return True


def scenario_record(identifier, group, seed, config):
    c = validate_config(config)
    return {'scenario_id': identifier, 'group': group, 'task': c['task']['type'], 'seed': seed,
            'config': c, 'initial_pose': {'position_xy_m': c['initial_state']['position_xy_m'], 'yaw_rad': c['initial_state']['yaw_rad'], 'marked_face': 'top'},
            'target_pose': {'position_xy_m': c['task']['target_position_xy_m'], 'yaw_rad': c['task']['target_yaw_rad']},
            'geometry_version': 'm1-orange-box-50x35x30mm-provisional-v1',
            'workspace': copy.deepcopy(c['workspace']), 'control_horizon_total_steps': 1500,
            'reset_steps': 40, 'maximum_execution_steps_including_settling': 1460,
            'success_tolerances': copy.deepcopy(c['evaluation']),
            'simulator': {'mujoco': '2.3.7', 'robosuite': '1.4.1', 'robot': 'Panda', 'environment_commit': M1_COMMIT},
            'controller': {'type': 'OSC_POSE', **copy.deepcopy(c['control'])}}


def build_suite():
    primary, rejected = [], []
    cardinal = [0, 90, 180, 270, 0, 90, 180, 270, 0, 90]
    varied = [30, 45, 60, 120, 135, 150, 210, 225, 300, 330]
    target_yaws = [0, 45, -45, 0, 90, 0, 45, -45, 0, 90]
    for task_index, task in enumerate(('feeding', 'loading')):
        for group_index, group in enumerate(('standard', 'varied')):
            for i in range(10):
                seed = 220000 + task_index*1000 + group_index*100 + i
                # Only analytic-layout rejection may advance the deterministic candidate seed.
                for candidate in range(100):
                    rng = np.random.default_rng(seed + candidate*100000)
                    c = load_config(task=task)
                    spread = .008 if group == 'standard' else (.030 if task == 'feeding' else .014)
                    c['initial_state']['position_xy_m'] = (np.array(c['initial_state']['position_xy_m']) + rng.uniform(-spread, spread, 2)).tolist()
                    yaw = math.radians((cardinal if group == 'standard' else varied)[i])
                    c['initial_state'].update(yaw_rad=yaw, yaw_choices_rad=[yaw], position_jitter_m=0.)
                    c['task']['target_position_jitter_m'] = 0.
                    if group == 'varied':
                        c['task']['target_position_xy_m'] = (np.array(c['task']['target_position_xy_m']) + rng.uniform(-.015, .015, 2)).tolist()
                        c['task']['target_yaw_rad'] = math.radians(target_yaws[i])
                    key = 'staging_position_xy_m' if task == 'feeding' else 'fixture_position_xy_m'
                    c['workspace'][key] = c['task']['target_position_xy_m'].copy()
                    try:
                        validate_layout(c)
                    except (ValueError, AssertionError) as exc:
                        rejected.append({'scenario_seed':seed,'candidate_index':candidate,'candidate_config':c,'reason':str(exc),'stage':'analytic_before_freeze_no_manipulation'})
                        continue
                    record = scenario_record(f'{task}-{group}-{i+1:02d}', group, seed, c)
                    record['candidate_index'] = candidate
                    primary.append(record)
                    break
                else:
                    raise RuntimeError('Cannot generate a valid analytic layout')
    diagnostics = []
    for i, mode in enumerate(('failed_initial_grasp','displaced_object','incorrect_orientation','already_inside_target','retry_after_unsuccessful_manipulation')):
        c = copy.deepcopy(primary[1]['config'])
        if mode == 'displaced_object':
            c['initial_state']['position_xy_m'] = [.13, -.12]
        if mode == 'incorrect_orientation':
            c['initial_state']['position_xy_m'] = c['task']['target_position_xy_m'].copy()
            c['initial_state'].update(yaw_rad=math.pi,yaw_choices_rad=[math.pi])
        if mode == 'already_inside_target':
            c['initial_state']['position_xy_m'] = c['task']['target_position_xy_m'].copy()
            c['initial_state'].update(yaw_rad=c['task']['target_yaw_rad'],yaw_choices_rad=[c['task']['target_yaw_rad']])
        r = scenario_record(f'diagnostic-{mode}', 'recovery_diagnostic',230000+i,c)
        r['diagnostic'] = mode
        r['force_first_grasp_miss'] = mode in ('failed_initial_grasp','retry_after_unsuccessful_manipulation')
        r['attempt_limit'] = 1 if mode == 'failed_initial_grasp' else 2
        r['initial_state_intervention'] = 'reset-only displacement/orientation; no during-action truth injection'
        diagnostics.append(r)
    return {'schema_version':1,'protocol_version':VERSION,'primary':primary,'diagnostics':diagnostics,'rejected_candidates':rejected}


def freeze(output=ROOT):
    output = Path(output)
    if (output/'protocol.json').exists() or (output/'scenarios.json').exists():
        raise FileExistsError('Protocol artifacts already exist; do not refreeze a benchmark after outcomes')
    output.mkdir(parents=True,exist_ok=True)
    suite = build_suite()
    protocol = {'schema_version':1,'protocol_version':VERSION,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'environment_commit':M1_COMMIT,'environment_source_sha256':environment_hashes(),
        'scenario_sha256':digest(suite),'primary_count':40,'groups':{'feeding_standard':10,'feeding_varied':10,'loading_standard':10,'loading_varied':10},
        'scenario_generation':'deterministic seed + conservative analytic layout/reach-envelope guards; no manipulation outcomes inspected',
        'replacement_rule':'candidate seed = scenario seed + candidate_index*100000; analytic invalid candidates retained; frozen after generation; no outcome-based replacement',
        'reachability_scope':'Conservative envelope near M1 nominal source/target corridor; analytic feasibility is not a guarantee of controller success.',
        'control_horizon':{'total_steps':1500,'reset_steps':40,'maximum_execution_steps':1460,'settling_steps_reserved':24,'frequency_hz':20},
        'termination':{'expert':'M1 scripted attempt; retry failed lift at most once, then 24 M1 retreat-hold settling steps; exceptions/timeouts retain final state and diagnostics',
                       'pretrained':'At most 1436 model-action steps, then 24 zero-motion hold actions retaining last model gripper command; no truth-based early stop or automatic release',
                       'truth_scoring':'Unchanged M1 score_episode on final truth and last12 samples every2 settling steps; physical success is separate from controller exception status'},
        'expert_version':'M1 scripted_attempt unchanged at dce6ffb; maximum2 attempts',
        'policy_observations':['primary RGB','secondary RGB','robot proprioception','camera calibration','task instruction/specification'],
        'privilege':'Only Baseline A expert may access object truth for control. Evaluator logs truth independently for all methods.',
        'metrics':{'physical_success':'all unchanged M1 criteria; independent of exception flag','position_error_mm':'1000*Euclidean XY distance','yaw_error_deg':'wrapped full-directed angle, 180deg remains distinct','grasp_attempts':'expert M1 attempt calls; policies close-command transitions with independent lift/contact diagnostics','failed_grasps':'attempt without verified70mm lift','recovery_attempts':'attempts after a failed manipulation/grasp','recovery_success':'prior failure followed by final physical success','premature_completion':'explicit completion claim with final physical failure; null if no completion signal exists','controller_failures':'nonconverged move phases + execution exceptions, separate from failed episodes','latency':'per inference chunk seconds; null for geometric expert'},
        'diagnostics':'five separate reset/forced-grasp diagnostics; never included in primary40 denominators',
        'transfer_guard':'Loading evaluation cases frozen. Do not use outcomes to choose training hyperparameters; no geometry/tolerance changes for M3 comparison.',
        'reproduce_command':'.venv-sim/bin/python -m supervisor.baseline.benchmark --method expert --protocol results/personal-m2-20261009 --output results/personal-m2-20261009/expert-reproduction'}
    (output/'scenarios.json').write_text(json.dumps(suite,indent=2,allow_nan=False)+'\n')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2,allow_nan=False)+'\n')
    lock={'protocol_sha256':digest(protocol),'scenarios_sha256':digest(suite)}
    (output/'frozen.lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    return protocol


def load_suite(protocol_path, scenarios_path=None):
    path=Path(protocol_path)
    if path.is_dir():path=path/'protocol.json'
    protocol=json.loads(path.read_text())
    suite=json.loads(Path(scenarios_path or path.with_name('scenarios.json')).read_text())
    lock=json.loads(path.with_name('frozen.lock.json').read_text())
    if digest(protocol)!=lock['protocol_sha256'] or digest(suite)!=lock['scenarios_sha256'] or digest(suite)!=protocol['scenario_sha256']:
        raise ValueError('Frozen protocol/scenario hash mismatch')
    if environment_hashes()!=protocol['environment_source_sha256']:
        raise ValueError('M1 environment/controller/scorer source changed since protocol freeze')
    return protocol,suite


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT)
    args=p.parse_args();print(json.dumps(freeze(args.output),indent=2))
