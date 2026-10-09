"""Replay frozen expert actions for failure diagnosis; never changes control or score."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from supervisor.feeding_loading.environment import FeedingLoadingEnv


def compare_truth(actual, expected):
    return max(float(np.max(np.abs(np.asarray(actual[k], dtype=float) - np.asarray(v, dtype=float))))
               for k, v in expected.items())


def replay(row, destination):
    source = Path(row['evidence_directory'])
    scenario = json.loads((source/'scenario.json').read_text())
    env = FeedingLoadingEnv(scenario['config'])
    # No image capture during analysis; observations do not change simulator state.
    env.observe = lambda: None
    env.reset(scenario['seed'])
    initial_error = compare_truth(env.truth(), row['initial_truth'])
    assert initial_error <= 1e-8
    sim, robot = env.env.sim, env.env.robots[0]
    positions = list(robot._ref_joint_pos_indexes)
    joint_ids = [int(np.flatnonzero(sim.model.jnt_qposadr == p)[0]) for p in positions]
    names = [sim.model.joint_id2name(j) for j in joint_ids]
    limits = sim.model.jnt_range[joint_ids].copy()
    samples = []
    maximum_trace_error = 0.
    trace_path = destination/(row['scenario_id']+'.jsonl.gz')
    with gzip.open(source/'trajectory.jsonl.gz', 'rt') as original, gzip.open(trace_path, 'wt') as output:
        for line in original:
            record = json.loads(line)
            env.step(record['requested_action'])
            truth = env.truth()
            error = compare_truth(truth, record['truth'])
            maximum_trace_error = max(maximum_trace_error, error)
            q = sim.data.qpos[positions].copy()
            margins = np.minimum(q-limits[:, 0], limits[:, 1]-q)
            constraint_indices = np.flatnonzero(np.asarray(sim.data.efc_type[:sim.data.nefc]) == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))
            active_limits = [{'joint':sim.model.joint_id2name(int(sim.data.efc_id[i])),
                              'constraint_force':float(sim.data.efc_force[i])} for i in constraint_indices]
            contacts = []
            for contact in sim.data.contact[:sim.data.ncon]:
                pair = [sim.model.geom_id2name(int(g)) or str(g) for g in (contact.geom1, contact.geom2)]
                if any(n.startswith('fixture_') for n in pair):
                    contacts.append({'geoms':pair, 'distance_m':float(contact.dist)})
            phase_index = next((i for i,p in enumerate(row['move_events']) if p['start_step'] < env.steps <= p['end_step']), None)
            sample = {'step':env.steps, 'joint_position_rad':q.tolist(), 'joint_limit_margin_rad':margins.tolist(),
                      'active_joint_limit_constraints':active_limits, 'fixture_contacts':contacts,
                      'phase_index':phase_index, 'eef_position_m':truth['eef_position_m'], 'truth_replay_error':error}
            if phase_index is not None:
                phase = row['move_events'][phase_index]
                orientation = sim.data.site_xmat[robot.eef_site_id].reshape(3,3)
                sample['eef_position_error_m'] = float(np.linalg.norm(np.asarray(truth['eef_position_m'])-phase['target_xyz_m']))
                sample['eef_orientation_error_rad'] = float(Rotation.from_matrix(env.wrist_orientation(phase['wrist_yaw_rad']) @ orientation.T).magnitude())
            samples.append(sample)
            output.write(json.dumps(sample)+'\n')
    final_error = compare_truth(env.truth(), row['final_truth'])
    env.close()
    assert final_error <= 1e-8, (row['scenario_id'], final_error)
    phases = []
    phase_names = [p['phase'] for a in row['attempts'] for p in a['phases']]
    for i, event in enumerate(row['move_events']):
        selected = [s for s in samples if s['phase_index'] == i]
        margins = np.array([s['joint_limit_margin_rad'] for s in selected])
        phases.append({**event, 'phase':phase_names[i], 'final_eef_position_error_m':selected[-1]['eef_position_error_m'],
            'final_eef_orientation_error_rad':selected[-1]['eef_orientation_error_rad'],
            'minimum_margin_by_joint_rad':dict(zip(names,margins.min(axis=0).tolist())),
            'final_joint_position_rad':dict(zip(names,selected[-1]['joint_position_rad'])),
            'final_joint_margin_rad':dict(zip(names,selected[-1]['joint_limit_margin_rad'])),
            'active_limit_samples_by_joint':dict(Counter(c['joint'] for s in selected for c in s['active_joint_limit_constraints'])),
            'fixture_contact_sample_count':sum(bool(s['fixture_contacts']) for s in selected),
            'fixture_contact_pairs':sorted({tuple(c['geoms']) for s in selected for c in s['fixture_contacts']})})
    return {'scenario_id':row['scenario_id'], 'seed':row['seed'], 'initial_truth_error_max':initial_error,
            'final_truth_error_max':final_error, 'all_steps_truth_error_max':maximum_trace_error,
            'joint_limits_rad':dict(zip(names,limits.tolist())), 'phases':phases,
            'original_success':row['success'], 'original_position_error_mm':row['position_error_mm'],
            'original_yaw_error_deg':row['yaw_error_deg'], 'original_criteria':row['evaluation']['criteria'],
            'original_grasp_attempts':row['grasp_attempts'], 'original_failed_grasps':row['failed_grasps'],
            'trace':str(trace_path), 'source_action_trace_sha256':hashlib.sha256((source/'trajectory.jsonl.gz').read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('results/personal-m2-20261009'))
    args = parser.parse_args()
    destination=args.root/'expert_failure_replays'
    destination.mkdir(exist_ok=False)
    rows=[json.loads(line) for line in (args.root/'expert/episodes.jsonl').read_text().splitlines()]
    report={'purpose':'Exact requested-action diagnostic replay of failed frozen expert episodes; no controller, scenario, tolerance or result changes.',
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'analysis_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'reproduce_command':'.venv-sim/bin/python -m supervisor.baseline.replay_failure (fresh analysis output paths required)', 'episodes':[]}
    for row in rows:
        if row['success'] is False:
            result=replay(row,destination);report['episodes'].append(result)
            print(json.dumps({'scenario_id':row['scenario_id'],'exact_replay_error':result['final_truth_error_max']}),flush=True)
    (args.root/'expert_failure_analysis.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
