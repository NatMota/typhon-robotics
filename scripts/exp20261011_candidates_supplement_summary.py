"""Threshold-free held-state phase metrics and strictly separate visual manifest."""
import argparse
import gzip
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from exp20261011_replan_candidates import read, save, CANDIDATE_SEEDS
from supervisor.feeding_loading.evaluation import angle_error, _upright_error


def phase_metrics(truth, cfg):
    pos = np.asarray(truth['object_position_m'])
    q = np.asarray(truth['object_quat_wxyz'])
    observed = Rotation.from_quat(q[[1, 2, 3, 0]])
    goal = Rotation.from_euler('z', cfg['task']['target_yaw_rad'])
    height = cfg['workspace']['table_z_m'] + cfg['object']['height_m'] / 2
    return {'target_xy_error_mm': float(np.linalg.norm(pos[:2] - cfg['task']['target_position_xy_m']) * 1000),
            'yaw_residual_deg': math.degrees(angle_error(truth['object_yaw_rad'], cfg['task']['target_yaw_rad'])),
            'full_pose_orientation_error_deg': math.degrees((observed * goal.inv()).magnitude()),
            'marked_face_upright_error_deg': math.degrees(_upright_error(q)),
            'height_above_target_mm': float((pos[2] - height) * 1000),
            'bilateral_grasp_contact': truth['bilateral_grasp_contact'],
            'any_finger_contact': truth['grasp_contact'], 'gripper_open': truth['gripper_open'],
            'released': bool(truth['gripper_open'] and not truth['grasp_contact'])}


def run(out):
    protocol = read(out / 'protocol.json')
    assert protocol['separate_from_primary48']
    qa = read(out / 'candidates/render-qualification-reproducible.json')
    assert qa['passed'] and qa['branches'] == 18
    results = []
    visual = []
    worst_current_changes = max(b['initial_render_difference']['nonzero_channels'] for state in qa['states'] for b in state['branches'])
    lines = ['# V4 held-state orientation and release supplement', '',
             'Sixteen proposals from two deliberately inspected held states, separate from the primary 48-proposal approach/grasp diagnostic. All new proposals use the unchanged V4 checkpoint. One state came from a prior V4 grasp and the other from a prior V3 grasp; these are counterfactual V4 continuations, not two independently acquired V4 grasps.', '',
             'Both states already hold the block. Existing contact or height receives no credit as a newly acquired pickup. Every proposal executes 30 native 20Hz actions then 24 fixed settling steps; no oracle selects actions for the policy.', '',
             '|Starting state|Candidate|Current yaw error|End yaw error|End full-pose error|End XY error mm|End bilateral contact|Settled released|Settled strict success|',
             '|---|---:|---:|---:|---:|---:|---|---|---|']
    for state in protocol['candidates']['states']:
        base = out / 'candidates'
        meta = read(base / 'inputs' / state / 'metadata.json')
        cfg = meta['scenario']['config']
        futures = base / 'futures' / state
        start = read(futures / 'qualification_original/start-state.json')['truth']
        initial = phase_metrics(start, cfg)
        assert initial['bilateral_grasp_contact'], (state, initial)
        candidates = []
        for seed in CANDIDATE_SEEDS:
            folder = futures / f'seed{seed}'
            result = read(folder / 'summary.json')
            with gzip.open(folder / 'trajectory.jsonl.gz', 'rt') as f:
                trace = [__import__('json').loads(line) for line in f]
            chunk = phase_metrics(result['chunk_end_truth'], cfg)
            settled = phase_metrics(result['terminal_truth'], cfg)
            item = {'candidate_id': str(seed), 'initial': initial, 'chunk_end': chunk, 'settled': settled,
                    'progress_at_chunk_end': {
                        'target_xy_mm': initial['target_xy_error_mm'] - chunk['target_xy_error_mm'],
                        'yaw_deg': initial['yaw_residual_deg'] - chunk['yaw_residual_deg'],
                        'full_pose_deg': initial['full_pose_orientation_error_deg'] - chunk['full_pose_orientation_error_deg'],
                        'lowering_mm': initial['height_above_target_mm'] - chunk['height_above_target_mm']},
                    'contact_retained_fraction_of_30_steps': sum(r['truth']['bilateral_grasp_contact'] for r in trace[:30]) / 30,
                    'first_nonbilateral_contact_step_1based': next((r['index'] + 1 for r in trace[:30] if not r['truth']['bilateral_grasp_contact']), None),
                    'first_open_command_step_1based': next((r['index'] + 1 for r in trace[:30] if r['requested_action'][-1] < 0), None),
                    'strict_success_after_settle': result['evaluation_after24step_settle']['success'],
                    'criteria_after_settle': result['evaluation_after24step_settle']['criteria'],
                    'current_truth': start, 'chunk_end_truth': result['chunk_end_truth'], 'settled_truth': result['terminal_truth'],
                    'generated_policy_checkpoint': protocol['checkpoint_sha256']}
            candidates.append(item)
            lines.append(f"|{state}|{seed}|{initial['yaw_residual_deg']:.2f}|{chunk['yaw_residual_deg']:.2f}|{chunk['full_pose_orientation_error_deg']:.2f}|{chunk['target_xy_error_mm']:.2f}|{chunk['bilateral_grasp_contact']}|{settled['released']}|{item['strict_success_after_settle']}|")
        results.append({'state_id': state, 'source_policy': meta['source_policy'], 'prefix_actions': meta['prefix_actions'],
                        'current_truth': start, 'current_metrics': initial, 'config_goal_and_tolerances': {k: cfg[k] for k in ['task', 'evaluation']},
                        'candidates': candidates})
        visual.append({'state_id': state, 'current_rgb': str(base / 'inputs' / state / 'current-raw-rgb.png'),
                       'current_robot_mask': str(futures / 'seed752001/initial-robot-mask.png'),
                       'candidates': [{'candidate_id': str(seed),
                                       'future_rgb': [str(futures / f'seed{seed}' / f'future-{t:02d}.png') for t in [10, 20, 30]],
                                       'future_robot_masks': [str(futures / f'seed{seed}' / f'future-{t:02d}-robot-mask.png') for t in [10, 20, 30]],
                                       'future_times_s': [.5, 1., 1.5],
                                       'settled_rgb': str(futures / f'seed{seed}/future-54.png')} for seed in CANDIDATE_SEEDS]})
    aggregate = []
    for state in results:
        rows = state['candidates']
        c = lambda row: row['criteria_after_settle']
        aggregate.append({'state': state['state_id'], 'candidates': len(rows),
            'strict_success': sum(r['strict_success_after_settle'] for r in rows),
            'region_placement_correct_orientation_released_stable_without_center': sum(all(c(r)[k] for k in ['orientation','marked_face_up','footprint_inside_target','placement_height','released','stable']) for r in rows),
            'criteria_counts': {k: sum(c(r)[k] for r in rows) for k in c(rows[0])},
            'settled_yaw_error_deg_range': [min(r['settled']['yaw_residual_deg'] for r in rows),max(r['settled']['yaw_residual_deg'] for r in rows)],
            'settled_xy_error_mm_range': [min(r['settled']['target_xy_error_mm'] for r in rows),max(r['settled']['target_xy_error_mm'] for r in rows)],
            'first_open_command_steps_1based': [r['first_open_command_step_1based'] for r in rows],
            'first_nonbilateral_contact_steps_1based': [r['first_nonbilateral_contact_step_1based'] for r in rows]})
    save(out / 'analysis.json', {'separate_from_primary48': True,'states': aggregate})
    lines += ['', '## Distinguish the failure criteria', '',
              '|Starting state|Stable correctly oriented region placement, without15mm centre test|Strict success|Final yaw range|Final centre error range mm|',
              '|---|---:|---:|---:|---:|']
    for row in aggregate:
        y,p = row['settled_yaw_error_deg_range'],row['settled_xy_error_mm_range']
        lines.append(f"|{row['state']}|{row['region_placement_correct_orientation_released_stable_without_center']}/8|{row['strict_success']}/8|{y[0]:.3f}–{y[1]:.3f}|{p[0]:.2f}–{p[1]:.2f}|")
    lines += ['', 'The unchanged V4 policy behaves differently from these two supplied grasp states. The off-axis grasp can finish in the square yet leave the numeral rotated. Correctly oriented regional placement from the near-expert grasp is useful partial evidence even when the separate15mm centring or stability criterion fails. This counterfactual does not prove that V4 can acquire the better grasp, and the two starting states differ in position and history as well as grasp geometry.', '']
    save(out / 'candidates/offline-comparator.json', {'schema': 'held-candidate-phase-comparator-v1',
        'truth_only_for_offline_scoring': True, 'visual_critic_must_not_receive_this_file': True,
        'adequacy_thresholds': 'No acceptability threshold here; fixed ghost-critic-v2 operational rubric is reported separately.',
        'held_at_start_is_not_new_pickup': True, 'states': results})
    save(out / 'candidates/visual-only-manifest.json', {'schema': 'candidate-ghost-visual-input-v2',
        'oracle_scores_and_object_truth_excluded': True, 'current_rgb_raw_and_unannotated': True,
        'current_rgb_qualification': str(out / 'candidates/render-qualification-reproducible.json'),
        'privileged_lookahead': 'Both future RGB and robot masks come from executed simulator branches. They are privileged lookahead even though numeric truth is excluded.',
        'camera_panels': 'Left primary overhead, right secondary oblique; same camera/lighting settings for all candidates.', 'states': visual})
    lines += ['', 'Yaw alone does not establish upright marked-face orientation. Full-pose error is the SO(3) angle between measured object orientation and the upright supplied target orientation. A release counts as task completion only when all unchanged position, yaw, marked-face, footprint, height, release and stability criteria pass.', '',
              f'Raw current RGB exactly matches the archived policy packet. Qualified simulator replay has at most{worst_current_changes}/1,572,864 channels differing by1/255 in initial renders; the failed exact assertion and every changed channel are retained, and fresh rendering is not described as pixel-exact. V4 receives its existing causal held marker downstream, but the critic receives the unmarked current image. Both future RGB and translucent robot ghosts require executed simulator lookahead and are privileged diagnostic evidence.', '',
              'The threshold-free offline comparator is `candidates/offline-comparator.json`; it must never be passed to the visual critic. The visual-only manifest contains images, neutral candidate IDs and sample times. The separate critic programme defines an operational progress/retention proxy before viewing outcomes.', '',
              'Reproduction (after preparing a fresh supplement output):', '```sh',
              '.venv-sim/bin/python scripts/exp20261011_candidates_supplement_prepare.py --output <supplement>',
              '.venv-sim/bin/python scripts/exp20261011_replan_candidates.py infer-candidates --output <supplement>',
              '.venv-sim/bin/python scripts/exp20261011_replan_candidates.py replay-candidates --output <supplement>',
              '.venv-sim/bin/python scripts/exp20261011_candidates_validate.py --output <supplement>',
              '.venv-sim/bin/python scripts/exp20261011_candidates_supplement_summary.py --output <supplement>', '```']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print({'states': len(results), 'candidates': sum(len(x['candidates']) for x in results), 'primary_pool_unchanged': True})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('/var/tmp/typhon-research-20261011/replanning-candidates-supplement'))
    run(p.parse_args().output)
