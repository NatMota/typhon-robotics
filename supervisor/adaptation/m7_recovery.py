"""Training-only, truth-gated corrections from a learner's unchanged live state.

This is expert-supervised dataset aggregation, not a reward optimizer. Object
truth is used by the training teacher and failure detector, never policy input.
"""
from collections import deque
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from .actions import robot_observation
from supervisor.baseline.xvla import to_action
from supervisor.feeding_loading.evaluation import score_episode


def goal_geometry(truth, config):
    criteria = score_episode(truth, truth, [], config, [])['criteria']
    return not truth['grasp_contact'] and all(ok for name, ok in criteria.items() if name not in {'stable', 'released'})


def failure_reason(initial, truths, config, budget_exhausted=False):
    """Independent training trigger; never supplied to the learned policy."""
    recent = list(truths)
    now = recent[-1]
    if goal_geometry(now, config):
        return 'completed_hold'
    if len(recent) >= 12:
        empty = all(not s['gripper_open'] and not s['grasp_contact'] for s in recent[-12:])
        above = now['eef_position_m'][2] > now['object_position_m'][2] + .06
        if empty and above:
            return 'failed_grasp_empty_closed_gripper'
    if budget_exhausted:
        check = score_episode(initial, now, recent[-12:], config, [])
        failed = [name for name, ok in check['criteria'].items() if not ok]
        return 'learner_budget_exhausted:' + ','.join(failed)
    return None


def move_pose(env, position, rotation, grip, limit=100):
    """Same OSC adapter as the learner, preserving arbitrary existing grasp pose."""
    stable = 0
    target = np.r_[position, rotation[:, 0], rotation[:, 1], float(grip > 0), np.zeros(10)]
    bounds = np.asarray(env.config['control']['workspace_bounds_m'])
    if np.any(position < bounds[:, 0]) or np.any(position > bounds[:, 1]):
        raise ValueError('Recovery target outside declared workspace')
    for _ in range(limit):
        obs = robot_observation(env)
        env.step(to_action(target, obs, env.config))
        current = robot_observation(env)['proprio']
        error = Rotation.from_matrix(rotation @ Rotation.from_quat(current['eef_quat_xyzw']).as_matrix().T).magnitude()
        stable = stable + 1 if np.linalg.norm(current['eef_position_m'] - position) < .008 and error < .06 else 0
        if stable >= 4:
            return True
    return False


def correct_live(env, phase, max_attempts=2):
    """Execute bounded geometric corrections, with no reset/state relocation.

    phase(label, callable) records exact supervised boundaries. An existing
    bilateral grasp keeps its measured rigid object-to-hand transform; an empty
    hand uses the established geometric source/grasp/place path.
    """
    config = getattr(env, 'episode_config', env.config)
    initial = env.truth()
    attempts = []
    if goal_geometry(initial, config):
        phase('maintain_completed_state', lambda: [env.step([0., 0., 0., 0., 0., 0., -1.]) for _ in range(24)])
        return attempts
    destination = np.r_[config['task']['target_position_xy_m'], config['workspace']['table_z_m'] + config['object']['height_m'] / 2]
    if config['task']['type'] == 'loading':
        destination[2] += config['workspace']['fixture_floor_height_m']
    target_yaw = config['task']['target_yaw_rad']
    for attempt_index in range(max_attempts):
        start = env.truth()
        source = np.asarray(start['object_position_m'])
        yaw = start['object_yaw_rad']
        attempt = {'attempt_index': attempt_index, 'forced_miss': False,
                   'entry_bilateral_contact': bool(start['bilateral_grasp_contact']), 'entry_truth': start}
        if start['bilateral_grasp_contact']:
            # Reuse the actual grasp transform, without opening or reattaching.
            robot = robot_observation(env)['proprio']
            eef = np.asarray(robot['eef_position_m'])
            hand_rotation = Rotation.from_quat(robot['eef_quat_xyzw']).as_matrix()
            object_rotation = Rotation.from_quat(np.asarray(start['object_quat_wxyz'])[[1, 2, 3, 0]]).as_matrix()
            hand_to_object = hand_rotation.T @ object_rotation
            offset = hand_rotation.T @ (source - eef)
            desired_object = Rotation.from_euler('z', target_yaw).as_matrix()
            desired_hand = desired_object @ hand_to_object.T
            clearance = config['workspace']['table_z_m'] + config['object']['height_m'] / 2 + .16
            lift_target = eef + [0., 0., max(0., clearance - source[2])]
            phase('lift', lambda: move_pose(env, lift_target, hand_rotation, 1.))
            lift = env.truth()
            attempt.update(lift_truth=lift, verified_lift=bool(lift['bilateral_grasp_contact'] and lift['object_position_m'][2] - config['workspace']['table_z_m'] > .07))
            if attempt['verified_lift']:
                high = destination + [0., 0., .16] - desired_hand @ offset
                low = destination + [0., 0., .012] - desired_hand @ offset
                phase('transport_and_reorient', lambda: move_pose(env, high, desired_hand, 1.))
                phase('place_lower', lambda: move_pose(env, low, desired_hand, 1.))
                phase('release_open', lambda: [env.step([0., 0., 0., 0., 0., 0., -1.]) for _ in range(20)])
                attempt['release_truth'] = env.truth()
                phase('retreat', lambda: move_pose(env, high, desired_hand, -1.))
        else:
            above = source + [0., 0., .16]
            grasp = source + [0., 0., .005]
            high = destination + [0., 0., .16]
            low = destination + [0., 0., .012]
            phase('approach_and_align', lambda: env.move_to(above, yaw, -1.))
            phase('descend_to_grasp', lambda: env.move_to(grasp, yaw, -1.))
            phase('grasp_close', lambda: env.tick(grasp, yaw, 1., steps=18))
            attempt['close_truth'] = env.truth()
            phase('lift', lambda: env.move_to(above, yaw, 1.))
            lift = env.truth()
            attempt.update(lift_truth=lift, verified_lift=bool(lift['object_position_m'][2] - source[2] >= .07 and lift['bilateral_grasp_contact']))
            if attempt['verified_lift']:
                phase('transport_and_reorient', lambda: env.move_to(high, target_yaw, 1.))
                phase('place_lower', lambda: env.move_to(low, target_yaw, 1.))
                phase('release_open', lambda: env.tick(low, target_yaw, -1., steps=20))
                attempt['release_truth'] = env.truth()
                phase('retreat', lambda: env.move_to(high, target_yaw, -1.))
        attempts.append(attempt)
        if attempt['verified_lift']:
            break
        # Physical reopen and clearance before retry; no simulator state edits.
        current = env.truth()
        safe = np.asarray(current['eef_position_m']); safe[2] = min(1.15, max(safe[2], source[2] + .16))
        phase('retreat', lambda: env.move_to(safe, current['object_yaw_rad'], -1.))
    phase('maintain_completed_state' if goal_geometry(env.truth(), config) else 'retreat',
          lambda: [env.step([0., 0., 0., 0., 0., 0., -1.]) for _ in range(24)])
    return attempts


def collect_episode(env, policy, learner_steps=120, seed=773901):
    """Return pre-action RGB/proprio, executed targets and a teacher-valid suffix.

    Caller constructs and resets the environment once. This function deliberately
    has no reset call. Stop detection is training-only privileged orchestration.
    """
    from .actions import absolute_target, proprio_vector
    config = getattr(env, 'episode_config', env.config)
    initial = env.truth()
    original_step = env.step
    primary, secondary, proprio, indices, targets, actions, raw_chunks, primitives = [], [], [], [], [], [], [], []
    truths = [initial]
    recent = deque([initial], maxlen=16)
    actor = 'learner'
    last_target = None
    force_frame = False
    handover_state = None
    expert_pre_state_digest = None
    state_unchanged_at_handover = False
    def step(command):
        nonlocal last_target, force_frame, expert_pre_state_digest, state_unchanged_at_handover
        index = len(targets)
        capture = index % 4 == 0 or force_frame
        observation = env.observe() if capture else robot_observation(env)
        if capture:
            primary.append(observation['rgb']['primary']); secondary.append(observation['rgb']['secondary'])
            p = proprio_vector(observation)
            if last_target is not None: p[:9] = last_target[:9]
            proprio.append(p); indices.append(index); force_frame = False
        target = absolute_target(command, observation, config)
        if actor == "geometric_expert" and expert_pre_state_digest is None:
            actual = env.env.sim.get_state().flatten().copy()
            state_unchanged_at_handover = bool(np.array_equal(actual, handover_state))
            assert state_unchanged_at_handover, "Simulator state changed at teacher handover"
            expert_pre_state_digest = hashlib.sha256(actual.tobytes()).hexdigest()
        original_step(command)
        truth = env.truth()
        targets.append(target); last_target = target
        truths.append(truth); recent.append(truth)
        actions.append({'control_step': env.steps, 'normalized_command': np.asarray(command).tolist(),
                        'controller': actor, 'teacher_supervised': actor == 'geometric_expert', 'post_action_truth': truth})
    def phase(label, operation):
        start = len(targets)
        converged = operation()
        end = len(targets)
        if end > start:
            primitives.append({'label': label, 'action_start': start, 'action_end_exclusive': end,
                               'converged': None if converged is None else bool(converged) if isinstance(converged, (bool, np.bool_)) else None})
    env.step = step
    error = None
    trigger = failure_reason(initial, recent, config)
    try:
        policy.reset(seed)
        while not trigger and len(targets) < learner_steps:
            chunk = np.asarray(policy.predict(env.observe()))
            if chunk.ndim != 2 or chunk.shape[1] != 20 or not np.isfinite(chunk).all():
                raise ValueError('Invalid learner chunk')
            raw_chunks.append(chunk)
            for target in chunk:
                if len(targets) >= learner_steps: break
                action_observation = env.observe() if len(targets) % 4 == 0 else robot_observation(env)
                command = policy.to_action(target, action_observation, config)
                env.step(command)
                trigger = failure_reason(initial, recent, config)
                if trigger: break
        trigger = trigger or failure_reason(initial, recent, config, budget_exhausted=True)
        handover = env.truth()
        handover_index = len(targets)
        # Capturing this exact state avoids labeling the preceding learner action.
        actor = 'geometric_expert'; force_frame = True
        handover_state = env.env.sim.get_state().flatten().copy()
        attempts = correct_live(env, phase)
        correction_initial_truth = truths[handover_index]
        assert correction_initial_truth == handover
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        env.step = original_step
    final = env.truth()
    score = score_episode(initial, final, truths[-12:], config, attempts)
    assert len(targets) == env.steps - initial['control_steps']
    assert all(a['controller'] == 'learner' for a in actions[:handover_index])
    assert all(a['teacher_supervised'] for a in actions[handover_index:])
    return {'arrays': {'primary': np.asarray(primary, np.uint8), 'secondary': np.asarray(secondary, np.uint8),
                      'proprio': np.asarray(proprio, np.float32), 'targets': np.asarray(targets, np.float32),
                      'frame_action_indices': np.asarray(indices, np.int64),
                      'teacher_mask': np.arange(len(targets)) >= handover_index},
            'actions': actions, 'primitives': primitives, 'raw_student_chunks': raw_chunks,
            'episode': {'initial_truth': initial, 'handover_truth': handover, 'final_truth': final,
                        'supervision_action_start': handover_index, 'learner_executed_actions': handover_index,
                        'trigger': trigger, 'attempts': attempts, 'success': score['success'], 'evaluation': score,
                        'reset_warmup_steps': initial['control_steps'], 'recorded_robot_action_count': len(targets),
                        'no_reset_or_state_relocation_after_learner': state_unchanged_at_handover,
                        'handover_sim_state_sha256': hashlib.sha256(handover_state.tobytes()).hexdigest(),
                        'first_expert_pre_action_sim_state_sha256': expert_pre_state_digest,
                        'privileged_truth_scope': 'training failure detector, geometric teacher and independent scoring only',
                        'policy_observation': 'RGB, robot proprioception and supplied task goal' + (' plus declared causal privileged held annotation' if getattr(policy, 'annotation_conditioned_v2', False) else ' only'),
                        'annotation_conditioned': bool(getattr(policy, 'annotation_conditioned_v2', False)),
                        'student_state_dataset_aggregation': handover_index > 0,
                        'completed_stop': trigger == 'completed_hold', 'controller_error': error}}


def self_check():
    """Small regression for independent empty-grasp and completed-stop semantics."""
    import copy
    from supervisor.feeding_loading.config import load_config
    config = load_config()
    target = config['task']['target_position_xy_m']
    truth = {'object_position_m': [*target, .815], 'object_quat_wxyz': [1., 0., 0., 0.],
             'object_yaw_rad': 0., 'object_linear_velocity_m_s': [0., 0., 0.],
             'object_angular_velocity_rad_s': [0., 0., 0.], 'eef_position_m': [0., 0., 1.08],
             'gripper_open': True, 'grasp_contact': False, 'bilateral_grasp_contact': False,
             'control_steps': 40, 'sim_time_s': 2.}
    assert failure_reason(truth, [truth], config) == 'completed_hold'
    failed = copy.deepcopy(truth); failed['object_position_m'][0] += .15; failed['gripper_open'] = False
    assert failure_reason(failed, [failed]*12, config) == 'failed_grasp_empty_closed_gripper'
    failed['gripper_open'] = True
    assert failure_reason(failed, [failed], config) is None
    assert failure_reason(failed, [failed], config, True).startswith('learner_budget_exhausted:')
    class Completed:
        def __init__(self): self.config = config; self.commands = []
        def truth(self): return truth
        def step(self, command): self.commands.append(command)
        def reset(self, *args): raise AssertionError('Reset forbidden during recovery')
    env = Completed(); phases = []
    def phase(label, operation): phases.append(label); operation()
    assert correct_live(env, phase) == []
    assert phases == ['maintain_completed_state']
    assert env.commands == [[0., 0., 0., 0., 0., 0., -1.]]*24
    print('Recovery self-check passed: completed hold only, empty grasp, timeout failure')


if __name__ == '__main__': self_check()
