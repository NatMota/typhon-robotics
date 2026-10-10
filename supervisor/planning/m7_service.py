"""Stateful, inference-only goal checking and bounded action supervision.

Local trusted callers only. EEF path and measured joint guards are not whole-arm
collision checking or a physical-robot safety system. No object truth is accepted.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import select
import subprocess
import time
import numpy as np

GOAL_SYSTEM = '''You check a simulated robot task using CURRENT camera images and a
DESIRED FINISHED reference image. The reference shows the requested final state,
not what has happened. Compare the orange DUPLO block's location, visible blue
numeral 2 orientation, and release from the gripper. Ignore irrelevant robot arm
pose changes. The block must be inside the marked target and match the reference
orientation. A crop alone does not prove location; use the full current view too.
Do not invent contact, hidden faces, poses or previous action success. If the
block is hidden, the numeral is illegible, or release cannot be established,
answer unknown. Do not give motor commands or coordinates.
Return ONE JSON object, no markdown, exactly these fields:
{"status":"complete|incomplete|unknown", "confidence":0.0,
 "released":true, "evidence":"brief visible evidence"}.
released may be true, false, or null when uncertain. confidence is your subjective
certainty, not a calibrated probability. Say complete only when visible position,
orientation and release all match the reference.'''


def parse_goal(raw, require_release=True):
    text = raw.strip()
    if text.startswith('```json') and text.endswith('```'):
        text = text[7:-3].strip()
    result = json.loads(text)
    if not isinstance(result, dict) or set(result) != {'status', 'confidence', 'released', 'evidence'}:
        raise ValueError('invalid_goal_schema')
    if result['status'] not in ('complete', 'incomplete', 'unknown'):
        raise ValueError('invalid_goal_status')
    if type(result['confidence']) not in (int, float) or not math.isfinite(result['confidence']) or not 0 <= result['confidence'] <= 1:
        raise ValueError('invalid_goal_confidence')
    if result['released'] is not None and type(result['released']) is not bool:
        raise ValueError('invalid_release')
    if not isinstance(result['evidence'], str) or len(result['evidence']) > 1200:
        raise ValueError('invalid_goal_evidence')
    if require_release and result['status'] == 'complete' and result['released'] is not True:
        raise ValueError('completion_without_release_evidence')
    return result


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def log_value(value):
    if isinstance(value, (np.ndarray, np.generic, Path)):
        return log_value(serial(value))
    if isinstance(value, dict):
        return {k: log_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [log_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def segment_box(start, end, box):
    """Closed line/AABB intersection; catches segments crossing between samples."""
    lower, upper = np.asarray(box, float).T
    t0, t1 = 0., 1.
    for x, dx, lo, hi in zip(start, np.asarray(end) - start, lower, upper):
        if abs(dx) < 1e-12:
            if x < lo or x > hi:
                return False
        else:
            near, far = sorted(((lo-x)/dx, (hi-x)/dx))
            t0, t1 = max(t0, near), min(t1, far)
            if t0 > t1:
                return False
    return True


def reference_agreement(current, reference):
    """Restricted fixed-camera RGB veto; never reads object pose or simulator truth.

    Colored silhouette matching only covers this DUPLO appearance and viewpoint.
    It can veto Qwen, not independently establish grasp release or general success.
    """
    import cv2
    from supervisor.adaptation.observations import object_crop
    def glyph(rgb):
        crop, info = object_crop(rgb)
        if crop is None:
            return None, info
        mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_RGB2HSV),
                          np.array([85, 90, 40], np.uint8), np.array([135, 255, 255], np.uint8))
        ys, xs = np.where(mask > 0)
        info['blue_pixels'] = int(len(xs))
        if len(xs) < 6:
            return None, info
        mask = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
        return cv2.resize(mask, (32, 32), interpolation=cv2.INTER_NEAREST) > 0, info
    if current.shape != reference.shape:
        return {'status': 'unknown', 'reason': 'camera_shape_mismatch'}
    a, current_info = glyph(current)
    b, goal_info = glyph(reference)
    result = {'current': current_info, 'reference': goal_info,
              'position_tolerance_px': .025*current.shape[1],
              'minimum_glyph_iou': .75, 'minimum_rotation_margin': .10}
    if a is None or b is None:
        return {**result, 'status': 'unknown', 'reason': 'unobservable_numeral'}
    scores = []
    for k in range(4):
        rotated = np.rot90(b, k)
        scores.append(float(np.logical_and(a, rotated).sum()/np.logical_or(a, rotated).sum()))
    def center(info):
        x, y, w, h = info['component_bbox_xywh']
        return np.array([x+w/2, y+h/2])
    distance = float(np.linalg.norm(center(current_info)-center(goal_info)))
    result.update(glyph_iou_0_90_180_270=scores, center_error_px=distance)
    match = distance <= result['position_tolerance_px'] and scores[0] >= .75 and scores[0]-max(scores[1:]) >= .10
    return {**result, 'status': 'match' if match else 'mismatch',
            'reason': 'visible_reference_agreement' if match else 'location_or_numeral_disagreement'}


@dataclass
class ServiceConfig:
    workspace_bounds_m: list
    joint_limits_rad: list
    no_go_boxes_m: tuple = ()
    clearance_m: float = .015
    max_joint_speed_rad_s: float = 3.
    max_target_step_m: float = .12
    max_candidates: int = 1
    execution_prefix: int = 3
    max_chunk_steps: int = 30
    decision_budget_s: float | None = 6.
    observation_max_age_s: float | None = 6.
    completion_frames: int = 3
    completion_min_span_s: float = .1
    completion_confidence: float = .8
    visual_success_authority: str = 'qwen_reference'

    def __post_init__(self):
        if self.visual_success_authority not in ('qwen_reference', 'simple_image_classifier'):
            raise ValueError('invalid_visual_success_authority')
        for name, shape in (('workspace_bounds_m', (3, 2)), ('joint_limits_rad', (7, 2))):
            a = np.asarray(getattr(self, name), float)
            if a.shape != shape or not np.isfinite(a).all() or np.any(a[:, 0] >= a[:, 1]):
                raise ValueError(name)
        for box in self.no_go_boxes_m:
            a = np.asarray(box, float)
            if a.shape != (3, 2) or not np.isfinite(a).all() or np.any(a[:, 0] >= a[:, 1]):
                raise ValueError('no_go_box')
        if not 1 <= self.max_candidates <= 8 or not 1 <= self.execution_prefix <= 30 or self.completion_frames < 2 or not 1 <= self.max_chunk_steps <= 80:
            raise ValueError('invalid_count_budget')
        if not all(math.isfinite(v) for v in (self.max_target_step_m, self.max_joint_speed_rad_s, self.clearance_m, self.completion_min_span_s, self.completion_confidence)) or min(self.max_target_step_m, self.max_joint_speed_rad_s) <= 0 or self.clearance_m < 0 or self.completion_min_span_s < 0 or not 0 <= self.completion_confidence <= 1:
            raise ValueError('invalid_budget_or_margin')
        for budget in (self.decision_budget_s, self.observation_max_age_s):
            if budget is not None and (not math.isfinite(budget) or budget <= 0):
                raise ValueError('invalid_optional_time_budget')


class Guard:
    def __init__(self, config):
        self.config = config

    def measured(self, observation):
        p = observation['proprio']
        q, dq, xyz = (np.asarray(p[k], float) for k in ('joint_position_rad', 'joint_velocity_rad_s', 'eef_position_m'))
        if q.shape != (7,) or dq.shape != (7,) or xyz.shape != (3,) or not all(np.isfinite(a).all() for a in (q, dq, xyz)):
            return 'invalid_proprio'
        joint = np.asarray(self.config.joint_limits_rad)
        if np.any(q < joint[:, 0]) or np.any(q > joint[:, 1]):
            return 'measured_joint_limit'
        if np.any(np.abs(dq) > self.config.max_joint_speed_rad_s):
            return 'measured_joint_speed'
        bounds = np.asarray(self.config.workspace_bounds_m)
        if np.any(xyz < bounds[:, 0]) or np.any(xyz > bounds[:, 1]):
            return 'measured_workspace_limit'
        for box in self.config.no_go_boxes_m:
            expanded = np.asarray(box, float) + np.array([-1, 1]) * self.config.clearance_m
            if segment_box(xyz, xyz, expanded):
                return 'measured_no_go'
        return None

    def candidate(self, actions, observation):
        reason = self.measured(observation)
        if reason:
            return reason
        a = np.asarray(actions, float)
        if a.ndim != 2 or a.shape[1] != 20 or not 1 <= len(a) <= self.config.max_chunk_steps or not np.isfinite(a).all():
            return 'invalid_action_chunk'
        if np.any(a[:, 9] < 0) or np.any(a[:, 9] > 1):
            return 'invalid_gripper_probability'
        # Native adapter expects first two independent rotation columns.
        if np.any(np.linalg.norm(a[:, 3:6], axis=1) < 1e-8) or np.any(np.linalg.norm(np.cross(a[:, 3:6], a[:, 6:9]), axis=1) < 1e-8):
            return 'degenerate_rotation'
        points = np.vstack([observation['proprio']['eef_position_m'], a[:, :3]])
        bounds = np.asarray(self.config.workspace_bounds_m)
        if np.any(points < bounds[:, 0]) or np.any(points > bounds[:, 1]):
            return 'proposed_workspace_limit'
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > self.config.max_target_step_m):
            return 'proposed_target_jump'
        for box in self.config.no_go_boxes_m:
            expanded = np.asarray(box, float) + np.array([-1, 1]) * self.config.clearance_m
            if any(segment_box(left, right, expanded) for left, right in zip(points[:-1], points[1:])):
                return 'proposed_no_go_crossing'
        return None


class SupervisorService:
    """Calls checker(obs, reference, deadline) and proposer(obs, deadline).

    Proposers must not advance executed-action history while sampling candidates.
    Output is a short target prefix; caller rechecks live proprio before each step.
    """
    def __init__(self, config, checker, proposer, log_path, clock=time.monotonic):
        self.config, self.guard = config, Guard(config)
        self.checker, self.proposer, self.clock = checker, proposer, clock
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.reset('initial')

    def record(self, event):
        with self.log_path.open('a') as stream:
            stream.write(json.dumps(log_value(event), allow_nan=False) + '\n')

    def reset(self, episode_id):
        self.episode_id, self.complete, self.streak = str(episode_id), False, []
        self.last_frame, self.decisions, self.bound_goal = None, 0, None
        self.record({'type': 'reset', 'episode_id': self.episode_id, 'config': asdict(self.config)})

    def check_execution(self, target, observation, valid_until):
        if not math.isfinite(valid_until) or self.clock() >= valid_until:
            return 'action_deadline'
        return self.guard.candidate(np.asarray([target]), observation)

    def step(self, observation, reference, frame_id, observed_at, episode_deadline=None):
        start = self.clock()
        deadline = min(start + (self.config.decision_budget_s if self.config.decision_budget_s is not None else math.inf),
                       observed_at + (self.config.observation_max_age_s if self.config.observation_max_age_s is not None else math.inf),
                       episode_deadline if episode_deadline is not None else math.inf)
        event = {'type': 'decision', 'episode_id': self.episode_id, 'index': self.decisions,
                 'frame_id': str(frame_id), 'observation_age_s': start-observed_at,
                 'image_sha256': {k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k, v in observation['rgb'].items()},
                 'goal': observation['task'], 'proprio': observation['proprio'], 'candidates': []}
        self.decisions += 1
        def finish(status, reason, actions=None):
            event.update(status=status, reason=reason, elapsed_s=self.clock()-start, completion_streak=len(self.streak), complete_latched=self.complete)
            if actions is not None:
                event['selected_actions'] = actions
            self.record(event)
            return {'status': status, 'reason': reason, 'actions': actions, 'valid_until': deadline, 'event': event}
        goal_key = hashlib.sha256(json.dumps(log_value(observation['task']), sort_keys=True).encode() + b''.join(np.asarray(reference[k]).tobytes() for k in sorted(reference))).hexdigest()
        if self.bound_goal is None:
            self.bound_goal = goal_key
        elif goal_key != self.bound_goal:
            self.complete, self.streak = False, []
            return finish('hold', 'goal_changed_reset_required')
        if self.complete:
            return finish('complete', 'completion_latched_until_reset')
        if frame_id == self.last_frame:
            return finish('hold', 'duplicate_frame')
        self.last_frame = frame_id
        if not math.isfinite(observed_at) or start < observed_at or start >= deadline:
            self.streak = []
            return finish('hold', 'stale_or_future_observation')
        reason = self.guard.measured(observation)
        if reason:
            self.streak = []
            return finish('hold', reason)
        try:
            result = self.checker(observation, reference, deadline)
            classifier_mode = self.config.visual_success_authority == 'simple_image_classifier'
            if classifier_mode and result.get('checker_kind') != 'simple_goal_classifier':
                raise ValueError('classifier_authority_mismatch')
            parsed = parse_goal(json.dumps(result['assessment']), require_release=not classifier_mode)
            event['checker'] = result
        except Exception as exc:
            self.streak = []
            event['checker_error'] = str(exc)
            return finish('hold', 'checker_timeout' if isinstance(exc, TimeoutError) else 'checker_error')
        if self.clock() >= deadline:
            self.streak = []
            return finish('hold', 'decision_deadline')
        decisive = classifier_mode or parsed['confidence'] >= self.config.completion_confidence
        if parsed['status'] == 'complete' and decisive:
            if not classifier_mode and result.get('reference_agreement', {}).get('status') != 'match':
                self.streak = []
                return finish('hold', 'reference_agreement_veto')
            grip = np.asarray(observation['proprio'].get('gripper_joint_position_m', []), float)
            if grip.shape != (2,) or not np.isfinite(grip).all() or np.abs(grip).sum() <= .055:
                self.streak = []
                return finish('hold', 'gripper_release_unconfirmed')
            self.streak.append(observed_at)
            if len(self.streak) >= self.config.completion_frames and self.streak[-1]-self.streak[0] >= self.config.completion_min_span_s:
                self.complete = True
                return finish('complete', 'consecutive_visual_completion')
            return finish('hold', 'verify_completion_again')
        self.streak = []
        if parsed['status'] != 'incomplete' or not decisive:
            return finish('hold', 'visual_uncertainty')
        accepted = []
        for index in range(self.config.max_candidates):
            if self.clock() >= deadline:
                break
            at = self.clock()
            try:
                actions = np.asarray(self.proposer(observation, deadline), float)
                reason = self.guard.candidate(actions, observation)
                log_actions = actions.tolist() if np.isfinite(actions).all() else 'nonfinite'
                item = {'index': index, 'actions': log_actions, 'rejection': reason, 'latency_s': self.clock()-at}
                event['candidates'].append(item)
                if reason is None:
                    score = 0.
                    if self.config.max_candidates > 1:
                        path = np.vstack([observation['proprio']['eef_position_m'], actions[:, :3]])
                        score = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum() + 2*np.linalg.norm(np.diff(path, n=2, axis=0), axis=1).sum())
                        item['smoothness_cost'] = score
                    accepted.append((score, index, actions))
            except Exception as exc:
                event['candidates'].append({'index': index, 'error': str(exc), 'latency_s': self.clock()-at})
                break
        # An overrun invalidates ALL proposals; inference cannot silently enlarge TTL.
        if self.clock() >= deadline:
            return finish('hold', 'decision_deadline')
        if not accepted:
            return finish('hold', 'no_admissible_candidate')
        if self.config.max_candidates == 1:
            _, chosen, actions = accepted[0]
            event['selection_rule'] = 'single proposal; no candidate ranking or resampling'
        else:
            _, chosen, actions = min(accepted, key=lambda value: value[0])
            event['selection_rule'] = 'shortest and smoothest admissible EEF path; no learned value or imagined rollout'
        event['selected_candidate'] = chosen
        return finish('execute', 'admissible_prefix', actions[:self.config.execution_prefix].copy())


class QwenGoalChecker:
    """Private subprocess uses installed Qwen3-VL-2B-Instruct; no network endpoint."""
    def __init__(self, root, model, log_path):
        self.log = open(log_path, 'a')
        self.process = subprocess.Popen([str(Path(root)/'.venv-qwen-planner/bin/python'), '-m',
            'supervisor.planning.m7_goal_worker', '--model', str(model)], cwd=root,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            env=dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false'))
        try:
            self.load_info = self.receive(300)
            if not self.load_info.get('ready'):
                raise RuntimeError(self.load_info)
        except Exception:
            self.close()
            raise

    def receive(self, timeout):
        if timeout <= 0 or not select.select([self.process.stdout], [], [], timeout)[0]:
            self.process.kill()
            raise TimeoutError('qwen_deadline')
        result = pickle.load(self.process.stdout)  # trusted local child only
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def __call__(self, observation, reference, deadline):
        if self.process.poll() is not None:
            raise RuntimeError('qwen_worker_unavailable')
        # Explicit channel whitelist; no evaluator truth or object states.
        payload = {'op': 'check', 'images': observation['rgb'], 'reference': reference,
                   'goal': observation['task']['instruction']}
        from supervisor.adaptation.observations import object_crop
        crops, crop_info = [], {}
        for label, rgb in [('DESIRED numeral crop', reference['primary']), ('CURRENT numeral crop', observation['rgb']['primary'])]:
            crop, info = object_crop(rgb)
            crop_info[label] = info
            if crop is not None:
                crops.append((label, crop))
        payload['crops'] = crops
        before = time.monotonic()
        pickle.dump(payload, self.process.stdin, protocol=4)
        self.process.stdin.flush()
        result = self.receive(deadline-time.monotonic())
        try:
            result['assessment'] = parse_goal(result['raw_text'])
            result['schema_valid'] = True
        except (ValueError, TypeError) as exc:
            result['schema_valid'] = False
            result['schema_error'] = str(exc)
            result['assessment'] = {'status': 'unknown', 'confidence': 0., 'released': None, 'evidence': 'Invalid checker output; hold.'}
        result['crop_info'] = crop_info
        result['reference_agreement'] = reference_agreement(observation['rgb']['primary'], reference['primary'])
        result['wall_s'] = time.monotonic()-before
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()
