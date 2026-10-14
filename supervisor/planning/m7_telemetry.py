"""Optional observer-only telemetry. Never mutates observations or motor commands."""
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone
import imageio.v2 as imageio
import numpy as np


def finite(value):
    if isinstance(value, np.ndarray): return finite(value.tolist())
    if isinstance(value, np.generic): return finite(value.item())
    if isinstance(value, dict): return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [finite(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


class LiveTelemetry:
    """Latest snapshot only; camera timestamp enables consumers to mark stale feeds."""
    def __init__(self, directory, run_id, model, scenario_id, interval_s=.2, metadata_provider=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id, self.model, self.scenario_id = run_id, model, scenario_id
        self.interval_s = interval_s
        self.last_update = 0.
        self.sequence = 0
        self.requested = self.applied = self.target = None
        self.last_error = None
        self.last_guard = None
        self.metadata_provider = metadata_provider

    def set_target(self, target):
        self.target = np.asarray(target).copy()

    def _atomic(self, name, data):
        target = self.directory/name
        temporary = self.directory/(name + f'.{os.getpid()}.tmp')
        temporary.write_bytes(data)
        os.replace(temporary, target)

    def publish(self, env, observation=None, phase='executing', status='running', guard=None, force=False):
        if not force and time.monotonic()-self.last_update < self.interval_s:
            return
        try:
            if guard is not None:
                self.last_guard = finite(guard)
            observation = observation if observation is not None else env.observe()
            # Only policy-observable proprio/camera channels plus simulator actuator telemetry.
            p = observation['proprio']
            robot = env.env.robots[0]
            names = list(robot.robot_model.joints)
            sim = env.env.sim
            quat = np.asarray(p['eef_quat_xyzw'])
            fingers = np.asarray(p['gripper_joint_position_m'])
            telemetry = {'source': 'MuJoCo controller observation; simulated measurements',
                'joint_names': names, 'joint_position_rad': p['joint_position_rad'],
                'joint_velocity_rad_s': p['joint_velocity_rad_s'],
                'joint_limits_rad': sim.model.jnt_range[[sim.model.joint_name2id(n) for n in names]],
                'eef_position_m': p['eef_position_m'], 'eef_quat_wxyz': quat[[3,0,1,2]],
                'gripper_joint_position_m': fingers, 'gripper_open': bool(np.abs(fingers).sum() > .055),
                'gripper_open_rule': 'sum(abs(measured finger positions)) > 0.055m',
                'joint_torque_nm': None, 'joint_torque_source': 'unavailable',
                'temperature_c': None, 'temperature_source': 'unavailable; no temperature sensor in simulation'}
            if hasattr(sim.data, 'qfrc_actuator'):
                telemetry['joint_torque_nm'] = np.asarray(sim.data.qfrc_actuator)[robot._ref_joint_vel_indexes].copy()
                telemetry['joint_torque_source'] = 'MuJoCo data.qfrc_actuator at arm DOF indices; simulated generalized actuator torques, not hardware sensors'
            now = time.time()
            self.sequence += 1
            frame = np.concatenate([observation['rgb'][key] for key in ['primary', 'secondary']], axis=1)
            payload = finite({'schema': 'm7-observer-telemetry-v1', 'sequence': self.sequence,
                'timestamp_unix': now, 'timestamp_utc': datetime.fromtimestamp(now, timezone.utc).isoformat(),
                'image_timestamp_unix': now, 'run_id': self.run_id, 'model': self.model,
                'scenario_id': self.scenario_id, 'control_step': env.steps,
                'sim_time_s': env.steps/env.config['control']['frequency_hz'],
                'phase': phase, 'status': status, 'live': status not in ('finished', 'error', 'inactive'),
                'image_file': 'latest.jpg', 'camera_order': ['primary', 'secondary'],
                'telemetry': telemetry,
                'command': {'source': 'policy target and requested/applied OSC commands, separate from measured state',
                    'target_20': self.target, 'target_format': 'baseline: primary xyz_m, rotation6d, gripper, second arm unused',
                    'requested_action': self.requested, 'applied_action': self.applied,
                    'units': '7 normalized OSC components: dx,dy,dz,rotation vector xyz,gripper; each in [-1,1]',
                    'translation_step_m': env.config['control']['translation_step_m'],
                    'rotation_step_rad': env.config['control']['rotation_step_rad']},
                'guard': guard, 'evaluator_truth_in_actor': bool(getattr(self,'privileged_held_cue',False)),
                'actor_auxiliary_annotation': 'causal simulator held label only' if getattr(self,'privileged_held_cue',False) else None,
                'object_truth': None,
                'note': 'Camera and robot telemetry only. Evaluation success must come from separately labelled evaluator results.'})
            if self.metadata_provider is not None:
                payload['source_contract'] = finite(self.metadata_provider(env, observation, self))
            self._atomic('latest.jpg', imageio.imwrite('<bytes>', frame, format='jpg', quality=85))
            encoded = (json.dumps(payload, indent=2, allow_nan=False)+'\n').encode()
            self._atomic('latest.json', encoded)
            self._atomic('frame.json', encoded)
            self.last_update = time.monotonic()
        except Exception as exc:
            # Observability must not change controller/scoring behavior.
            self.last_error = f'{type(exc).__name__}: {exc}'
            try: self._atomic('publisher-error.json', json.dumps({'error': self.last_error, 'timestamp_unix': time.time()}).encode())
            except Exception: pass


def attach_live(env, run_id, model, scenario_id, metadata_provider=None):
    directory = os.environ.get('M7_LIVE_DIR')
    if not directory:
        return None
    publisher = LiveTelemetry(directory, run_id, model, scenario_id, metadata_provider=metadata_provider)
    raw_inner = env.env.step
    def inner(action, *args, **kwargs):
        publisher.applied = np.asarray(action).copy()
        return raw_inner(action, *args, **kwargs)
    env.env.step = inner
    raw_step = env.step
    def step(action):
        publisher.requested = np.asarray(action).copy()
        result = raw_step(action)
        publisher.publish(env)
        return result
    env.step = step
    if hasattr(env, 'close'):
        raw_close = env.close
        def close():
            publisher.publish(env, phase='finished', status='finished', guard=publisher.last_guard, force=True)
            return raw_close()
        env.close = close
    publisher.publish(env, phase='initial', force=True)
    return publisher


def mark_finished(publisher, result_status='evaluated'):
    try:
        path = publisher.directory/'latest.json'
        data = json.loads(path.read_text())
        if data.get('scenario_id') != publisher.scenario_id or data.get('run_id') != publisher.run_id:
            return
        data.update(phase='finished', status='finished', live=False, evaluation_status=result_status,
                    finished_timestamp_unix=time.time())
        encoded = (json.dumps(data, indent=2)+'\n').encode()
        publisher._atomic('latest.json', encoded)
        publisher._atomic('frame.json', encoded)
    except Exception as exc:
        publisher.last_error = str(exc)
