"""Unchanged X-VLA LIBERO adapter. Policy inputs never include object truth."""
from pathlib import Path
import os
import pickle
import select
import subprocess
import time
import numpy as np
from scipy.spatial.transform import Rotation

MODEL_MANIFEST = {
    'method': 'Baseline B — Unchanged pretrained X-VLA',
    'repository': '2toINF/X-VLA-Libero',
    'revision': '129e71460678b7236cee6fc9707f09d9fa0c3590',
    'reference_source_revision': '6bc2513f5f1cbec715cc668b414392a6cae5c671',
    'adapter_version': 'm2-xvla-v2-supplied-goal', 'weights_changed': False,
    'uses_simulator_truth_for_control': False, 'dtype': 'float32',
    'domain_id': 3, 'denoising_steps': 10, 'chunk_size': 30,
    'language_max_length': 128, 'language_length_override_reason': 'Complete supplied goal text exceeds released 50-token limit',
    'execute_actions_per_chunk': 30, 'control_frequency_hz': 20,
    'images': ['M1 primary overhead', 'M1 secondary oblique'],
    'camera_mismatch': 'Released LIBERO external+wrist differs from M1 overhead+oblique',
    'proprio': 'initial measured world EE xyz+first two rotation columns+0+10zero; subsequent first9 = final target of previous chunk, matching released client',
    'prompt': 'M1 instruction + deterministic supplied task target centre world xy metres and target yaw degrees; identical formatter for every scenario',
    'action': 'absolute world xyz metres, first two rotation columns, close probability index9, unused second arm10:20',
    'conversion': 'live EE proprio to world OSC delta, elementwise translation/.05 and rotvec/.5 clipped[-1,1]; existing M1 workspace clamp',
    'completion_claim_interface': False,
}


def task_prompt(task):
    """Serialize supplied goal, never an observed object pose or evaluator state."""
    x, y = task['target_position_xy_m']
    yaw_deg = np.degrees(task['target_yaw_rad'])
    if not np.isfinite([x,y,yaw_deg]).all():
        raise ValueError('Task goal must be finite')
    return (task['instruction'] + f' Target centre in robot world metres: x={x:.6f}, y={y:.6f}.'
            f' Required block yaw: {yaw_deg:.6f} degrees about world +z;'
            ' yaw 0 means the upper edge of blue 2 points world +y.')


def to_action(model_action, observation, config):
    """Map one absolute model target into M1 OSC; requires only robot proprio."""
    a = np.asarray(model_action, dtype=float)
    if a.shape != (20,) or not np.isfinite(a).all():
        raise ValueError('X-VLA action must be a finite 20-vector')
    b1 = a[3:6]
    if np.linalg.norm(b1) < 1e-8: raise ValueError('Degenerate first rotation column')
    b1 = b1/np.linalg.norm(b1)
    b2 = a[6:9]-np.dot(b1,a[6:9])*b1
    if np.linalg.norm(b2) < 1e-8: raise ValueError('Degenerate second rotation column')
    b2 = b2/np.linalg.norm(b2)
    target = np.column_stack((b1,b2,np.cross(b1,b2)))
    p = observation['proprio']
    actual = Rotation.from_quat(p['eef_quat_xyzw']).as_matrix()
    rotation = Rotation.from_matrix(target @ actual.T).as_rotvec()
    ctl = config['control']
    return np.r_[np.clip((a[:3]-p['eef_position_m'])/ctl['translation_step_m'],-1,1),
                 np.clip(rotation/ctl['rotation_step_rad'],-1,1), 1. if a[9]>.5 else -1.]


class XvlaPolicy:
    """Long-lived GPU worker using a private, local-only pickle pipe."""
    model_manifest = MODEL_MANIFEST
    to_action = staticmethod(to_action)

    def __init__(self, root=None, log_path=None):
        root = Path(root or Path(__file__).resolve().parents[2])
        self.proprio = None
        self.inferences = []
        self.log = open(log_path or root/'results/personal-m2-20261009/pretrained/worker.log','a')
        env = dict(os.environ, HF_HOME=str(root/'.cache-xvla/hf'), HF_HUB_OFFLINE='1')
        self.process = subprocess.Popen([str(root/'.venv-xvla/bin/python'),'-m',
            'supervisor.baseline.xvla_worker',str(root/'.cache-xvla/checkpoint')],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,cwd=root,env=env)
        try:
            self.load_info = self._receive(240)
            if not self.load_info.get('ready'): raise RuntimeError(self.load_info)
        except Exception:
            self.close()
            raise

    def _receive(self, timeout=180):
        if not select.select([self.process.stdout],[],[],timeout)[0]:
            self.process.kill(); raise TimeoutError('X-VLA worker timed out; see worker.log')
        try: result = pickle.load(self.process.stdout)
        except EOFError as e: raise RuntimeError('X-VLA worker exited; see worker.log') from e
        if 'error' in result: raise RuntimeError(result['error'])
        return result

    def _request(self, payload):
        pickle.dump(payload,self.process.stdin,protocol=4); self.process.stdin.flush()
        return self._receive()

    def reset(self, seed):
        self.proprio = None
        self.inferences = []
        self._request({'op':'reset','seed':int(seed)})

    def predict(self, observation):
        # Whitelist RGB, robot proprio and supplied task goals as text. Never object truth.
        if self.proprio is None:
            p = observation['proprio']
            r = Rotation.from_quat(p['eef_quat_xyzw']).as_matrix()
            self.proprio = np.r_[p['eef_position_m'],r[:,0],r[:,1],0.,np.zeros(10)].astype(np.float32)
        start = time.perf_counter()
        result = self._request({'op':'predict','images':[observation['rgb']['primary'],observation['rgb']['secondary']],
            'instruction':task_prompt(observation['task']), 'proprio':self.proprio})
        actions = result.pop('actions')
        self.proprio[:9] = actions[-1,:9]
        result['adapter_wall_s'] = time.perf_counter()-start
        self.inferences.append(result)
        return actions

    def close(self):
        if hasattr(self,'process') and self.process.poll() is None:
            try:
                pickle.dump({'op':'close'},self.process.stdin,protocol=4);self.process.stdin.flush()
                self.process.wait(timeout=15)
            except Exception: self.process.kill()
        if hasattr(self,'log'): self.log.close()


def self_check():
    from supervisor.feeding_loading.config import load_config
    obs = {'proprio':{'eef_position_m':np.array([.1,.1,1.]),'eef_quat_xyzw':np.array([0,0,0,1])}}
    a=np.r_[.1,.1,1.,1.,0.,0.,0.,1.,0.,0.,np.zeros(10)]
    np.testing.assert_allclose(to_action(a,obs,load_config()),[0,0,0,0,0,0,-1])
    a[0]+=.1;a[9]=1
    np.testing.assert_allclose(to_action(a,obs,load_config()),[1,0,0,0,0,0,1])
    a[3:9]=0
    try: to_action(a,obs,load_config())
    except ValueError: pass
    else: raise AssertionError('degenerate rotation accepted')
    text=task_prompt({'instruction':'Place block.','target_position_xy_m':[.1,-.2],'target_yaw_rad':np.pi/2})
    assert 'x=0.100000, y=-0.200000' in text and '90.000000 degrees' in text
    print('X-VLA adapter self-check: 4 passed')

if __name__ == '__main__': self_check()
