"""Inference-only text bridge. No model weights or frozen evaluation code change."""
import json
import os
import pickle
import select
import subprocess
import time
from pathlib import Path

SYSTEM = '''You are a visual task planner for a simulated Franka Panda.
You receive current overhead and oblique RGB views and a supplied task goal.
Describe only visible evidence. The orange DUPLO has a blue numeral 2.
Do not invent object poses, contact, grasp success, or hidden orientation.
Choose one next short manipulation instruction that serves the supplied goal.
EXECUTE means request motion from an existing X-VLA policy, not guarantee success.
WAIT means hold the current robot pose while uncertainty remains.
FINISH is allowed only if the current views clearly show the requested final placement
and orientation, with the object released. If uncertain, choose WAIT.
Do not prescribe joints, coordinates, code, or numeric motor actions.
Return exactly one JSON object, no markdown, with exactly these keys:
"decision": "execute", "wait", or "finish";
"description": a short visible-scene description, at most 400 characters;
"instruction": one next robot instruction, at most 25 words (empty unless execute).
A previous instruction is not evidence that it was executed successfully.'''


def validate_plan(raw):
    text = raw.strip()
    if text.startswith('```json') and text.endswith('```'):
        text = text[7:-3].strip()
    plan = json.loads(text)
    if not isinstance(plan, dict) or set(plan) != {'decision', 'description', 'instruction'}:
        raise ValueError('Unexpected planner schema')
    if plan['decision'] not in ('execute', 'wait', 'finish'):
        raise ValueError('Unknown decision')
    if not isinstance(plan['description'], str) or len(plan['description']) > 400:
        raise ValueError('Invalid description')
    instruction = plan['instruction']
    if not isinstance(instruction, str) or len(instruction.split()) > 25 or len(instruction) > 250:
        raise ValueError('Invalid instruction')
    if bool(instruction.strip()) != (plan['decision'] == 'execute'):
        raise ValueError('Instruction must be nonempty exactly for execute')
    return plan


def instruction_observation(observation, plan):
    """Append a subgoal to language; preserve the supplied target and all sensor data."""
    validate_plan(json.dumps(plan))
    if plan['decision'] != 'execute':
        raise ValueError('WAIT/FINISH must bypass the VLA, not become motor instructions')
    task = dict(observation['task'])
    task['instruction'] += ' Next step: ' + plan['instruction']
    return {**observation, 'task': task}


class QwenPlanner:
    def __init__(self, root, model, log_path):
        root = Path(root)
        self.log = open(log_path, 'a')
        self.process = subprocess.Popen([
            str(root / '.venv-qwen-planner/bin/python'), '-m',
            'supervisor.planning.qwen_worker', '--model', str(Path(model).resolve())],
            cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            env=dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false'))
        try:
            self.load_info = self.receive(300)
            if not self.load_info.get('ready'):
                raise RuntimeError(self.load_info)
        except Exception:
            self.close()
            raise

    def receive(self, timeout=120):
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            self.process.kill()
            raise TimeoutError('Qwen worker timed out')
        result = pickle.load(self.process.stdout)  # trusted child process only
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def plan(self, observation, previous=None):
        # Explicit whitelist: no evaluator truth, object poses, or future trajectory.
        from supervisor.baseline.xvla import task_prompt
        payload = {'op': 'plan', 'images': [observation['rgb']['primary'], observation['rgb']['secondary']],
                   'goal': task_prompt(observation['task']), 'previous_instruction': previous or ''}
        start = time.perf_counter()
        pickle.dump(payload, self.process.stdin, protocol=4)
        self.process.stdin.flush()
        result = self.receive()
        try:
            result['plan'] = validate_plan(result['raw_text'])
            result['schema_valid'] = True
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            result['plan'] = {'decision': 'wait', 'description': 'Invalid planner output; hold.', 'instruction': ''}
            result['schema_valid'] = False
            result['validation_error'] = str(exc)
        result['wall_s'] = time.perf_counter() - start
        return result

    def close(self):
        if self.process.poll() is None:
            try:
                pickle.dump({'op': 'close'}, self.process.stdin, protocol=4)
                self.process.stdin.flush()
                self.process.wait(timeout=15)
            except Exception:
                self.process.kill()
                self.process.wait(timeout=15)
        self.log.close()


def self_check():
    plan = validate_plan('{"decision":"execute","description":"Orange block outside target","instruction":"Grasp the orange block."}')
    obs = {'task': {'instruction': 'Place the block.', 'target_position_xy_m': [.1, .2], 'target_yaw_rad': 0}, 'rgb': object()}
    changed = instruction_observation(obs, plan)
    assert changed['rgb'] is obs['rgb'] and obs['task']['instruction'] == 'Place the block.'
    assert changed['task']['target_position_xy_m'] == [.1, .2]
    for bad in ('{}', '{"decision":"run_code","description":"x","instruction":"x"}',
                '{"decision":"finish","description":"x","instruction":"Move"}'):
        try:
            validate_plan(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    print('Planner schema, goal preservation, and no-motion decision contracts passed')


if __name__ == '__main__':
    self_check()
