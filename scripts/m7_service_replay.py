"""CPU-only replay of recorded real Qwen/X-VLA outputs through current guards.

No model calls or robot execution. Preserves earlier negative results, explicitly
separating final guard verification from the live model/simulator smoke.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imageio.v2 as imageio
import numpy as np
from supervisor.planning.m7_service import ServiceConfig, SupervisorService, reference_agreement


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for folder in sorted(path for path in a.source.iterdir() if (path/'events.jsonl').exists()):
        records = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        config = ServiceConfig(**records[0]['config'])
        clock = [100.]
        active = {}
        reference = {camera: imageio.imread(folder/f'goal-{camera}.png') for camera in ('primary', 'secondary')}
        def checker(observation, reference, deadline):
            result = copy.deepcopy(active['record']['checker'])
            result['reference_agreement'] = reference_agreement(observation['rgb']['primary'], reference['primary'])
            return result
        def proposer(observation, deadline):
            value = next(active['candidates'])
            clock[0] += value['latency_s']
            return np.asarray(value['actions'])
        service = SupervisorService(config, checker, proposer, a.output/(folder.name+'.jsonl'), clock=lambda: clock[0])
        service.reset(folder.name)
        decisions = []
        for record in (item for item in records if item['type'] == 'decision'):
            active.update(record=record, candidates=iter(record['candidates']))
            index = record['index']
            observation = {'rgb': {camera: imageio.imread(folder/f'{index:03d}-{camera}.png') for camera in ('primary', 'secondary')},
                           'proprio': record['proprio'], 'task': record['goal']}
            clock[0] += .2
            decision = service.step(observation, reference, record['frame_id'], clock[0])
            decisions.append({'status': decision['status'], 'reason': decision['reason'],
                              'reference_agreement': decision['event'].get('checker', {}).get('reference_agreement')})
        expected = 'complete' if folder.name == 'recovery-already_completed-01' else 'execute' if folder.name == 'm5-source-01' else 'hold'
        passed = decisions[-1]['status'] == expected
        if folder.name == 'service-negative-yaw180-in-target':
            passed = passed and all(row['reason'] == 'reference_agreement_veto' for row in decisions)
        rows.append({'scenario_id': folder.name, 'expected_final_status': expected, 'passed': passed,
                     'replay_only_no_robot_execution': True, 'decisions': decisions})
        print(folder.name, 'PASS' if passed else 'FAIL', decisions[-1]['status'], flush=True)
    # A fully hidden numeral must produce explicit unknown rather than fake orientation.
    reference = {camera: imageio.imread(a.source/'recovery-already_completed-01'/f'goal-{camera}.png') for camera in ('primary', 'secondary')}
    hidden = np.full_like(reference['primary'], 128)
    hidden_result = reference_agreement(hidden, reference['primary'])
    summary = {'passed': all(row['passed'] for row in rows) and hidden_result['status'] == 'unknown',
               'cases': rows, 'hidden_numeral_veto': hidden_result,
               'type': 'recorded model-output replay with current guard code; no new Qwen inference or robot execution'}
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    assert summary['passed'], summary


if __name__ == '__main__':
    main()
