"""Summarize the fixed post-recovery panels without combining distinct protocols."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, inputs = [], {}

    def read(relative):
        path = args.evidence / relative
        raw = path.read_bytes()
        inputs[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    native = []
    for variant in ('v1-m5', 'v1-m6', 'v2-m5', 'v2-m6'):
        run = variant + '-native-panel-v3'
        summary = read(run + '/summary.json')
        assert summary['recorded'] == 8, (run, 'panel incomplete')
        assert len({r['scenario_id'] for r in summary['results']}) == 8
        panel = {'run': run, 'unfinished_successes': 0, 'unfinished_count': 0,
                 'preserved': 0, 'completed_count': 0, 'execution_errors': 0}
        for result in summary['results']:
            completed = result['group'] == 'completed'
            panel['completed_count' if completed else 'unfinished_count'] += 1
            panel['preserved' if completed else 'unfinished_successes'] += int(result['success'])
            panel['execution_errors'] += int(result['status'] != 'evaluated' or any(result.get(k) for k in ('controller_exceptions','artifact_errors','evaluation_errors')))
            rows.append(dict(run=run, requested_horizon_s=1.5, scenario_id=result['scenario_id'], group=result['group'],
                success=result['success'], position_error_mm=result['position_error_mm'],
                yaw_error_deg=result['yaw_error_deg'], policy_actions=result['policy_actions_executed'],
                completion_claim=None, false_completion=None))
        native.append(panel)

    supervised = []
    for run in ('v3-m6-4s-live-v3', 'v3-m6-tuned-live-v2', 'v2-m6-supervised-live-v1'):
        cases = read(run + '/summary.json')
        manifest = read(run + '/manifest.json')
        assert len(cases) == 4
        for case in cases:
            score = case['evaluation']
            scenario = read(run + '/' + case['scenario_id'] + '/scenario.json')
            events_path = args.evidence / run / case['scenario_id'] / 'events.jsonl'
            reasons = Counter(json.loads(line).get('reason', 'other') for line in events_path.read_text().splitlines())
            row = dict(run=run, requested_horizon_s=manifest['horizon_s'], scenario_id=case['scenario_id'], group=scenario['group'],
                success=score['success'], position_error_mm=1000*score['position_error_m'],
                yaw_error_deg=score['orientation_error_deg'], policy_actions=case['executed_control_steps'],
                completion_claim=case['completion_claim'], false_completion=case['false_completion'])
            rows.append(row)
            supervised.append(dict(row, stop_reason=case['stop_reason'], runtime_errors=case.get('runtime_errors', []),
                reasons=dict(reasons), mean_inference_latency_s=case.get('mean_inference_latency_s')))

    expert = read('expert-reachability-v1/summary.json')
    frozen = read('motion-review-v3/validation_summary.json')
    summary = {'schema': 'typhon-recovery-review-v1', 'native_common_1_5s': native,
        'supervised_panels': supervised, 'privileged_expert_diagnostic': expert,
        'frozen_motion_review': frozen, 'inputs_sha256': inputs,
        'protocol_limit': 'Common 1.5-second full-intent requests can differ from a variant training request distribution. Four-second diagnostics are separate. Completed scenes measure preservation, not manipulation.',
        'data_limit': 'Personal observations configure scenarios. The action-supervised RGB/action pairs are robot simulation plus geometric expert; raw phone frames are not directly aligned robot-action supervision.',
        'excluded_technical_attempts': ['v1-m5-native-panel-v1: missing PEFT', 'v1-m5-native-panel-v2: PEFT version incompatible with strict checkpoint keys', 'v3-m6-4s-live-v1: privileged cue not enabled', 'v3-m6-4s-live-v2: reset observation clock mismatch', 'v3-m6-tuned-live-v1: terminal control budget overrun'],
        'no_training': True, 'simulation_pauses_during_inference': True}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'validation_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (args.output/'episodes.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'native_panels':native,'supervised_cases':len(supervised),'expert_successes':sum(r['success'] for r in expert['results'])}))


if __name__ == '__main__':
    main()
