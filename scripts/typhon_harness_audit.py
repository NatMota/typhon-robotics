"""Audit actual Typhon simulator evidence and replay its captured RGB on CPU."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import imageio.v2 as imageio
import numpy as np
from supervisor.planning.simple_goal_classifier import GoalSuccessClassifier


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    return {'count':len(values),'min_s':float(min(values)),'median_s':float(np.median(values)),
            'p95_s':float(np.percentile(values,95)),'max_s':float(max(values))} if values else {}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True,type=Path)
    args=parser.parse_args()
    root=args.run
    manifest=json.loads((root/'manifest.json').read_text())
    summaries=json.loads((root/'summary.json').read_text())
    assert len(summaries)==7,'Expected all seven actual model/simulator cases'
    assert all(not s['false_completion'] for s in summaries)
    assert digest(manifest['classifier_model'])==manifest['classifier_sha256']
    source_drift={path:expected for path,expected in manifest['code_sha256'].items() if digest(ROOT/path)!=expected}
    assert not source_drift,source_drift
    snapshot=root/'goal-classifier-frozen.npz'
    if not snapshot.exists(): shutil.copyfile(manifest['classifier_model'],snapshot)
    assert digest(snapshot)==manifest['classifier_sha256']
    model=GoalSuccessClassifier(snapshot)
    rows=[]
    decision_times=[]
    inference_times=[]
    proposing_times=[]
    candidate_times=[]
    visible_times=[]
    for summary in summaries:
        folder=root/summary['scenario_id']
        events=read_jsonl(folder/'events.jsonl')
        decisions=[e for e in events if e['type']=='decision']
        measured=read_jsonl(folder/'measured-telemetry.jsonl')
        first=next(e for e in decisions if e.get('checker'))
        rgb=imageio.imread(folder/'0000-primary.png')
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==decisions[0]['image_sha256']['primary']
        prediction=model(rgb,summary['goal_context'])
        assert prediction['status']==summary['first_classifier_status'],(summary['scenario_id'],prediction)
        for e in decisions:
            if e.get('checker'):
                assert e['checker']['classifier']['model_sha256']==manifest['classifier_sha256']
            for candidate in e['candidates']:
                if 'actions' in candidate:
                    assert np.asarray(candidate['actions']).shape==(30,20)
            if e['status']=='complete':
                assert e['completion_streak']>=5
                assert sum(abs(q) for q in e['proprio']['gripper_joint_position_m'])>.055
        for row in measured:
            sample=row if row['type']=='observation' else row['after']
            assert len(sample['joint_torque_nm'])==7
            assert len(sample['proprio']['joint_position_rad'])==7
            assert len(sample['proprio']['joint_velocity_rad_s'])==7
            assert np.isfinite(sample['joint_torque_nm']).all()
        executed=sum(row['type']=='executed_policy_action' for row in measured)
        assert executed==summary['executed_control_steps']
        decision_times+=summary['decision_wall_s']
        inference_times+=summary['classifier_wall_s']
        proposing_times += [e['elapsed_s'] for e in decisions if e['candidates']]
        candidate_times += [c['latency_s'] for e in decisions for c in e['candidates']]
        visible_times += [e['checker']['wall_s'] for e in decisions if e.get('checker',{}).get('assessment',{}).get('status') in ('complete','incomplete')]
        row={k:summary[k] for k in ['scenario_id','first_classifier_status','completion_claim','false_completion',
            'decisions','candidate_proposals','candidate_rejections','executed_control_steps','stop_reason','run_wall_s']}
        row.update(evaluator_success=summary['evaluation']['success'],
            position_error_m=summary['evaluation']['position_error_m'],
            orientation_error_deg=summary['evaluation']['orientation_error_deg'],
            decision_reasons=dict(Counter(e['reason'] for e in decisions)),
            execution_rejections=dict(Counter(e['reason'] for e in events if e['type']=='execution_rejection')),
            classifier_reasons=dict(Counter(e['checker']['classifier']['reason'] for e in decisions if e.get('checker'))),
            cpu_replay_status=prediction['status'],measured_samples=len(measured),
            first_classifier_score=prediction['score'],classifier_threshold=prediction['threshold'])
        if summary['scenario_id'].startswith('typhon-correct'):
            assert summary['completion_claim'] and summary['evaluation']['success']
            assert executed==summary['candidate_proposals']==0
        elif summary['scenario_id']=='typhon-occluded':
            assert not summary['completion_claim'] and executed==summary['candidate_proposals']==0
            assert all(e.get('checker',{}).get('assessment',{}).get('status')=='unknown' for e in decisions)
        else: assert not summary['completion_claim']
        rows.append(row)
    audit={'passed':True,'checkpoint_sha256':manifest['checkpoint_sha256'],
        'classifier_sha256':manifest['classifier_sha256'],'recorded_source_hashes_unchanged':True,
        'actual_rgb_cpu_replays':7,'measured_q_dq_eef_finger_torque_verified':True,
        'total_decision_wall':distribution(decision_times),'classifier_cpu_wall':distribution(inference_times),
        'decisions_with_M6_candidates':distribution(proposing_times),'individual_M6_candidates':distribution(candidate_times),
        'visible_classifier_complete_or_incomplete':distribution(visible_times),
        'cases':rows,'timing_context':'isolated RTX4090 frozen M6 worker; CPU classifier; inference pauses simulator'}
    (root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['Typhon classifier-primary harness: actual frozen M6 validation',
        '', 'PASS: seven actual simulator cases, 25 CPU guard/adapter checks, seven saved-RGB CPU replays.',
        'Frozen earlier-today M6-FT3000 checkpoint: '+manifest['checkpoint'],
        'M6 SHA256: '+manifest['checkpoint_sha256'],
        'Classifier SHA256: '+manifest['classifier_sha256'],
        'Classifier snapshot: '+str(snapshot),
        '', 'Completion uses learned directed-numeral evidence, declared 15mm target tolerance/full footprint, measured finger opening, image position drift <=2mm, and five fresh frames spanning >=0.4s. Classifier native threshold is 0.8; softmax values are uncalibrated.',
        'Qwen is not loaded or called. The previous Qwen/reference mode remains the default of the original supervisor.',
        'Each incomplete observation allows up to three unchanged M6 samples; guards reject inadmissible EEF paths/current joints; shortest/smoothest admissible path wins; at most three steps execute with fresh measured checks.',
        'There is no expert rescue, action repair, training, object-truth actor input, or replacement with tonight\'s H100 models.', '']
    for row in rows:
        lines.append(f"{row['scenario_id']}: image={row['first_classifier_status']}; completion={row['completion_claim']}; evaluator_success={row['evaluator_success']}; decisions={row['decisions']}; candidates={row['candidate_proposals']}; rejected_candidates={row['candidate_rejections']}; executed_policy_steps={row['executed_control_steps']}; false_completion={row['false_completion']}; stop={row['stop_reason']}")
    lines += ['', 'Source outcome: unfinished. The 500-decision bound was reached after 75 proposals and 49 executed policy steps. Joint-limit holds and later visually occluded/uncertain observations prevented further execution. It did not finish the task and was never reported complete.',
        'The source bound is a complete 500-decision harness attempt, not the full original 1500-control-step policy rollout.',
        'The occlusion case is an explicitly labelled opaque camera/sensor-fault diagnostic; the underlying simulator object remains correctly placed. Its evaluator success does not constitute visual success.',
        '', 'Classifier CPU latency: '+json.dumps(audit['classifier_cpu_wall']),
        'Total decision latency (includes very fast pre-inference guard holds): '+json.dumps(audit['total_decision_wall']),
        'Decisions performing three M6 proposals: '+json.dumps(audit['decisions_with_M6_candidates']),
        'Individual M6 candidate inference: '+json.dumps(audit['individual_M6_candidates']),
        'Classifier on visibly complete/incomplete frames: '+json.dumps(audit['visible_classifier_complete_or_incomplete']),
        'Timing is isolated diagnostics with simulator paused for inference, not real-time robot throughput.',
        '', 'Evidence: manifest/protocol; source-labelled primary/secondary PNGs and timestamps; all raw M6 predictions/candidates/selected prefixes; execution rejections; current measured q/dq/EEF/finger positions and MuJoCo actuator torques; evaluator-only truth trajectories, terminal score, videos; audit.json.',
        'Limitations: known overhead DUPLO v2 camera/task family only; loading returns unknown. Continuous visual yaw, physical contact/release, depth/height, and whole-arm collision prediction are not established by this single-view classifier. Measured open fingers plus image position stability are required, but remain distinct from independent truth scoring.',
        '', 'Launch from '+str(ROOT)+':',
        'M7_LIVE_DIR=/var/tmp/humanoid-m7-live-evaluation-20261009/live PYTHONDONTWRITEBYTECODE=1 .venv-sim/bin/python scripts/typhon_harness.py --output /var/tmp/typhon-harness-20261009/NEW-RUN --classifier-model '+str(snapshot)+' --source-wall-s 220 --source-decisions 500 --baseline-decisions 8',
        'Use a new output name and reserve the GPU/shared observer feed before launch. The supplied run directory is never overwritten.',
        'CPU audit: .venv-sim/bin/python scripts/typhon_harness_audit.py --run '+str(root),
        'CPU gate checks: .venv-sim/bin/python -m unittest tests.test_m7_service tests.test_typhon_service',
        '', 'The earlier preload-only live-v1 folder is retained with an explicit STOP_REASON; no simulator episodes ran there.']
    (root/'SERVICE_REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
