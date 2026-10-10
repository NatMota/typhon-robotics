"""Reanalyse frozen common M7 episodes; never run policies or change scoring."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def angle_deg(a, b):
    return abs(math.degrees(math.atan2(math.sin(a - b), math.cos(a - b))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=Path.cwd())
    ap.add_argument('--evidence', type=Path, default=Path('/var/tmp/typhon-harness-20261010'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    history = args.repo / 'experiment-history/recovery-20261010'
    entries = []
    for row in csv.DictReader((history / 'frozen-motion-audit.csv').open()):
        if row['scope'] == 'common_1.5s':
            entries.append((row['model'], Path(row['evidence_directory']), row['episode_sha256']))
    for row in csv.DictReader((history / 'episodes.csv').open()):
        if 'native-panel' in row['run']:
            model = '-'.join(row['run'].split('-')[:2])
            entries.append((model, args.evidence / row['run'] / row['scenario_id'], None))
    assert len(entries) == 128 and len(set((m, p.name) for m, p, _ in entries)) == 128
    rows, source_hashes = [], {}
    for model, folder, expected_hash in entries:
        paths = [folder / name for name in ('episode.json', 'scenario.json', 'trajectory.jsonl.gz')]
        for path in paths:
            source_hashes[str(path)] = sha(path)
        if expected_hash:
            assert source_hashes[str(paths[0])] == expected_hash, paths[0]
        ep, scenario = read(paths[0]), read(paths[1])
        cfg = scenario['config']; criteria = ep['evaluation']['criteria']
        with gzip.open(paths[2], 'rt') as stream:
            truths = [ep['initial_truth']] + [json.loads(line)['truth'] for line in stream]
        initial = ep['initial_truth']; target = cfg['task']
        initial_position_error = 1000 * math.dist(initial['object_position_m'][:2], target['target_position_xy_m'])
        initial_yaw_error = angle_deg(initial['object_yaw_rad'], target['target_yaw_rad'])
        peak_lift = max(t['object_position_m'][2] - initial['object_position_m'][2] for t in truths)
        contact_steps = sum(t['grasp_contact'] for t in truths[1:])
        bilateral_lift_steps = sum(t['bilateral_grasp_contact'] and t['object_position_m'][2] - initial['object_position_m'][2] >= .07 for t in truths[1:])
        attempt_lift = any(a.get('verified_lift') and a.get('any_contact') for a in ep['attempts'])
        pickup = bool(attempt_lift and bilateral_lift_steps)
        # Check this stricter contact-supported diagnostic agrees with saved lift outcomes.
        assert pickup == bool(ep['evaluation']['successful_grasp_count']), (model, folder.name)
        max_displacement = 1000 * max(math.dist(t['object_position_m'], initial['object_position_m']) for t in truths)
        max_yaw_change = max(angle_deg(t['object_yaw_rad'], initial['object_yaw_rad']) for t in truths)
        transfer = ep['group'] not in ('completed', 'correction')
        if transfer:
            assert initial_position_error > 1000 * cfg['evaluation']['position_tolerance_m']
        physical_placement = all(criteria[k] for k in ('position', 'marked_face_up', 'footprint_inside_target', 'placement_height', 'released', 'stable'))
        rows.append(dict(
            model=model, scenario_id=ep['scenario_id'], group=ep['group'], task=ep['task'],
            transfer_required=transfer, pickup=pickup, bilateral_lift_steps=bilateral_lift_steps,
            new_placement=bool(transfer and physical_placement), physical_placement=physical_placement,
            full_success=ep['success'], orientation_pass=criteria['orientation'],
            initial_position_error_mm=initial_position_error, final_position_error_mm=ep['position_error_mm'],
            initial_yaw_error_deg=initial_yaw_error, final_yaw_error_deg=ep['yaw_error_deg'],
            yaw_error_reduction_deg=initial_yaw_error - ep['yaw_error_deg'],
            peak_lift_mm=1000 * peak_lift, max_object_displacement_mm=max_displacement,
            max_object_yaw_change_deg=max_yaw_change, contact_steps=contact_steps,
            object_undisturbed=bool(not pickup and max_displacement <= 2 and max_yaw_change <= 2),
            released=criteria['released'], stable=criteria['stable'],
            eef_path_m=sum(math.dist(a['eef_position_m'], b['eef_position_m']) for a, b in zip(truths, truths[1:])),
            evidence_directory=str(folder), episode_sha256=source_hashes[str(paths[0])],
        ))
    summaries = []
    for model in sorted({r['model'] for r in rows}):
        rr = [r for r in rows if r['model'] == model]
        active = [r for r in rr if r['group'] != 'completed']
        transfers = [r for r in rr if r['transfer_required']]
        completed = [r for r in rr if r['group'] == 'completed']
        corrections = [r for r in rr if r['group'] == 'correction']
        assert (len(rr), len(active), len(transfers), len(completed), len(corrections)) == (8, 6, 5, 2, 1)
        summaries.append(dict(
            model=model, unfinished_n=6, verified_pickups=sum(r['pickup'] for r in active),
            transfer_n=5, new_placements=sum(r['new_placement'] for r in transfers),
            mean_initial_yaw_error_deg=statistics.mean(r['initial_yaw_error_deg'] for r in active),
            mean_final_yaw_error_deg=statistics.mean(r['final_yaw_error_deg'] for r in active),
            mean_yaw_error_reduction_deg=statistics.mean(r['yaw_error_reduction_deg'] for r in active),
            mean_final_position_error_mm=statistics.mean(r['final_position_error_mm'] for r in active),
            orientation_passes=sum(r['orientation_pass'] for r in active),
            full_unfinished_successes=sum(r['full_success'] for r in active),
            completed_n=2, completed_without_pickup=sum(not r['pickup'] for r in completed),
            completed_without_object_disturbance=sum(r['object_undisturbed'] for r in completed),
            completed_final_score_passes=sum(r['full_success'] for r in completed),
            correction_final_yaw_error_deg=corrections[0]['final_yaw_error_deg'],
        ))
    old = read(args.repo / 'frozen_results/personal-m2-20261009/scenarios.json')['primary']
    latest_cfg = read(entries[0][1] / 'scenario.json')['config']
    assert all(sc['config']['evaluation'] == latest_cfg['evaluation'] for sc in old)
    assert all(read(folder / 'scenario.json')['config']['evaluation'] == latest_cfg['evaluation'] for _, folder, _ in entries)
    for path, digest in source_hashes.items():
        assert sha(Path(path)) == digest, path
    result = dict(
        schema='typhon-component-review-v1', episode_count=len(rows), checkpoint_count=len(summaries),
        scope='Saved common 1.5-second native policy panels only; no harness, separate 4-second diagnostic, assisted or partial runs pooled.',
        metric_definitions={
            'pickup': 'Saved verified lift plus contact during the attempt, with bilateral contact independently observed at least 70 mm above initial height. No gripper-close-only credit.',
            'new_placement': 'Initially outside target; final XY <=15 mm, full footprint inside target, face up, supported at correct height, released and stable. Yaw success is scored separately.',
            'orientation': 'Absolute directed yaw error [0,180] degrees; <=10 degrees passes. Mean across six unfinished cases, including the correction case.',
            'completed_without_pickup': 'No verified pickup in the two initially completed cases; does not establish learned completion recognition or absence of arm motion.',
            'object_undisturbed': 'New trajectory diagnostic: displacement never exceeds 2 mm and yaw change never exceeds 2 degrees, with no verified pickup. Not a replacement for frozen success.',
            'improvement': 'Initial-to-final error reduction describes within-episode progress, not improvement against a missing matched vanilla baseline.'},
        suite_comparison={
            'earlier': '40/model: 20 feeding and 20 loading, standard and varied goals; matched vanilla available.',
            'latest': '8/model: 5 transfer starts (1 source, 2 held-out personal, 2 synthetic), 1 wrong-orientation, 2 completed; feeding only. Common full-intent 1.5-second request; no matched vanilla.',
            'unchanged': '15 mm position / 10 degree yaw criteria and other physical success tolerances; 1500 total steps at 20 Hz. Same provisional 50 x 35 x 30 mm, 80 g physical body.',
            'limitations': 'Small diagnostic panel; whole-intent wording at 1.5 seconds is a declared request-distribution shift. V2-V4 use an explicit held-state annotation condition. Not a definitive variant ranking.'},
        summaries=summaries, rows=rows, source_sha256=source_hashes,
        reproduction='python3 scripts/typhon_component_review.py --output /var/tmp/typhon-component-review-reproduced',
        frozen_sources_verified_unchanged=True,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'component-metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    for filename, data in [('component-episodes.csv', rows), ('component-summary.csv', summaries)]:
        with (args.output / filename).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(data)
    print(json.dumps({'episodes': len(rows), 'checkpoints': len(summaries), 'M6': [r for r in summaries if r['model'].endswith('m6')], 'sources_unchanged': True, 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
