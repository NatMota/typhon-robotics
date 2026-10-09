"""Independent structural M0 validation; agent review is never human approval."""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_format', '-show_streams',
        '-of', 'json', str(path),
    ]))


def validate(root, stage):
    data = root / 'data/personal/m0'
    results = root / 'results/personal-m0-20261009'
    errors, pending, checks = [], [], []
    source_pts, pts_source_audit = {}, []
    semantic_review = {'status': 'not_checked_inventory_stage'}
    semantic_ok = False
    excluded_validation = {'declared_count': 0, 'preserved_count': 0}
    pts_sample_errors, pts_evidence_errors = [], []

    def require(condition, message):
        (checks if condition else errors).append(message)

    def required(path):
        if path.exists():
            return True
        pending.append(str(path.relative_to(root)))
        return False

    manifest = rows(data / 'raw_manifest.csv')
    original = json.loads((data / 'source_inventory_local.json').read_text())
    sources = {row['recording_id']: row for row in manifest}
    require(len(manifest) == len(sources) == len(original) == 8,
            'Eight unique raw recording IDs accounted for')
    expected = {row['recording_id']: row for row in original}
    checksum_lines = (data / 'source_checksums.sha256').read_text().splitlines()
    checksums = dict(line.split('  ', 1)[::-1] for line in checksum_lines if line)
    for rid, source in sources.items():
        path = root / source['relative_path']
        require(path.is_file(), f'{rid}: preserved raw source exists')
        if not path.is_file():
            continue
        require(source['filename'] == expected[rid]['filename'], f'{rid}: original filename retained')
        require(path.stat().st_size == int(source['file_size_bytes']) == expected[rid]['bytes'],
                f'{rid}: original file size unchanged')
        require(digest(path) == source['sha256'] == expected[rid]['sha256'] ==
                checksums.get('raw/' + source['filename']), f'{rid}: original SHA-256 unchanged')
        measured = probe(path)
        require(abs(float(measured['format']['duration']) - float(source['duration_s'])) < .001,
                f'{rid}: source duration agrees with ffprobe')
        pts_probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
            '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(path)]))
        raw_pts = [frame.get('best_effort_timestamp_time') for frame in pts_probe['frames']]
        require(all(value is not None for value in raw_pts), f'{rid}: every decoded video frame has actual PTS')
        pts = [float(value) if value is not None else float('nan') for value in raw_pts]
        source_pts[rid] = pts
        require(bool(pts) and all(math.isfinite(t) for t in pts) and all(a <= b for a, b in zip(pts, pts[1:])),
                f'{rid}: independently decoded source PTS are finite and monotonic')
        audit = {'recording_id': rid, 'decoded_source_frame_count': len(pts),
                 'sampled_frame_count': 0, 'sampled_pts_max_error_s': 0.0}
        pts_source_audit.append(audit)
        timestamps = results / 'inspection' / rid / 'sample_timestamps.json'
        if required(timestamps):
            samples = json.loads(timestamps.read_text())
            times = [float(sample['source_time_s']) for sample in samples]
            audit['sampled_frame_count'] = len(samples)
            for sample in samples:
                index = sample['decoded_frame_index']
                valid_index = isinstance(index, int) and 0 <= index < len(pts)
                require(valid_index, f'{rid}: sampled decoded frame index resolves in independently decoded PTS')
                if valid_index:
                    error = abs(float(sample['source_time_s']) - pts[index])
                    pts_sample_errors.append(error)
                    audit['sampled_pts_max_error_s'] = max(audit['sampled_pts_max_error_s'], error)
                    require(error <= 1e-5, f'{rid}: sampled source time agrees with actual frame PTS at decoded index')
            require(bool(times) and all(math.isfinite(t) and 0 <= t <= float(source['duration_s'])
                                       for t in times), f'{rid}: sample timestamps within original source')
            require(all(a < b for a, b in zip(times, times[1:])),
                    f'{rid}: actual sample timestamps strictly increasing')
            require(all((timestamps.parent / sample['frame']).is_file() for sample in samples),
                    f'{rid}: all sampled-frame artifacts exist')

    demos = []
    human_review_types = set()
    source_tracking = {}
    if stage == 'final':
        for name in ['camera_pairs.csv', 'trial_annotations.csv', 'quality_report.csv',
                     'trajectories.csv', 'dataset_manifest.json']:
            required(data / name)
        pairs_path = data / 'camera_pairs.csv'
        pairs = {p['pair_id']: p for p in rows(pairs_path)} if pairs_path.exists() else {}
        metadata_files = sorted((data / 'demonstrations').glob('demo_*/metadata.json'))
        require(bool(metadata_files), 'At least one traceable manipulation demonstration exported')
        ids = set()
        for path in metadata_files:
            demo = json.loads(path.read_text())
            demos.append(demo)
            did = demo['demo_id']
            require(did == path.parent.name and did not in ids, f'{did}: stable unique directory/metadata ID')
            ids.add(did)
            require(isinstance(demo.get('manipulation_attempt'), bool),
                    f'{did}: manipulation occurrence declared independently from task-category count')
            for view in ['camera_a', 'camera_b']:
                camera = demo.get(view)
                if camera is None:
                    pending.append(f'{did}: {view} unresolved')
                    continue
                source = sources.get(camera['recording_id'])
                require(source is not None, f'{did}/{view}: source recording ID resolves')
                if source is None:
                    continue
                start, end = float(camera['start_s']), float(camera['end_s'])
                require(0 <= start < end <= float(source['duration_s']) + .02,
                        f'{did}/{view}: clip bounds within original source')
                require(camera['filename'] == source['filename'], f'{did}/{view}: exact original filename traceable')
                clip = Path(camera['clip'])
                candidates = [root / clip, path.parent / clip, data / clip]
                clip_path = next((candidate for candidate in candidates if candidate.is_file()), None)
                require(clip_path is not None, f'{did}/{view}: exported clip exists')
                if clip_path:
                    clip_probe = probe(clip_path)
                    actual = float(clip_probe['format']['duration'])
                    require(abs(actual - (end - start)) <= .12,
                            f'{did}/{view}: decoded clip duration agrees with source interval')
                    stream = next(s for s in clip_probe['streams'] if s['codec_type'] == 'video')
                    transform = camera.get('clip_transform')
                    expected_transform = 'rotate_90_degrees_counterclockwise' if view == 'camera_a' else 'none'
                    require(transform == expected_transform and
                            camera.get('source_coordinates') == 'original_decoded_image',
                            f'{did}/{view}: source coordinates and derived transform explicit')
                    width, height = int(source['width']), int(source['height'])
                    expected_dimensions = (height, width) if view == 'camera_a' else (width, height)
                    require((stream['width'], stream['height']) == expected_dimensions,
                            f'{did}/{view}: encoded clip dimensions agree with declared rotation')
            if demo.get('camera_a') and demo.get('camera_b'):
                a, b, offset = demo['camera_a'], demo['camera_b'], float(demo['sync_offset_s'])
                require(abs(float(b['start_s']) - float(a['start_s']) - offset) < .05 and
                        abs(float(b['end_s']) - float(a['end_s']) - offset) < .05,
                        f'{did}: secondary = primary + offset at both clip endpoints')
                pair = pairs.get(demo['pair_id'])
                require(pair is not None, f'{did}: declared camera pair resolves')
                if pair:
                    require(a['recording_id'] == pair['primary_recording_id'] and
                            b['recording_id'] == pair['secondary_recording_id'] and
                            abs(offset - float(pair['offset_s'])) < .0001,
                            f'{did}: source IDs and signed offset agree with pairing manifest')
            require(demo.get('review_type') == 'agent_visual_review_not_human_review' or
                    demo.get('human_reviewed') is True,
                    f'{did}: reviewer provenance explicit')
            for name, image in demo.get('initial_final_frames', {}).items():
                require((root / image['path']).is_file(), f'{did}: {name} evidence frame exists')
                camera = demo['camera_a' if name.startswith('camera_a') else 'camera_b']
                decoded = float(image['decoded_source_time_s'])
                pts = source_pts.get(camera['recording_id'], [])
                insertion = bisect.bisect_left(pts, decoded)
                nearest = [pts[index] for index in (insertion - 1, insertion) if 0 <= index < len(pts)]
                error = min((abs(decoded - actual) for actual in nearest), default=float('inf'))
                pts_evidence_errors.append(error)
                require(error <= 1e-5, f'{did}: {name} evidence source time matches independently decoded actual PTS')
                require(float(camera['start_s']) - .04 <= decoded <= float(camera['end_s']) + .04,
                        f'{did}: {name} decoded evidence timestamp within original clip interval')
            tracking = path.parent / 'tracking.csv'
            if required(tracking):
                track_rows = rows(tracking)
                require(bool(track_rows), f'{did}: observation tracking rows present')
                previous_times = {}
                for row in track_rows:
                    require(row.get('demo_id') == did, f'{did}: tracking row ID traceable')
                    view = row.get('view', '')
                    camera = demo.get(view) or demo.get({'a': 'camera_a', 'b': 'camera_b',
                                                       'primary': 'camera_a', 'secondary': 'camera_b'}.get(view, ''))
                    if not camera:
                        errors.append(f'{did}: unrecognized tracking view {view!r}')
                        continue
                    source_time = row.get('source_time_s', row.get('time_s', ''))
                    if source_time:
                        t = float(source_time)
                        require(float(camera['start_s']) - .04 <= t <= float(camera['end_s']) + .04,
                                f'{did}/{view}: tracking source timestamp within exported bounds')
                        require(abs(float(row['clip_time_s']) - (t - float(camera['start_s']))) < .002,
                                f'{did}/{view}: clip timestamp derived from original timestamp')
                        require(t > previous_times.get(view, -math.inf),
                                f'{did}/{view}: original tracking timestamps strictly increasing')
                        previous_times[view] = t
                        rid = camera['recording_id']
                        if rid not in source_tracking:
                            source_csv = results / 'tracking_sources' / (rid + '.csv')
                            source_tracking[rid] = {float(r['source_time_s']): r for r in rows(source_csv)} if required(source_csv) else {}
                        observed = source_tracking[rid].get(t)
                        require(observed is not None,
                                f'{did}/{view}: every observation timestamp traces to original sampled detection')
                        if observed:
                            require(all(row.get(field) == value for field, value in observed.items()),
                                    f'{did}/{view}: original measured fields retained without interpolation or relabeling')
                    for key in ['interpolated', 'is_interpolated', 'position_interpolated']:
                        if key in row:
                            require(row[key].lower() not in ['1', 'true', 'yes'],
                                    f'{did}: no silently interpolated observation measurement')
                    detected = row.get('detected', '').lower() in ['true', '1', 'yes']
                    measurements = ['centre_x_px', 'centre_y_px', 'bbox_x_px', 'bbox_y_px',
                                    'canonical_centre_x_px', 'canonical_centre_y_px',
                                    'orientation_raw_clockwise_deg', 'orientation_canonical_clockwise_deg']
                    if not detected:
                        require(all(row.get(field, '') == '' for field in measurements),
                                f'{did}/{view}: missing detection retains unknown object measurements')
                    for field in measurements:
                        if row.get(field, ''):
                            require(math.isfinite(float(row[field])),
                                    f'{did}/{view}: {field} finite when observed')
                    if row.get('tracking_confidence', ''):
                        require(0 <= float(row['tracking_confidence']) <= 1,
                                f'{did}/{view}: heuristic confidence within declared range')
                    raw_angle = row.get('orientation_raw_clockwise_deg', '')
                    canonical_angle = row.get('orientation_canonical_clockwise_deg', '')
                    if canonical_angle:
                        require(bool(raw_angle) and abs((float(canonical_angle) -
                                (float(raw_angle) - 90) % 360)) < .001,
                                f'{did}/{view}: canonical orientation derives from explicit CCW90 rotation')
                    if not canonical_angle:
                        require(row.get('orientation_correct_canonical_primary', '') == '',
                                f'{did}/{view}: unknown angle cannot invent orientation correctness')
                    if row.get('canonical_centre_x_px', ''):
                        source = sources[camera['recording_id']]
                        require(abs(float(row['canonical_centre_x_px']) - float(row['centre_y_px'])) < .002 and
                                abs(float(row['canonical_centre_y_px']) -
                                    (int(source['width']) - 1 - float(row['centre_x_px']))) < .002,
                                f'{did}/{view}: canonical position derives from explicit CCW90 rotation')

        dataset_manifest = data / 'dataset_manifest.json'
        if dataset_manifest.exists():
            derived = json.loads(dataset_manifest.read_text())
            require(derived.get('recording_count') == 8, 'Derived manifest accounts for eight raw sources')
            listed = derived.get('demonstrations', [])
            listed_ids = [item['demo_id'] if isinstance(item, dict) else item for item in listed]
            require(len(listed_ids) == len(set(listed_ids)) and set(listed_ids) == ids,
                    'Derived manifest exactly accounts for stable demonstration IDs')
            require(derived.get('demonstration_count') == len(demos),
                    'Derived demonstration count agrees with actual exported metadata')
            require(derived.get('manipulation_attempt_count') == sum(d.get('manipulation_attempt') is True for d in demos),
                    'Derived manipulation count agrees with explicit episode occurrence flags')
            require(derived.get('canonical_view') == 'primary_rotated_90_degrees_counterclockwise_user_confirmed',
                    'Canonical camera rotation explicitly declared as user-confirmed derived transform')
        semantic_path = results / 'review/task_semantics_review.json'
        if semantic_path.exists():
            semantic_review = json.loads(semantic_path.read_text())
            semantic_ok = semantic_review.get('status') == 'reviewed' and not semantic_review.get('unresolved_recording_ids', [])
            labels = semantic_review.get('episode_labels', {})
            allowed_types = {'feeding', 'normal_feeding', 'target_local_reorientation',
                             'orientation_correction', 'awkward_pause_resume_feeding', 'already_completed'}
            require(isinstance(labels, dict), 'Task semantic episode labels have explicit ID/type mapping')
            if isinstance(labels, dict):
                require(set(labels) <= ids, 'Every explicitly labeled semantic episode exists in active metadata')
                for did, label in labels.items():
                    require(label in allowed_types, f'{did}: reviewed semantic label is supported')
                    demo = next((d for d in demos if d['demo_id'] == did), None)
                    if demo:
                        require(demo['attempt_type'] == label, f'{did}: active annotation agrees with explicit semantic review label')
            selected_expected = semantic_review.get('selected_expected_counts', {})
            selected_counts = {}
            for demo in demos:
                category = demo.get('task_category')
                selected_counts[category] = selected_counts.get(category, 0) + 1
            for category, expected_count in selected_expected.items():
                require(selected_counts.get(category, 0) == expected_count,
                        f'Selected {category} count agrees with resolved user review, not initial suggested count')
            excluded_ids = set(semantic_review.get('excluded_episode_ids', []))
            require(not (excluded_ids & ids), 'Explicitly excluded episodes are outside curated active metadata')
            excluded_path = data / 'excluded_demo_manifest.json'
            excluded_manifest = json.loads(excluded_path.read_text()) if excluded_path.exists() else {'excluded_demonstrations': []}
            excluded_records = excluded_manifest.get('excluded_demonstrations', [])
            preserved_ids = [d['demo_id'] for d in excluded_records]
            require(set(preserved_ids) == excluded_ids and len(preserved_ids) == len(set(preserved_ids)),
                    'Excluded demonstration manifest exactly preserves each user-selected exclusion')
            if excluded_ids:
                require(excluded_manifest.get('review_file') == 'results/personal-m0-20261009/review/task_semantics_review.json',
                        'Excluded manifest links explicit user selection provenance')
                require(bool(semantic_review.get('selection_reason')), 'Excluded episodes have an explicit user selection reason')
            excluded_validation = {'declared_count': len(excluded_ids), 'preserved_count': len(excluded_records),
                'excluded_ids': sorted(excluded_ids), 'selected_expected_counts': selected_expected,
                'observed_selected_counts': selected_counts}
            for excluded in excluded_records:
                did = excluded['demo_id']
                require(excluded.get('excluded_from_active_dataset') is True and bool(excluded.get('exclusion_reason')),
                        f'{did}: excluded status and reason recorded, not relabeled as failure')
                folder = (root / excluded['camera_a']['clip']).parent
                require(folder.is_relative_to(data / 'excluded_demonstrations'), f'{did}: excluded artifacts reside outside curated set')
                metadata_path = folder / 'metadata.json'
                require(metadata_path.is_file(), f'{did}: excluded reviewed metadata retained')
                if metadata_path.exists():
                    retained = json.loads(metadata_path.read_text())
                    require(retained['demo_id'] == did and retained.get('excluded_from_active_dataset') is True,
                            f'{did}: manifest resolves to explicitly excluded metadata')
                    require(all(retained.get(view) == excluded.get(view) for view in ['camera_a', 'camera_b']),
                            f'{did}: retained excluded metadata and manifest have identical source/clip bounds')
                    require(retained.get('manual_annotations') == excluded.get('manual_annotations'),
                            f'{did}: reviewed excluded annotations retained unchanged in manifest')
                    require(retained.get('exclusion_reason') == excluded.get('exclusion_reason'),
                            f'{did}: excluded metadata and manifest share selection reason')
                for view in ['camera_a', 'camera_b']:
                    camera = excluded[view]
                    source = sources.get(camera['recording_id'])
                    require(source is not None, f'{did}/{view}: excluded source recording resolves')
                    if source is None:
                        continue
                    require(camera['filename'] == source['filename'] and camera.get('source_sha256') == source['sha256'],
                            f'{did}/{view}: excluded clip retains original source filename/hash')
                    start, end = float(camera['start_s']), float(camera['end_s'])
                    require(0 <= start < end <= float(source['duration_s']) + .02, f'{did}/{view}: excluded bounds remain source-traceable')
                    clip_path = root / camera['clip']
                    require(clip_path.parent == folder, f'{did}/{view}: both excluded clips retained in excluded episode directory')
                    require(clip_path.is_file(), f'{did}/{view}: excluded derivative clip preserved')
                    if clip_path.is_file():
                        actual = float(probe(clip_path)['format']['duration'])
                        require(abs(actual - (end - start)) <= .12, f'{did}/{view}: excluded clip duration agrees with source interval')
                a, b, offset = excluded['camera_a'], excluded['camera_b'], float(excluded['sync_offset_s'])
                require(abs(float(b['start_s']) - float(a['start_s']) - offset) < .05 and
                        abs(float(b['end_s']) - float(a['end_s']) - offset) < .05,
                        f'{did}: excluded camera intervals retain signed synchronization offset')
                tracking_path = folder / 'tracking.csv'
                require(tracking_path.is_file(), f'{did}: excluded observations retained or reproduced')
                if tracking_path.exists():
                    tracked = rows(tracking_path)
                    require(bool(tracked), f'{did}: excluded tracking contains actual source observations')
                    for view in ['camera_a', 'camera_b']:
                        camera = excluded[view]
                        expected_samples = [r for r in rows(results / 'tracking_sources' / (camera['recording_id'] + '.csv'))
                                            if float(camera['start_s']) <= float(r['source_time_s']) < float(camera['end_s'])]
                        require(sum(r.get('view') == view for r in tracked) == len(expected_samples),
                                f'{did}/{view}: excluded tracker retains every sampled source observation inside bounds')
                    for row in tracked:
                        view = row.get('view')
                        require(row.get('demo_id') == did and view in ['camera_a', 'camera_b'],
                                f'{did}: excluded tracker row retains ID/view')
                        if view not in ['camera_a', 'camera_b']:
                            continue
                        camera = excluded[view]
                        t = float(row['source_time_s'])
                        require(float(camera['start_s']) <= t < float(camera['end_s']) and
                                abs(float(row['clip_time_s']) - (t - float(camera['start_s']))) < .002,
                                f'{did}/{view}: excluded tracking timestamp remains traceable')
                        rid = camera['recording_id']
                        if rid not in source_tracking:
                            source_tracking[rid] = {float(r['source_time_s']): r for r in rows(results / 'tracking_sources' / (rid + '.csv'))}
                        original_row = source_tracking[rid].get(t)
                        require(original_row is not None and all(row.get(k) == v for k, v in original_row.items()),
                                f'{did}/{view}: excluded observations exactly reproduce unchanged source tracker')
            if not semantic_ok:
                pending.append('Task semantic annotation review unresolved: ' + str(semantic_review.get('status')) +
                               ' ' + ','.join(semantic_review.get('unresolved_recording_ids', [])))
        else:
            semantic_review = {'status': 'missing'}
            pending.append('review/task_semantics_review.json with reviewed task semantics required')
        review_path = results / 'user_review.json'
        if review_path.exists():
            user_review = json.loads(review_path.read_text())
            # Finalizer may provide structured review items; never infer approval from agent notes.
            review_items = user_review.get('reviews', []) if isinstance(user_review, dict) else user_review
            for item in review_items:
                if item.get('reviewer_type') == 'human' and item.get('confirmed') is True:
                    human_review_types.add(item.get('review_type'))
        if not {'correct_placement', 'tracking_or_annotation_limitation'} <= human_review_types:
            pending.append('Human confirmation of one correct placement and one tracking/annotation limitation')

    # Keep thousands of per-row checks out of reports; preserve all failures.
    return {'validator': 'scripts/m0_validate.py', 'stage': stage,
            'structural_checks_passed': not errors,
            'acceptance_complete': stage == 'final' and not errors and not pending and semantic_ok,
            'errors': sorted(set(errors)),
            'pending': sorted(set(pending)), 'checks_passed_count': len(checks),
            'recording_count': len(sources), 'demonstration_count': len(demos),
            'human_reviewed_count': sum(d.get('human_reviewed') is True for d in demos),
            'annotation_semantics_reviewed': semantic_ok, 'task_semantics_review': semantic_review,
            'excluded_demonstration_validation': excluded_validation,
            'actual_pts_validation': {'sources': pts_source_audit,
                'sampled_frames_checked': len(pts_sample_errors),
                'sampled_pts_max_error_s': max(pts_sample_errors, default=0.0),
                'initial_final_evidence_frames_checked': len(pts_evidence_errors),
                'evidence_pts_max_error_s': max(pts_evidence_errors, default=0.0),
                'tolerance_s': 1e-5, 'timestamp_source': 'ffprobe best_effort_timestamp_time every original video frame'},
            'scope': 'Structural checks cover hashes, traceability and timing. Annotation acceptance separately requires resolved task semantics; neither certifies physical success or substitutes for human review.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--stage', choices=['inventory', 'final'], default='final')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true', help='Prove a tampered manifest fails without editing sources')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.self_check:
        with tempfile.TemporaryDirectory(prefix='m0-validator-') as scratch:
            fixture = Path(scratch)
            fixture_data = fixture / 'data/personal/m0'
            fixture_results = fixture / 'results/personal-m0-20261009'
            fixture_data.mkdir(parents=True)
            fixture_results.mkdir(parents=True)
            (fixture_data / 'raw').symlink_to(root / 'data/personal/m0/raw', target_is_directory=True)
            (fixture_results / 'inspection').symlink_to(root / 'results/personal-m0-20261009/inspection', target_is_directory=True)
            for name in ['raw_manifest.csv', 'source_inventory_local.json', 'source_checksums.sha256']:
                (fixture_data / name).write_bytes((root / 'data/personal/m0' / name).read_bytes())
            assert validate(fixture, 'inventory')['structural_checks_passed']
            manifest_path = fixture_data / 'raw_manifest.csv'
            contents = manifest_path.read_text()
            original_hash = rows(manifest_path)[0]['sha256']
            manifest_path.write_text(contents.replace(original_hash, '0' * 64, 1))
            assert not validate(fixture, 'inventory')['structural_checks_passed']
        print('Self-check passed: altered SHA-256 manifest rejected; original recordings untouched.')
        return
    result = validate(root, args.stage)
    output = args.output or root / 'results/personal-m0-20261009/independent_validation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['structural_checks_passed'] else 1)


if __name__ == '__main__':
    main()
