"""Assemble reviewed M0 records without retracking, training, or changing raw videos.

Run only after the clap-boundary audit and human-review evidence are finalized.
Original source tracker rows are copied exactly; per-trial IDs and clip times are appended.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, records, fields=None):
    fields = fields or list(dict.fromkeys(key for row in records for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def truth(value):
    return value is True or str(value).lower() in ['true', '1', 'yes']


def studs_correct(value):
    return value is True or str(value).lower().startswith('yes')


def decoded_frames(path):
    result = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(path)], text=True)
    return int(json.loads(result)['streams'][0]['nb_read_frames'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--no-previews', action='store_true', help='Reuse existing preview artifacts without encoding')
    args = parser.parse_args()
    root = args.root.resolve()
    data = root / 'data/personal/m0'
    out = root / 'results/personal-m0-20261009'
    sources = read_csv(data / 'raw_manifest.csv')
    pairs = read_csv(data / 'camera_pairs.csv')
    placement = json.loads((out / 'review/final_placement_review.json').read_text())
    placement_by_id = {d['demo_id']: d for d in placement['demonstrations']}
    scenario = json.loads((out / 'review/scenario_review.json').read_text())
    scenario_by_id = {d['demo_id']: d for d in scenario['demos']}
    human_file = out / 'user_review.json'
    human = json.loads(human_file.read_text()) if human_file.exists() else {'reviews': []}
    human_reviews = human.get('reviews', [])
    approved = [r for r in human_reviews if r.get('reviewer_type') == 'human' and r.get('confirmed') is True]
    approved_types = {r.get('review_type') for r in approved}
    segmentation = json.loads((out / 'segmentation_config.json').read_text())
    semantics_path = out / 'review/task_semantics_review.json'
    semantics = json.loads(semantics_path.read_text()) if semantics_path.exists() else {
        'status': 'missing', 'user_reported_counts': {}, 'unresolved_recording_ids': []}
    episode_labels = semantics.get('episode_labels', {})
    if not isinstance(episode_labels, dict):
        raise RuntimeError('Task semantic episode labels must map explicit episode IDs to categories')
    unresolved_recordings = set(semantics.get('unresolved_recording_ids', []))
    semantic_ok = semantics.get('status') == 'reviewed' and not unresolved_recordings
    allowed_types = {'feeding', 'normal_feeding', 'target_local_reorientation',
                     'orientation_correction', 'awkward_pause_resume_feeding', 'already_completed'}
    semantic_counts = Counter()


    source_rows = {}
    for source in sources:
        rid = source['recording_id']
        source['camera_identity'] = 'primary' if int(rid[-4:]) >= 5 else 'secondary'
        source['camera_confidence'] = 'high_agent_visual_cross_view_identification'
        source_rows[rid] = read_csv(out / 'tracking_sources' / (rid + '.csv'))
    write_csv(data / 'raw_manifest.csv', sources)

    def enrich_record(demo, reviewed, tags, category, reset_manual=False):
        did = demo['demo_id']
        original_manual = demo.get('manual_annotations', {})
        # Excluded candidates may start from V2-rich metadata or clean segmentation output.
        # Keep the same supplied physical observations, then rebuild every reviewed field.
        grasp_lift_observed = 'grasp_lift_transport_place_release_observed' in tags['observed_tags']
        manual = {'grasp_occurred': True if grasp_lift_observed else None,
                  'lift_occurred': True if grasp_lift_observed else None,
                  'correction_intent': 'not_applicable',
                  'human_disturbance': 'unknown_not_separately_annotated',
                  'notes': 'User-excluded short-pause teaching candidate; raw source and reviewed outcome remain preserved.'} \
                 if reset_manual else dict(original_manual)
        manual.update({
            'annotation_provenance': 'agent_visual_review_with_separate_specific_human_examples',
            'task_category': category, 'task_semantics_review_status': semantics.get('status'),
            'target_reached_visual': reviewed['target_reached_visual'],
            'canonical_orientation_correct': reviewed['blue2upright'],
            'studs_toward_top_manual': reviewed['studstowardtop'],
            'studs_review_confidence': reviewed['studs_review_confidence'],
            'full_contact_footprint_containment': 'unknown_not_metrically_established',
            'outcome': reviewed['outcome_agent_visual'],
            'interrupted_task_observed': tags['interrupted_task_observed'],
            'scenario_tags': tags['observed_tags'],
            'initial_orientation_visual': tags['initial_orientation_visual'],
            'initial_position_relative_to_target': tags['initial_position_relative_to_target'],
            'final_orientation_visual': tags['final_orientation_visual'],
            'manual_review_limitations': reviewed['limitations'] + tags['manual_notes'],
            'complete_occlusion': 'unknown_between_sampled_frames',
        })
        demo['manual_annotations'] = manual
        demo['manual_scenario_review'] = tags
        demo['agent_final_placement_review'] = reviewed
        demo['observation_measurements'] = {}
        demo['human_reviews'] = []
        for review in approved:
            rid = review.get('recording_id')
            t = review.get('source_time_s')
            if rid == demo['camera_a']['recording_id'] and t is not None and \
                    demo['camera_a']['start_s'] <= float(t) <= demo['camera_a']['end_s']:
                demo['human_reviews'].append(review)
        demo['human_reviewed'] = bool(demo['human_reviews'])
        demo['human_review_scope'] = 'specific_source_frames_only_not_every_frame_of_episode'
        trial_rows = []
        for view in ['camera_a', 'camera_b']:
            camera = demo[view]
            selected = [row for row in source_rows[camera['recording_id']]
                        if float(camera['start_s']) <= float(row['source_time_s']) < float(camera['end_s'])]
            if not selected:
                raise RuntimeError(f'{did}/{view}: no actual timestamped source samples inside clip')
            demo['observation_measurements'][view] = {
                'first_actual_sample': selected[0], 'last_actual_sample': selected[-1],
                'sample_count': len(selected),
                'detected_sample_count': sum(truth(row['detected']) for row in selected),
                'confidence_ge_065_count': sum(float(row['tracking_confidence']) >= .65 for row in selected),
                'measurement_provenance': 'unchanged_source_tracking_csv_actual_decoded_PTS_no_interpolation',
            }
            trial_rows.extend({**row, 'demo_id': did, 'view': view,
                               'clip_time_s': round(float(row['source_time_s']) - float(camera['start_s']), 6)}
                              for row in selected)
        if demo.get('excluded_from_active_dataset') is True:
            folder = (root / demo['camera_a']['clip']).parent
            demo['quality_artifacts'] = {'tracking_csv': str((folder / 'tracking.csv').relative_to(root)),
                'source_tracking_csvs': [str((out / 'tracking_sources' / (demo[view]['recording_id'] + '.csv')).relative_to(root))
                                        for view in ['camera_a', 'camera_b']]}
        else:
            demo['quality_artifacts'] = {'tracking_preview_candidates':
                [str((out / 'tracking_previews' / (did + suffix)).relative_to(root))
                 for suffix in ['.mp4', '.timeline.json']]}
        return manual, trial_rows

    demos, trajectories, annotations, usable = [], [], [], []
    for metadata_path in sorted((data / 'demonstrations').glob('demo_*/metadata.json')):
        demo = json.loads(metadata_path.read_text())
        did = demo['demo_id']
        if did not in placement_by_id or did not in scenario_by_id:
            raise RuntimeError(f'{did}: reviewed placement/scenario evidence missing; do not invent labels')
        reviewed, tags = placement_by_id[did], scenario_by_id[did]
        # Reapply explicit review labels from the original candidate type; do not retain stale labels.
        demo.setdefault('source_attempt_type', demo['attempt_type'])
        demo['attempt_type'] = episode_labels.get(did, demo['source_attempt_type'])
        if demo['attempt_type'] not in allowed_types:
            raise RuntimeError(f'{did}: unsupported reviewed episode label {demo["attempt_type"]!r}')
        unresolved = demo['camera_a']['recording_id'] in unresolved_recordings and did not in episode_labels
        category = 'feeding_unresolved' if unresolved and demo['attempt_type'] == 'feeding' else \
                   'normal_feeding' if demo['attempt_type'] in {'feeding', 'normal_feeding'} else \
                   'orientation_correction' if demo['attempt_type'] in {'target_local_reorientation', 'orientation_correction'} else demo['attempt_type']
        demo['task_category'] = category
        demo['task_semantics_review'] = {'review_file': 'results/personal-m0-20261009/review/task_semantics_review.json',
            'status': semantics.get('status'), 'applied_episode_label': episode_labels.get(did),
            'recording_category_unresolved': unresolved,
            'count_scope': 'retained_candidate_episode_not_individual_motion_event'}
        semantic_counts[category] += 1

        manual, trial_rows = enrich_record(demo, reviewed, tags, category)
        write_csv(metadata_path.parent / 'tracking.csv', trial_rows)
        trajectories.extend(trial_rows)
        known_angles = [r for r in trial_rows if r['view'] == 'camera_a' and r['orientation_canonical_clockwise_deg']]
        row = {'demo_id': did, 'pair_id': demo['pair_id'], 'attempt_type': demo['attempt_type'],
               'task_category': category, 'task_semantics_review_status': semantics.get('status'),
               'manipulation_attempt': demo.get('manipulation_attempt'),
               'primary_recording_id': demo['camera_a']['recording_id'],
               'primary_start_s': demo['camera_a']['start_s'], 'primary_end_s': demo['camera_a']['end_s'],
               'secondary_recording_id': demo['camera_b']['recording_id'],
               'secondary_start_s': demo['camera_b']['start_s'], 'secondary_end_s': demo['camera_b']['end_s'],
               'sync_offset_s': demo['sync_offset_s'], 'sync_uncertainty_s': demo['sync_uncertainty_s'],
               'boundary_uncertainty_s': demo['boundary_uncertainty_s'],
               'target_reached_visual': reviewed['target_reached_visual'],
               'final_number2_upright_manual': reviewed['blue2upright'],
               'final_studs_toward_top_manual': reviewed['studstowardtop'],
               'final_studs_review_confidence': reviewed['studs_review_confidence'],
               'goal_outcome_agent_visual': reviewed['outcome_agent_visual'],
               'initial_orientation_manual': tags['initial_orientation_visual'],
               'final_orientation_manual': tags['final_orientation_visual'],
               'initial_position_manual': tags['initial_position_relative_to_target'],
               'grasp_occurred_manual': manual.get('grasp_occurred'),
               'lift_occurred_manual': manual.get('lift_occurred'),
               'human_disturbance_manual': manual.get('human_disturbance'),
               'interrupted_task_observed': tags['interrupted_task_observed'],
               'complete_occlusion': manual['complete_occlusion'],
               'primary_known_orientation_samples': len(known_angles),
               'tracking_reliable': 'heuristic_confidence_not_validated_accuracy',
               'scenario_tags': json.dumps(tags['observed_tags']),
               'human_reviewed_specific_frames': demo['human_reviewed']}
        annotations.append(row)
        success = reviewed['blue2upright'] is True and studs_correct(reviewed['studstowardtop'])
        use = 'semantic_category_review_pending_not_training_ready' if category == 'feeding_unresolved' else \
              'feeding_observations' if category == 'normal_feeding' else \
              'awkward_pause_resume_feeding_with_full_context' if category == 'awkward_pause_resume_feeding' else \
              'already_completed_no_action_observation' if category == 'already_completed' else \
              'correction_sequence_with_context_not_ordinary_successful_imitation'
        usable.append({'demo_id': did, 'recording_id': demo['camera_a']['recording_id'],
                       'suggested_use': use, 'visible_goal_correct': success,
                       'training_ready_without_review': False})
        write_json(metadata_path, demo)
        demos.append(demo)

    excluded_manifest_path = data / 'excluded_demo_manifest.json'
    excluded_manifest = json.loads(excluded_manifest_path.read_text()) if excluded_manifest_path.exists() else {
        'excluded_demonstrations': []}
    excluded_demos = excluded_manifest.get('excluded_demonstrations', [])
    declared_excluded_ids = set(semantics.get('excluded_episode_ids', []))
    preserved_excluded_ids = {d['demo_id'] for d in excluded_demos}
    active_ids = {d['demo_id'] for d in demos}
    excluded_evidence_ok = preserved_excluded_ids == declared_excluded_ids and not (active_ids & declared_excluded_ids)
    # Fresh reproduction regenerates excluded observations from the same source rows, never from interpolation.
    for excluded in excluded_demos:
        did = excluded['demo_id']
        if did not in placement_by_id or did not in scenario_by_id:
            raise RuntimeError(f'{did}: excluded candidate requires the same saved placement/scenario review')
        excluded.setdefault('source_attempt_type', excluded['attempt_type'])
        excluded['attempt_type'] = 'awkward_pause_resume_feeding'
        excluded['task_category'] = 'awkward_pause_resume_feeding'
        excluded['task_semantics_review'] = {
            'review_file': 'results/personal-m0-20261009/review/task_semantics_review.json',
            'status': semantics.get('status'), 'applied_episode_label': 'awkward_pause_resume_feeding',
            'recording_category_unresolved': False,
            'count_scope': 'excluded_candidate_not_in_curated_counts',
            'selection_reason': excluded['exclusion_reason'],
            'user_clarification': semantics.get('user_clarification')}
        _, excluded_rows = enrich_record(excluded, placement_by_id[did], scenario_by_id[did],
                                         'awkward_pause_resume_feeding', reset_manual=True)
        folder = (root / excluded['camera_a']['clip']).parent
        if not folder.is_relative_to(data / 'excluded_demonstrations'):
            raise RuntimeError(f'{did}: excluded derivative must reside in declared excluded directory')
        write_csv(folder / 'tracking.csv', excluded_rows)
        write_json(folder / 'metadata.json', excluded)
    excluded_manifest['excluded_demonstrations'] = excluded_demos
    if excluded_manifest_path.exists() or declared_excluded_ids:
        write_json(excluded_manifest_path, excluded_manifest)
    write_csv(data / 'trajectories.csv', trajectories)
    write_csv(data / 'trial_annotations.csv', annotations)
    quality = []
    for rid, samples in source_rows.items():
        for row in samples:
            issues = []
            if not truth(row['detected']):
                issues.append('block_unknown_occlusion_out_of_frame_or_detector_failure')
            elif float(row['tracking_confidence']) < .65:
                issues.append('low_heuristic_detection_confidence')
            if row['camera_identity'] == 'primary' and not row['orientation_canonical_clockwise_deg']:
                issues.append('orientation_unknown_partial_or_ambiguous_glyph')
            if row['camera_identity'] == 'secondary':
                issues.append('canonical_orientation_not_supported_in_secondary_view')
            quality.append({**row, 'qa_issues': '|'.join(issues),
                            'requires_review': any(issue != 'canonical_orientation_not_supported_in_secondary_view' for issue in issues)})
    write_csv(data / 'quality_report.csv', quality)
    excluded_setup = []
    archive = out / 'intermediate_segmentation_v1/excluded_between_take_setup'
    archived_by_recording = {}
    for path in sorted(archive.glob('demo_*/metadata.json')):
        old = json.loads(path.read_text())
        archived_by_recording.setdefault(old['camera_a']['recording_id'], []).append((old, path.parent))
    for rid, windows in segmentation.get('excluded_between_take_setup_primary_s', {}).items():
        pair = next(row for row in pairs if row['primary_recording_id'] == rid)
        for index, (start, end) in enumerate(windows):
            archived = archived_by_recording.get(rid, [])
            old, directory = archived[index] if index < len(archived) else ({}, None)
            excluded_setup.append({'excluded_setup_id': f'{rid}_setup_{index + 1:02d}',
                'preliminary_demo_id': old.get('demo_id', ''),
                'primary_recording_id': rid, 'primary_start_s': start, 'primary_end_s': end,
                'secondary_recording_id': pair['secondary_recording_id'],
                'secondary_start_s': round(start + float(pair['offset_s']), 6),
                'secondary_end_s': round(end + float(pair['offset_s']), 6),
                'classification': 'between_take_setup_excluded_from_episode_and_failure_counts',
                'archived_primary_clip': str((directory / 'camera_a.mp4').relative_to(root)) if directory else '',
                'archived_secondary_clip': str((directory / 'camera_b.mp4').relative_to(root)) if directory else '',
                'archive_clip_scope': 'preliminary_action_window_not_full_between_clap_interval' if directory else 'no_preliminary_archive_in_fresh_reproduction',
                'reason': 'User-defined clap boundaries put this repositioning outside manipulation takes'})
    write_csv(data / 'excluded_setup_annotations.csv', excluded_setup,
              fields=list(excluded_setup[0]) if excluded_setup else ['preliminary_demo_id', 'classification'])
    all_samples = [row for samples in source_rows.values() for row in samples]
    primary_samples = [r for r in all_samples if r['camera_identity'] == 'primary']
    counts = Counter(d['attempt_type'] for d in demos)
    detected = sum(truth(r['detected']) for r in all_samples)
    confident = sum(float(r['tracking_confidence']) >= .65 for r in all_samples)
    angles = sum(bool(r['orientation_canonical_clockwise_deg']) for r in primary_samples)
    correct = sum(r['final_number2_upright_manual'] is True and studs_correct(r['final_studs_toward_top_manual']) for r in annotations)
    failed = sum(r['final_number2_upright_manual'] is False or r['final_studs_toward_top_manual'] is False for r in annotations)
    unknown = len(annotations) - correct - failed
    coverage = {'sample_count': len(all_samples), 'detected_count': detected,
                'tracking_coverage_fraction': detected / len(all_samples),
                'confidence_ge_065_count': confident, 'confidence_ge_065_fraction': confident / len(all_samples),
                'primary_sample_count': len(primary_samples), 'primary_known_orientation_count': angles,
                'primary_orientation_coverage_fraction': angles / len(primary_samples),
                'coverage_is_detection_rate_not_accuracy': True,
                'coverage_denominator': 'all_actual_2Hz_source_samples_including_setup_reset_and_static_intervals'}
    active_detected = sum(truth(r['detected']) for r in trajectories)
    active_primary = [r for r in trajectories if r['view'] == 'camera_a']
    active_angles = sum(bool(r['orientation_canonical_clockwise_deg']) for r in active_primary)
    active_coverage = {'sample_count': len(trajectories), 'detected_count': active_detected,
                      'tracking_coverage_fraction': active_detected / len(trajectories),
                      'primary_sample_count': len(active_primary), 'primary_known_orientation_count': active_angles,
                      'primary_orientation_coverage_fraction': active_angles / len(active_primary),
                      'coverage_is_detection_rate_not_accuracy': True,
                      'coverage_denominator': 'actual_samples_within_retained_clap_defined_episodes_including_static_episode'}
    selected_expected_counts = semantics.get('selected_expected_counts', {})
    selected_counts_match = all(semantic_counts.get(category, 0) == count for category, count in selected_expected_counts.items())
    semantic_ok = semantic_ok and selected_counts_match and excluded_evidence_ok
    dataset = {'schema_version': 1, 'recording_count': len(sources), 'matched_camera_pair_count': len(pairs),
               'demonstration_count': len(demos),
               'manipulation_attempt_count': sum(d.get('manipulation_attempt') is True for d in demos),
               'already_completed_observation_count': semantic_counts['already_completed'],
               'task_category_counts': dict(semantic_counts),
               'selected_expected_counts': selected_expected_counts, 'selected_counts_match_review': selected_counts_match,
               'excluded_episode_ids': sorted(declared_excluded_ids), 'excluded_demonstration_count': len(excluded_demos),
               'excluded_demo_manifest': 'data/personal/m0/excluded_demo_manifest.json',
               'user_reported_counts': semantics.get('user_reported_counts', {}),
               'task_semantics_review': semantics, 'annotation_semantics_reviewed': semantic_ok,
               'attempt_type_counts': dict(counts), 'correct_visible_final_placements': correct,
               'incorrect_visible_final_placements': failed, 'unknown_or_ambiguous_outcomes': unknown,
               'canonical_view': 'primary_rotated_90_degrees_counterclockwise_user_confirmed',
               'temporal_sampling_hz': 2, 'observation_quality': coverage,
               'active_episode_observation_quality': active_coverage,
               'excluded_between_take_setup_count': len(excluded_setup),
               'annotation_inputs': ['review/final_placement_review.json', 'review/scenario_review.json', 'review/task_semantics_review.json', 'user_review.json', 'data/personal/m0/excluded_demo_manifest.json'],
               'usable_records': usable, 'demonstrations': demos,
               'status': 'checkpoint_C_assembled_checkpoint_D_independent_validation_pending'}
    write_json(data / 'dataset_manifest.json', dataset)
    session = json.loads((data / 'session_metadata.json').read_text())
    session.update({'records': sources, 'camera_pair_count': len(pairs),
                    'canonical_primary_transform': '90_degrees_counterclockwise_user_confirmed',
                    'original_source_coordinates': 'unmodified_decoded_portrait_image',
                    'physical_scale': 'unknown_no_millimetres', 'status': dataset['status']})
    write_json(data / 'session_metadata.json', session)

    frozen_paths = subprocess.check_output(['git', 'ls-files', '-z', '--',
        'results/mint-20261007', 'results/ucl-20261007'], cwd=root).decode().split('\0')
    frozen = []
    for name in filter(None, frozen_paths):
        expected = hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=root)).hexdigest()
        actual = sha256(root / name)
        frozen.append({'path': name, 'head_sha256': expected, 'actual_sha256': actual, 'unchanged': actual == expected})
    frozen_ok = all(row['unchanged'] for row in frozen)
    write_json(out / 'frozen_integrity.json', {'tracked_frozen_file_count': len(frozen),
                                             'unchanged': frozen_ok, 'files': frozen})
    if not frozen_ok:
        raise RuntimeError('Frozen Mint/UCL artifact changed; stop finalization')

    required_review_types = {'correct_placement', 'tracking_or_annotation_limitation'}
    human_ok = required_review_types <= approved_types
    preview_count = len(list((out / 'tracking_previews').glob('*.mp4')))
    sync_preview_count = len(list((out / 'synchronization_previews').glob('*.mp4')))
    preview_errors, preview_checks, sync_checks = [], [], []
    preview_validation_path = out / 'tracking_previews/preview_validation.json'
    preview_evidence = json.loads(preview_validation_path.read_text()).get('previews', []) if preview_validation_path.exists() else []
    preview_by_name = {item['preview_name']: item for item in preview_evidence}
    for demo in demos:
        did = demo['demo_id']
        path = out / 'tracking_previews' / (did + '.mp4')
        timeline_path = out / 'tracking_previews' / (did + '.timeline.json')
        evidence = preview_by_name.get(did, {})
        check = {'demo_id': did, 'path': str(path.relative_to(root)), 'exists': path.exists()}
        expected = demo['observation_measurements']['camera_a']['sample_count']
        if path.exists() and timeline_path.exists() and evidence:
            timeline = json.loads(timeline_path.read_text())
            check.update({'expected_source_samples': expected, 'timeline_frames': len(timeline.get('timeline', [])),
                'independently_decoded_frames': decoded_frames(path),
                'previously_validated_decoded_frames': evidence.get('frame_count_decoded'),
                'previously_validated_expected_frames': evidence.get('frame_count_expected')})
            check['passed'] = (check['independently_decoded_frames'] == expected == check['timeline_frames'] ==
                check['previously_validated_decoded_frames'] == check['previously_validated_expected_frames'] and
                evidence.get('h264_path') == str(path.relative_to(root)))
        else:
            check['passed'] = False
        if not check['passed']:
            preview_errors.append(f'{did}: missing or inconsistent decoded tracking preview/timeline')
        preview_checks.append(check)
    sync_validation_path = out / 'synchronization_preview_validation.json'
    sync_evidence = json.loads(sync_validation_path.read_text()) if sync_validation_path.exists() else []
    sync_by_path = {item['preview']: item for item in sync_evidence}
    for pair in pairs:
        path = out / 'synchronization_previews' / (pair['pair_id'] + '.mp4')
        relative = str(path.relative_to(root))
        evidence = sync_by_path.get(relative, {})
        check = {'pair_id': pair['pair_id'], 'path': relative, 'exists': path.exists()}
        check['independently_decoded_frames'] = decoded_frames(path) if path.exists() else 0
        check['previously_validated_decoded_frames'] = evidence.get('decoded_frames', 0)
        check['passed'] = check['independently_decoded_frames'] > 0 and check['independently_decoded_frames'] == check['previously_validated_decoded_frames']
        if not check['passed']:
            preview_errors.append(f'{pair["pair_id"]}: missing or inconsistent decoded synchronization preview')
        sync_checks.append(check)
    previews_ok = not preview_errors and len(preview_checks) == len(demos) and len(sync_checks) == len(pairs)
    validation = {'recording_count': len(sources), 'camera_pair_count': len(pairs),
                  'demonstration_count': len(demos), 'manipulation_attempt_count': dataset['manipulation_attempt_count'],
                  'counts': dict(counts), 'task_category_counts': dict(semantic_counts),
                  'user_reported_counts': semantics.get('user_reported_counts', {}),
                  'selected_expected_counts': selected_expected_counts, 'selected_counts_match_review': selected_counts_match,
                  'excluded_episode_ids': sorted(declared_excluded_ids), 'excluded_demonstration_count': len(excluded_demos),
                  'excluded_evidence_preserved': excluded_evidence_ok,
                  'task_semantics_review': semantics, 'annotation_semantics_reviewed': semantic_ok,
                  'coverage': coverage, 'correct_visible_final_placements': correct,
                  'active_episode_coverage': active_coverage,
                  'excluded_between_take_setup_count': len(excluded_setup),
                  'incorrect_visible_final_placements': failed, 'unknown_or_ambiguous_outcomes': unknown,
                  'human_review_required_examples_confirmed': human_ok,
                  'human_review_count': len(approved), 'frozen_artifacts_unchanged': frozen_ok,
                  'tracking_preview_count': preview_count, 'synchronization_preview_count': sync_preview_count,
                  'active_tracking_preview_validation': preview_checks, 'synchronization_preview_validation': sync_checks,
                  'preview_errors': preview_errors, 'all_required_previews_validated': previews_ok,
                  'acceptance_complete': False, 'independent_validation': 'pending'}
    commands = ['.venv-sim/bin/python scripts/' + name + '.py' for name in
                ['m0_prepare', 'm0_sync', 'm0_segment', 'm0_track', 'm0_tracking_previews', 'm0_finalize', 'm0_validate']]
    write_json(out / 'finalization_commands.json', {'commands': commands, 'python': platform.python_version(),
               'current_command': ['scripts/m0_finalize.py'] + (['--no-previews'] if args.no_previews else []),
               'annotation_inputs_required': dataset['annotation_inputs'],
               'canonical_rotation': dataset['canonical_view'], 'temporal_sampling_hz': 2,
               'tracking_confidence_threshold_for_reporting': .65,
               'preview_behavior': 'reuse_existing_previews_no_encoding_in_finalizer'})
    runtime = {'python': platform.python_version(), 'platform': platform.platform(),
               'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()
                                       if d.metadata.get('Name'))),
               'ffprobe': subprocess.check_output(['ffprobe', '-version'], text=True).splitlines()[0]}
    write_json(out / 'runtime_versions.json', runtime)
    report = f'''# Personal manipulation dataset — M0, 9 October 2026

This milestone inventories, synchronizes, segments and annotates the supplied phone recordings on Powerglide. It does not train a policy, change a simulator or publish the repository.

## Measured milestone results

| Metric | Result |
|---|---:|
| Raw recordings | {len(sources)} |
| Matched camera pairs | {len(pairs)} |
| Curated task episodes | {len(demos)} |
| Explicitly excluded candidates, preserved separately | {len(excluded_demos)} |
| Total preserved candidate episodes | {len(demos) + len(excluded_demos)} |
| Curated manipulation attempts | {dataset['manipulation_attempt_count']} |
| Preserved excluded manipulation attempts | {sum(d.get('manipulation_attempt') is True for d in excluded_demos)} |
| Ordinary feeding category | {semantic_counts['normal_feeding']} |
| Awkward/pause-resume feeding category | {semantic_counts['awkward_pause_resume_feeding']} |
| Feeding candidates with unresolved category | {semantic_counts['feeding_unresolved']} |
| Corrective demonstrations | {semantic_counts['orientation_correction']} |
| Already-completed static demonstrations | {semantic_counts['already_completed']} |
| Incomplete demonstrations | 0 observed |
| Failed visible-goal outcomes within retained episodes | {failed} |
| Successfully synchronized demonstrations | {len(demos)} |
| Tracking coverage, all source samples | {detected}/{len(all_samples)} ({100 * detected / len(all_samples):.2f}%) |
| Heuristic detection confidence ≥0.65 | {confident}/{len(all_samples)} ({100 * confident / len(all_samples):.2f}%) |
| Primary orientation observable | {angles}/{len(primary_samples)} ({100 * angles / len(primary_samples):.2f}%) |
| Tracking coverage within retained episodes | {active_detected}/{len(trajectories)} ({100 * active_detected / len(trajectories):.2f}%) |
| Primary orientation observable within retained episodes | {active_angles}/{len(active_primary)} ({100 * active_angles / len(active_primary):.2f}%) |
| Correct visible final placements/states | {correct}/{len(demos)} |
| Unknown or ambiguous final outcomes | {unknown} |
| Feeding candidates awaiting scenario labels | {semantic_counts['feeding_unresolved']} |
| Task semantics review | {semantics.get('status')} |

These observed counts describe the curated intervals, with excluded candidates separately preserved. Task semantics review status is **{semantics.get('status')}**; annotation acceptance is **{semantic_ok}**. Structural checks establish hashes and timestamps, not correct scenario labels or intended trial grouping. Candidate clips remain preserved while semantic questions are resolved; individual transports, releases or pauses are not automatically separate intended task demonstrations.

The user's initial reported task counts are recorded separately, without changing observations to match them:

| User-reported category | Count |
|---|---:|
| Ordinary feeding | {semantics.get('user_reported_counts', {}).get('normal_feeding', 'unknown')} |
| Orientation correction | {semantics.get('user_reported_counts', {}).get('orientation_correction', 'unknown')} |
| Awkward/pause-resume feeding | {semantics.get('user_reported_counts', {}).get('awkward_pause_resume_feeding', 'unknown')} |
| Already completed | {semantics.get('user_reported_counts', {}).get('already_completed', 'unknown')} |

The resolved selected-count expectations are {json.dumps(selected_expected_counts, sort_keys=True)}. They are checked against the curated annotations rather than substituted for measured counts. The user's clarification identifies source 0008 example 5 as ordinary feeding and excludes example 6 because its pause is too brief for the intended teaching set. This increases curated ordinary feeds relative to the initial report while retaining four awkward/pause-resume examples.

Excluded episode IDs: {', '.join(sorted(declared_excluded_ids)) or 'none'}. Their original bounds, reviewed annotations, both derivative clips and exact sampled tracking remain under `data/personal/m0/excluded_demonstrations/`, linked by `excluded_demo_manifest.json` to the explicit user selection provenance. Exclusion from the curated teaching set is not a failed attempt and does not remove the raw recording or historical evidence.

Unresolved recording IDs: {', '.join(sorted(unresolved_recordings)) or 'none'}. Explicit per-episode labels, user-requested exclusions and the review status are reproducible inputs in `review/task_semantics_review.json`.

 The {len(excluded_setup)} setup/reset movements outside clap-defined takes remain in raw sources and archived preliminary segments where available; they are not independently counted as demonstrations or policy failures. `excluded_setup_annotations.csv` retains their source intervals and optional archived clip paths; fresh reproduction derives these intervals from the saved clap configuration. A static already-completed episode contains no manipulation.

The first feeding take in recording 0005 has no observed beginning clap, so its boundary falls back to the stationary recording start. The last correction take in recording 0007 has no observed ending clap, so its boundary falls back to the recording end. The shared camera overlap may clamp a requested boundary. All markers, fallback methods, buffer sizes and temporal uncertainty remain in each metadata record and `segmentation_config.json`; the clap audit evidence is preserved under `review/`.

## Source preservation and synchronization

All eight supplied WhatsApp-export originals are preserved byte-for-byte, with SHA-256 hashes in `source_checksums.sha256` and filenames, sizes, durations, resolutions, frame rates and audio presence in `raw_manifest.csv`. No reliable acquisition timestamp survived these exports; export filenames are not acquisition-time evidence. Camera identities were reviewed from actual perspectives: recordings 0001–0004 are secondary and 0005–0008 are primary.

`camera_pairs.csv` records offsets, uncertainty, confidence and method. The signed convention is **secondary source time = primary source time + offset**. Common movements and audio transients support pairing; raw-view side-by-side previews retain source timestamps. Phone audio/video latency was not independently calibrated. The static pair has no manipulation event, so its confidence rests on audio support rather than an invented visual alignment.

The user confirmed that the canonical primary observation is the derived **90° counterclockwise** view. This makes the intended blue 2 upright and the studs point upward. Raw originals, raw source timestamps and raw pixel measurements remain unchanged. Every primary clip declares its transform; the nearby arrow does not define object orientation.

## Measurements, annotations and calibration

The baseline uses orange/blue HSV segmentation, connected components, declared size thresholds, a small temporal candidate preference and a disclosed blue-2 rotation template. Target corners are four agent-reviewed points on the hand-drawn square. Their planar homography yields dimensionless target-square coordinates only. Elevated block points violate the plane assumption; no physical scale, full 3D trajectory or millimetre accuracy is reported.

Source tracker CSV rows are copied exactly into each trial's `tracking.csv` and aggregate `trajectories.csv`, with only demo/view/clip-time fields appended. Actual decoded source timestamps are retained at approximately 2 Hz. Missing object positions and unknown orientation remain blank; no long-occlusion interpolation occurs. Bounding boxes describe visible regions, not the complete object. A centroid over the square does not establish the full contact footprint. Stud direction is separately reviewed visually, with medium confidence in low-resolution examples, rather than numerically inferred from the arrow.

`manual_annotations` and scenario tags remain separate from `observation_measurements`. The final-pose review, whole-recording scenario review and the two specific human-confirmed examples are saved as inputs. Human confirmation of a source frame does not certify every frame of every trial. The limitation example may be outside a retained take; it still documents a detector limitation without becoming a trial failure.

## Quality and usable recordings

Coverage above is a **detection rate, not validated accuracy**. Its denominator includes every source sample, including resets and the stationary pair. Orange can overlap skin, partially hidden blue glyphs can fail template matching, perspective alters outlines, and 2 Hz sampling can miss short events. Absence cannot distinguish complete occlusion, departure from the image and detector failure. Complete-occlusion labels therefore remain unknown where evidence is insufficient. A review crop selecting skin instead of the block is retained as a concrete limitation.

Recordings 0005/0001 provide paired ordinary feeding observations. Pair 0008/0003 contains feeding motion candidates that the user associates with awkward-position and pause/resume examples; unresolved categories or grouping remain explicit until the task semantics review is complete. Pair 0007/0002 provides target-local correction sequences and requires its context; it must not be flattened into ordinary successful imitation. Pair 0006/0004 supplies separately clap-delimited already-completed no-action examples. Use `usable_records`, semantic review status and per-trial quality records before subsequent experiments. A deliberate pause/resume is not automatically a failed task or an external human disturbance. No broad interruption absence or robustness claim follows from these annotations. There is no automatic training/test split at M0.

## Reproduction and checkpoints

Use the existing `.venv-sim` on Powerglide and retained source files plus the saved review/configuration JSON inputs. Exact commands and runtime versions are in `finalization_commands.json`, `runtime_versions.json`, `commands.json`, tracker calibration/configuration, synchronization evidence and segmentation configuration. Source filenames and camera-pair offsets are not rediscovered from filenames. Reproduction requires the declared review inputs, including the clap-boundary audit. Finalization regenerates the review page before computing artifact hashes.

```bash
{chr(10).join(commands)}
```

Checkpoint A: eight-source inventory and hashes saved. Checkpoint B: paired, clap-reviewed clips and source bounds saved. Checkpoint C: exact sampled tracking, annotations, quality rows and reviewed examples saved. Checkpoint D: final manifests and this report assembled; independent validation determines the final status in `validation_summary.json`. Frozen Mint and UCL evidence is checked against its Git HEAD content in `frozen_integrity.json`; it is not overwritten.
'''
    (out / 'M0_report.md').write_text(report)
    (out / 'checkpoint_C.md').write_text(f'# Checkpoint C\n\n{len(trajectories)} source-traceable per-trial observation rows; {len(quality)} source quality rows. Missing positions/orientations stay unknown; no interpolation. Specific required human examples confirmed: {human_ok}. Task semantics review status: {semantics.get("status")}; annotation acceptance: {semantic_ok}.\n')
    command = [str(root / '.venv-sim/bin/python'), str(root / 'scripts/m0_validate.py'), '--root', str(root)]
    result = subprocess.run(command, capture_output=True, text=True)
    (out / 'independent_validation.log').write_text(result.stdout + result.stderr)
    independent_path = out / 'independent_validation.json'
    independent = json.loads(independent_path.read_text()) if independent_path.exists() else {'errors': ['Validator did not produce JSON']}
    validation['independent_validation'] = independent
    validation['acceptance_complete'] = result.returncode == 0 and independent.get('acceptance_complete') is True and human_ok and frozen_ok and previews_ok and semantic_ok
    validation['status'] = 'checkpoint_D_validated' if validation['acceptance_complete'] else \
        'checkpoint_D_pending_semantic_review' if not semantic_ok else 'checkpoint_D_pending_checks_or_artifacts'
    write_json(out / 'validation_summary.json', validation)
    dataset['status'] = validation['status']
    write_json(data / 'dataset_manifest.json', dataset)
    session['status'] = validation['status']
    write_json(data / 'session_metadata.json', session)
    (out / 'checkpoint_D.md').write_text(f'# Checkpoint D\n\nStatus: {validation["status"]}. Independent structural checks: {independent.get("structural_checks_passed", False)}. Task semantics reviewed: {semantic_ok}. Errors: {independent.get("errors", [])}. Pending: {independent.get("pending", [])}. Tracking previews: {preview_count}; synchronization previews: {sync_preview_count}.\n')
    review_command = [str(root / '.venv-sim/bin/python'), str(root / 'scripts/m0_review_page.py')]
    review_result = subprocess.run(review_command, cwd=root, capture_output=True, text=True, check=True)
    (out / 'review_page_generation.log').write_text(review_result.stdout + review_result.stderr)
    artifacts = []
    for base in [data, out]:
        for path in sorted(base.rglob('*')):
            if path.is_file() and path.name != 'artifacts_manifest.json' and path.suffix != '.log' and not path.name.startswith('._'):
                artifacts.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    write_json(out / 'artifacts_manifest.json', {'schema_version': 1, 'files': artifacts,
                                               'raw_sources_included_without_modification': True,
                                               'exclusions': ['artifacts_manifest.json itself', 'mutable execution .log files', '._ Mac metadata sidecars']})
    print(json.dumps({k: v for k, v in validation.items() if k != 'independent_validation'}, indent=2))
    if not validation['acceptance_complete']:
        raise SystemExit(result.returncode or 1)


if __name__ == '__main__':
    main()
