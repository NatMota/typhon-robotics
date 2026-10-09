#!/usr/bin/env python3
"""Optional causal M7 RGB/proprio features and segregated simulator-oracle labels.

Run on Powerglide beside the frozen canonical NPZs. Never writes those inputs or
changes actions. The default model loader does not consume this optional export.
"""
import argparse
import collections
import copy
import hashlib
import json
import math
import pathlib
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from supervisor.adaptation.observations import object_crop

VERSION = 'm7-visual-enrichment-v1'
SENSOR_FIELDS = ('eef_position_m', 'gripper_joint_position_m')
THRESHOLDS = dict(opening_m=.055, lift_m=.02, contact_samples=3,
                  relative_stability_m=.015, release_stability_m=.004)


def sha(path):
    with pathlib.Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def field(value, confidence, basis, latest_action_index, available=True):
    return dict(value=value, confidence=confidence, basis=basis,
                latest_consumed_action_index=latest_action_index,
                runtime_eligible=available)


def sensor_features(sensor_history, previous_command, latest_action_index, hz):
    """Only whitelisted measured robot fields, never object truth or contact."""
    s = sensor_history[-1]
    opening = sum(abs(v) for v in s['gripper_joint_position_m'])
    velocity = acceleration = None
    if len(sensor_history) > 1:
        velocity = ((np.array(s['eef_position_m']) - sensor_history[-2]['eef_position_m']) * hz)
    if len(sensor_history) > 2:
        prev = (np.array(sensor_history[-2]['eef_position_m']) - sensor_history[-3]['eef_position_m']) * hz
        acceleration = ((velocity - prev) * hz).tolist()
    measured = 'archived measured robot state; these fields exist in environment.observe().proprio'
    return {
        'is_grasped': field(None, 'unknown', 'RGB and opening alone do not verify grasp', latest_action_index),
        'gripper_last_command': field(previous_command, 'exact_command' if previous_command is not None else 'unknown',
                                     'last already-executed command, +1 close/-1 open; not contact', latest_action_index),
        'gripper_opening_m': field(opening, 'exact_simulated_encoder', measured, latest_action_index),
        'eef_position_m': field(s['eef_position_m'], 'exact_simulated_kinematics', measured, latest_action_index),
        'eef_velocity_m_s': field(None if velocity is None else velocity.tolist(), 'backward_finite_difference' if velocity is not None else 'unknown',
                                 'past/current measured EEF positions, 20 Hz; no smoothing or future sample', latest_action_index),
        'eef_acceleration_m_s2': field(acceleration, 'backward_finite_difference' if acceleration is not None else 'unknown',
                                      'two past velocity differences; noise-sensitive, not force', latest_action_index),
        'object_lift_m': field(None, 'unknown', 'not inferred from a monocular orange centroid', latest_action_index),
        'object_yaw_rad': field(None, 'unknown', 'orange silhouette alone has 180-degree ambiguity; numeral estimator not implemented', latest_action_index),
        'bilateral_contact': field(None, 'unknown', 'not present in current runtime observation interface', latest_action_index),
        'object_mass_kg': field(None, 'unknown', 'no observation-derived measurement', latest_action_index),
        'contact_force_N': field(None, 'unknown', 'no runtime force sensor in source observation', latest_action_index),
        'friction_coefficient': field(None, 'unknown', 'not identifiable from these frames', latest_action_index),
    }


def oracle_states(truths, commands):
    """Sequential annotation only. 'Verified' means a declared simulator criterion."""
    z0 = truths[0]['object_position_m'][2]
    had_verified = False
    states = []
    for i, t in enumerate(truths):
        window = truths[max(0, i - 2):i + 1]
        bilateral = t['bilateral_grasp_contact']
        sustained = len(window) == 3 and all(x['bilateral_grasp_contact'] for x in window)
        height = t['object_position_m'][2] - z0
        rel = [np.array(x['object_position_m']) - x['eef_position_m'] for x in window]
        rel_change = max(float(np.linalg.norm(x - rel[0])) for x in rel)
        last_command = commands[i - 1] if i else None
        opening = sum(abs(v) for v in t['gripper_joint_position_m'])
        close = last_command is not None and last_command > 0
        position_change = max(float(np.linalg.norm(np.array(x['object_position_m']) - window[0]['object_position_m'])) for x in window)
        verified = sustained and height >= THRESHOLDS['lift_m'] and rel_change <= THRESHOLDS['relative_stability_m']
        if verified:
            state, grasped = 'VERIFIED_LIFT', True
            had_verified = True
        elif bilateral:
            state, grasped = 'BILATERAL_CONTACT', None
        elif had_verified and not close and opening > THRESHOLDS['opening_m'] and abs(height) < .01 and position_change < .004:
            state, grasped = 'RELEASED', False
            had_verified = False
        elif had_verified and close and len(window) == 3 and not any(x['grasp_contact'] for x in window):
            state, grasped = 'GRASP_LOST', False
            had_verified = False
        elif not t['grasp_contact'] and close:
            state, grasped = 'CLOSED_UNVERIFIED', False
        elif not t['grasp_contact'] and opening > THRESHOLDS['opening_m']:
            state, grasped = 'NOT_GRASPED', False
        else:
            state, grasped = 'UNKNOWN', None
        states.append(dict(state=state, is_grasped=grasped,
                           bilateral_contact=bilateral, any_finger_contact=t['grasp_contact'],
                           lift_above_initial_rest_m=height, object_relative_eef_change_m=rel_change,
                           sustained_contact_samples=len(window) if sustained else 0,
                           object_position_m=t['object_position_m'], object_yaw_rad=t['object_yaw_rad'],
                           latest_consumed_action_index=i - 1, confidence='simulator_annotation_rule_not_human_sensor_measurement'))
    return states


def rgb_features(rgb, previous, dt, frame_index, action_index):
    crop, info = object_crop(rgb)
    centroid = bbox = image_speed = None
    if info['detected']:
        x, y, w, h = info['component_bbox_xywh']
        bbox = [x, y, w, h]
        # Bounding-box centre is deliberately not presented as a calibrated pose.
        centroid = [x + w / 2, y + h / 2]
        if previous is not None and previous['object_bbox_centre_px']['value'] is not None and dt > 0:
            image_speed = ((np.array(centroid) - previous['object_bbox_centre_px']['value']) / dt).tolist()
    conf = info.get('confidence', 0.)
    basis = dict(source='current RGB orange mask only', frame_index=frame_index,
                 confidence_type='uncalibrated area/fill heuristic; partial occlusion can bias centre')
    return dict(
        detection=field(bool(info['detected']), conf, basis, action_index),
        object_bbox_xywh=field(bbox, conf, basis, action_index),
        object_bbox_centre_px=field(centroid, conf, basis, action_index),
        object_image_velocity_px_s=field(image_speed, min(conf, previous['detection']['confidence']) if image_speed is not None else 'unknown',
                                        'current and immediately previous visible detection; no interpolation across missing frames', action_index),
        visibility=field('orange_region_detected' if info['detected'] else 'unknown_or_occluded', conf, basis, action_index),
        full_object_visible=field(None, 'unknown', 'colour area alone cannot establish complete visibility', action_index),
        crop_metadata=info), crop


def make_panel(runtime, oracle=None, crop=None):
    """224 px optional third view; raw primary/secondary are never overpainted."""
    im = Image.new('RGB', (224, 224), (22, 26, 33))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    big = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 17)
    title = 'LIVE FEATURES' if oracle is None else 'ORACLE ONLY'
    d.text((7, 5), title, font=big, fill='white')
    state = 'GRASP UNKNOWN' if oracle is None else {'VERIFIED_LIFT':'GRASP VERIFIED','BILATERAL_CONTACT':'CONTACT ONLY','CLOSED_UNVERIFIED':'CLOSE / NO GRASP','NOT_GRASPED':'NOT GRASPED','RELEASED':'RELEASED','GRASP_LOST':'GRASP LOST','UNKNOWN':'GRASP UNKNOWN'}[oracle['state']]
    color = '#fdc96a' if oracle is None or oracle['is_grasped'] is None else ('#76e09b' if oracle['is_grasped'] else '#ed9a9a')
    d.text((7, 32), state, font=big, fill=color)
    opening = runtime['gripper_opening_m']['value']
    d.text((7, 62), f'OPEN {opening * 1000:.1f} mm', font=font, fill='white')
    d.rectangle((7, 84, 216, 96), outline='white')
    d.rectangle((8, 85, 8 + int(207 * min(1, opening / .08)), 95), fill='#67b5f1')
    if crop is not None:
        im.paste(Image.fromarray(crop), (7, 110))
    else:
        d.rectangle((7, 110, 103, 206), outline='#fdc96a')
        d.text((11, 138), 'UNSEEN', font=font, fill='#fdc96a')
    v = runtime['eef_velocity_m_s']['value']
    d.text((113, 110), 'EEF SPEED', font=font, fill='white')
    d.text((113, 133), '?' if v is None else f'{np.linalg.norm(v):.2f} m/s', font=font, fill='white')
    d.text((113, 165), 'YAW ?', font=font, fill='#fdc96a')
    d.text((113, 188), 'RGB CROP', font=font, fill='white')
    return im


def self_check():
    base = dict(object_position_m=[0., 0., .8], eef_position_m=[0., 0., .9],
                gripper_joint_position_m=[.016, -.016], grasp_contact=False,
                bilateral_grasp_contact=False, object_yaw_rad=0.)
    # Closing and moving an empty gripper cannot become a verified grasp.
    traces = [copy.deepcopy(base) for _ in range(12)]
    for i, t in enumerate(traces):
        t['eef_position_m'][2] += i * .01
    assert not any(x['is_grasped'] is True for x in oracle_states(traces, [1.] * 11))
    # Verification requires current/past contact, lift, stable relative pose.
    for i in range(2, 12):
        traces[i]['grasp_contact'] = traces[i]['bilateral_grasp_contact'] = True
        traces[i]['object_position_m'][2] += i * .01
    full = oracle_states(traces, [1.] * 11)
    assert full[4]['state'] == 'VERIFIED_LIFT'
    for end in range(2, len(traces)):
        assert oracle_states(traces[:end], [1.] * (end - 1)) == full[:end]
    assert all(x['latest_consumed_action_index'] == i - 1 for i, x in enumerate(full))
    return dict(empty_closing_never_verified=True, prefix_invariance=True, no_future_action_consumed=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=pathlib.Path, default=ROOT/'frozen_results/personal-m3-20261009/data-canonical')
    p.add_argument('--out', type=pathlib.Path, default=ROOT/'results/personal-m7-data-prep-20261009/visual-enrichment-v1')
    p.add_argument('--checkpoint', type=pathlib.Path, default=ROOT/'.cache-xvla/checkpoint')
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=False)
    (out/'panels').mkdir(); (out/'review').mkdir()
    checks = self_check()
    dataset = json.loads((a.source/'dataset.json').read_text())
    source_hashes = {'dataset.json': sha(a.source/'dataset.json'), 'exporter': sha(__file__),
                     'observations_helper': sha(ROOT/'supervisor/adaptation/observations.py')}
    manifest = []
    counts = collections.Counter(); coverage = collections.Counter(); selected = {}
    rows_total = 0
    with (out/'runtime_features.jsonl').open('w') as runtime_file, (out/'oracle_annotations.jsonl').open('w') as oracle_file:
        for ep in dataset['episodes']:
            if ep['split'] not in ('P_train', 'P_validation'):
                continue
            eid = ep['id']; npz_path = a.source/ep['path']
            trace_dir = a.source/'personal_generation/episode_artifacts'/eid
            path_action, path_episode = trace_dir/'robot_actions.jsonl', trace_dir/'episode.json'
            before = dict(npz=sha(npz_path), actions=sha(path_action), episode=sha(path_episode))
            assert before == dict(npz=ep['sha256'], actions=ep['source_action_sha256'], episode=ep['source_episode_sha256'])
            actions = [json.loads(x) for x in path_action.read_text().splitlines()]
            episode = json.loads(path_episode.read_text())
            truths = [episode['initial_truth']] + [x['post_action_truth'] for x in actions]
            commands = [x['normalized_command'][-1] for x in actions]
            sensors = [{key: truth[key] for key in SENSOR_FIELDS} for truth in truths]
            oracle = oracle_states(truths, commands)
            # Real-data prefix check: future events and eventual episode outcome cannot affect earlier labels.
            mid = len(truths)//2
            assert oracle_states(truths[:mid], commands[:mid-1]) == oracle[:mid]
            with np.load(npz_path) as n:
                previous = {'primary': None, 'secondary': None}; previous_k = None
                for frame, k_ in enumerate(n['frame_action_indices']):
                    k = int(k_); hz = ep['config']['control']['frequency_hz']
                    assert truths[k]['control_steps'] == episode['reset_warmup_steps'] + k
                    runtime = sensor_features(sensors[max(0, k-2):k+1], commands[k-1] if k else None, k-1, hz)
                    ident = dict(frame_id=f'{eid}:f{frame:04d}', recording_id=eid, source_demo_id=ep['source_demo_id'], split=ep['split'],
                                 frame_index=frame, pre_action_index=k, preceding_action_index=k-1,
                                 time_since_first_exported_action_s=k/hz, simulator_time_s=truths[k]['sim_time_s'],
                                 source_npz_sha256=before['npz'], source_actions_sha256=before['actions'],
                                 source_npz_uri='ssh://powerglide'+str(npz_path.resolve()))
                    rgbs = {}; crops = {}
                    for camera in ['primary', 'secondary']:
                        feat, crop = rgb_features(n[camera][frame], previous[camera], 0 if previous_k is None else (k-previous_k)/hz, frame, k-1)
                        rgbs[camera], crops[camera] = feat, crop
                        previous[camera] = feat
                        coverage[camera+'_detected'] += feat['detection']['value']
                    previous_k = k
                    runtime_row = dict(schema=VERSION, **ident, features=runtime, rgb_features=rgbs,
                                       interpretation='eligible observation channels only; simulated sensor values, not real human measurements')
                    label = oracle[k]
                    oracle_row = dict(schema=VERSION+'-oracle-only', **ident, annotations=label,
                                      input_forbidden=True, source='simulator-truth annotation; no future maxima, phases or terminal success consumed')
                    runtime_file.write(json.dumps(runtime_row, separators=(',', ':'), allow_nan=False)+'\n')
                    oracle_file.write(json.dumps(oracle_row, separators=(',', ':'), allow_nan=False)+'\n')
                    counts[label['state']] += 1; rows_total += 1
                    # Runtime-only panel is optional replacement of O2 crop slot, never a silent fourth view.
                    panel = make_panel(runtime, crop=crops['primary'])
                    panel_rel = f'panels/{eid}-f{frame:04d}-runtime.png'
                    panel.save(out/panel_rel)
                    if label['state'] not in selected:
                        selected[label['state']] = dict(identity=ident, runtime=runtime, oracle=label,
                                                        primary=n['primary'][frame].copy(), secondary=n['secondary'][frame].copy(), crop=crops['primary'])
                    if frame == 0:
                        manifest.append(dict(recording_id=eid, split=ep['split'], frame_count=len(n['primary']), source_before=before,
                                             source_npz_uri=ident['source_npz_uri']))
            after = dict(npz=sha(npz_path), actions=sha(path_action), episode=sha(path_episode))
            assert before == after
            manifest[-1]['source_after'] = after
            print(eid, 'frames', manifest[-1]['frame_count'], flush=True)
    # Use the frozen native processor; do not load model weights or touch CUDA.
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.checkpoint, trust_remote_code=True, local_files_only=True)
    probe = []; sheets = []
    for state, item in selected.items():
        live = make_panel(item['runtime'], crop=item['crop'])
        diagnostic = make_panel(item['runtime'], item['oracle'], item['crop'])
        packed = processor(images=[item['primary'], item['secondary'], np.asarray(live)], language_instruction='Grasp the orange block.')
        assert tuple(packed['image_input'].shape) == (1,3,3,224,224)
        assert packed['image_mask'].tolist() == [[True,True,True]]
        arr = packed['image_input'][0,2].permute(1,2,0).numpy()
        mean = np.array(processor.image_processor.image_mean); std = np.array(processor.image_processor.image_std)
        recovered = np.clip(np.rint((arr*std+mean)*255),0,255).astype('uint8')
        error = int(np.abs(recovered.astype(int)-np.asarray(live).astype(int)).max())
        assert error <= 1
        Image.fromarray(recovered).save(out/'review'/f'{state}-native224.png')
        diagnostic.save(out/'review'/f'{state}-ORACLE-ONLY.png')
        for camera in ['primary','secondary']:
            Image.fromarray(item[camera]).save(out/'review'/f'{state}-{camera}-raw.png')
        strip = Image.new('RGB',(896,256),(15,17,23)); d = ImageDraw.Draw(strip)
        for col,img in enumerate([Image.fromarray(item['primary']).resize((224,224)),Image.fromarray(item['secondary']).resize((224,224)),Image.fromarray(recovered),diagnostic]):
            strip.paste(img,(col*224,32))
        d.text((8,8),f'{state} | {item["identity"]["frame_id"]} | t={item["identity"]["time_since_first_exported_action_s"]:.2f}s',fill='white')
        strip.save(out/'review'/f'{state}-contact-strip.png'); sheets.append(strip)
        probe.append(dict(state=state, identity=item['identity'], image_shape=list(packed['image_input'].shape), mask=packed['image_mask'].tolist(),
                          native_round_trip_max_uint8_error=error, text_height_px=17,
                          visual_review_required=True, interpretation='tensor/legibility check only; not proof the VLA understands the panel'))
    sheet = Image.new('RGB',(896,256*len(sheets)),(15,17,23))
    for i,strip in enumerate(sheets): sheet.paste(strip,(0,i*256))
    sheet.save(out/'review'/'contact_sheet.png')
    write_json(out/'source_manifest.json',dict(schema=VERSION, source_hashes=source_hashes, episodes=manifest,
                                               field_provenance='Each field carries a basis, confidence and last consumed action index. IDs align to original RGB pre-action indices.'))
    write_json(out/'native224_probes.json',probe)
    write_json(out/'schema_and_config.json',dict(schema=VERSION, thresholds=THRESHOLDS, native_view_contract=['primary raw RGB','secondary raw RGB','optional runtime panel replaces crop'],
                                                default_model_loader_unchanged=True, oracle_annotations_forbidden_as_model_inputs=True,
                                                measured_sensor_source='whitelisted EEF/gripper values in archived truth also provided by observe().proprio; not NPZ command-history proprio',
                                                runtime_grasp='UNKNOWN pending validated sensor/visual estimator; no oracle contact copied to model',
                                                future_features=['validated numeral yaw','calibrated goal-relative image/table position','causal visual co-motion/lift/slip evidence','joint angles and velocities when consistently exported']))
    write_json(out/'validation_summary.json',dict(passed=True, checks=checks, frame_count=rows_total, episode_count=len(manifest),
                                                  oracle_state_counts=dict(counts), rgb_detection_counts=dict(coverage),
                                                  all_source_hashes_unchanged=True, all_runtime_grasp_unknown=True,
                                                  representative_states=list(selected), missing_representative_states=[s for s in ['GRASP_LOST'] if s not in selected],
                                                  native224_probes=len(probe), training_started=False))
    (out/'REPRODUCE.txt').write_text('Run from Powerglide humanoid-m6-joint; outputs must not exist:\n.venv-xvla/bin/python scripts/m7_prepare_visual_enrichment.py --out results/personal-m7-data-prep-20261009/visual-enrichment-v1\nNo model weights are loaded. Dependencies: existing numpy, OpenCV, Pillow, Transformers and pinned native processor. Copy this output directory to Mint Fast Drive. No original data or targets are altered.\n')
    report = f'''M7 optional visual enrichment v1 — data preparation only\n\n{len(manifest)} existing canonical generated-expert episodes, {rows_total} synchronized source frames.\nOracle states: {dict(counts)}. RGB detections: {dict(coverage)}.\n\nThe runtime file and panels contain only current/past RGB and whitelisted measured EEF/gripper state. Closing is not grasping: runtime is_grasped is explicitly UNKNOWN in every row. Bilateral contact and object truth are kept only in oracle_annotations.jsonl and ORACLE ONLY review panels. Verified lift requires three consecutive bilateral-contact samples, >=20 mm lift above initial rest, and <=15 mm variation in object-to-EEF displacement. Contact without lift is distinct. Release and loss use only past/current state. No whole-episode max lift, eventual success or future phase is fed to earlier observations.\n\nMeasured EEF and finger opening come from archived trace fields that are also exposed by environment.observe().proprio. They do not come from the NPZ's previous-command history vector. Encoder-derived velocity/acceleration use backward differences and can be noisy. The orange bounding-box centre is an image observation, not a physical centre or depth estimate. Full visibility, yaw, force, mass, friction and runtime lift remain unknown. No hand or human physical quantities are invented.\n\nPanels are a separate optional third-view variant alongside unchanged primary and secondary RGB. This replaces the O2 crop slot with a panel containing that crop; it is not a fourth view or a change to the default loader. Labels are 17 px at native224. The exact frozen processor preserves panels within <=1 uint8 level. This validates formatting only, not model understanding or task improvement. Inspect review/contact_sheet.png.\n\nAll nine episodes are successful source-expert executions; transient close-without-contact frames are not a held-out failed-grasp episode. Grasp loss has no representative unless reported above, so no observed-loss capability is claimed. The scripted self-check tests loss-relevant temporal causality, but does not create training data. Future grasp-estimator training needs failed/occluded examples and validated runtime sensing. No training, controller changes or simulation runs occurred.\n\nAll source NPZ/action/episode hashes match the frozen manifest and match again after reading. Reproduction and field provenance are saved.\n'''
    (out/'REPORT.txt').write_text(report)
    files=[p for p in out.rglob('*') if p.is_file()]
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in sorted(files)))
    print(report)

if __name__ == '__main__':
    main()
