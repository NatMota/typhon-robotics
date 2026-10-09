"""Reconstruct declared virtual tabletop scenarios from immutable M0 observations.

No image coordinate is called a measured physical metre. The target-square
homography supplies dimensionless layout; an explicit engineering scale supplies
simulation metres. Run with the existing M1 Python environment, or plain Python.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supervisor.feeding_loading.config import load_config, apply_m0_contract


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def project(h, point):
    x, y = point
    denominator = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(denominator) < 1e-9:
        raise ValueError('Homography point is at infinity')
    return [(h[i][0] * x + h[i][1] * y + h[i][2]) / denominator for i in (0, 1)]


def rotate(point, angle):
    x, y = point
    return [math.cos(angle) * x - math.sin(angle) * y,
            math.sin(angle) * x + math.cos(angle) * y]


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def heading(h, point, canonical_clockwise_deg):
    # Raw portrait angle is canonical angle +90deg. Numeral's top-edge vector.
    angle = math.radians(canonical_clockwise_deg + 90)
    a = project(h, point)
    b = project(h, [point[0] + 12 * math.sin(angle), point[1] - 12 * math.cos(angle)])
    dx, dy = b[1] - a[1], b[0] - a[0]  # canonical-right, canonical-up axes
    return math.atan2(-dx, dy)  # local +y is the upper edge of the blue numeral


def map_pose(row, h, config):
    uv = project(h, [float(row['centre_x_px']), float(row['centre_y_px'])])
    displacement = rotate([uv[1] - .5, uv[0] - .5], config['world_rotation_rad'])
    xy = [config['target_origin_sim_xy_m'][i] + config['sim_target_side_m'] * displacement[i]
          for i in (0, 1)]
    yaw = wrap(heading(h, [float(row['centre_x_px']), float(row['centre_y_px'])],
                       float(row['orientation_canonical_clockwise_deg'])) + config['world_rotation_rad'])
    return {'position_xy_m': xy, 'yaw_rad': yaw, 'target_unit_xy': uv,
            'coordinate_frame': 'm15_virtual_tabletop_m_not_measured_physical_pose'}


def stable_endpoint(rows, start, end, which):
    anchor = start if which == 'initial' else end
    window = [r for r in rows if (0 <= float(r['source_time_s']) - anchor <= 1.75
              if which == 'initial' else 0 <= anchor - float(r['source_time_s']) <= 1.75)]
    valid = [r for r in window if r['detected'] == 'True'
             and r['orientation_canonical_clockwise_deg'] != ''
             and float(r['tracking_confidence']) >= .65
             and float(r['orientation_confidence']) >= .69
             and float(r['orientation_margin']) >= .045]
    pairs = list(zip(valid, valid[1:]))
    if which == 'final':
        pairs.reverse()
    for a, b in pairs:
        dt = float(b['source_time_s']) - float(a['source_time_s'])
        distance = math.hypot(float(a['target_unit_x_apparent']) - float(b['target_unit_x_apparent']),
                              float(a['target_unit_y_apparent']) - float(b['target_unit_y_apparent']))
        yaw_delta = abs(wrap(math.radians(float(a['orientation_canonical_clockwise_deg']) -
                                        float(b['orientation_canonical_clockwise_deg']))))
        if .3 <= dt <= .75 and distance <= .12 and yaw_delta <= math.radians(10):
            return (a if which == 'initial' else b), [a, b]
    return None, []


def resolved_config(scenario, mapping):
    cfg = load_config(task='feeding')
    cfg['object'].update(width_m=.032, depth_m=.043, height_m=.032, mass_kg=.0127,
                         dimensions_status='catalogue_envelope_proxy_unmeasured',
                         dimensions_source={k: 'user_confirmed_part_31110pb074_catalogue_envelope'
                                            for k in ['width_m', 'depth_m', 'height_m']})
    cfg['cameras']['overhead'].update(position_m=[0., 0., 2.2], fovy_deg=36., width=512, height=512)
    cfg['cameras']['oblique'].update(width=512, height=512)
    contract = {'schema_version': 1, 'coordinate_frame': 'm15_virtual_tabletop_m',
                'source_demo_id': scenario.get('source_demo_id'),
                'object': {'width_m': None, 'depth_m': None, 'height_m': None, 'marked_face': 'top'},
                'initial_state': {k: scenario['initial_sim'][k] for k in ['position_xy_m', 'yaw_rad']},
                'task': {'type': 'feeding', 'target_position_xy_m': scenario['target_sim']['position_xy_m'],
                         'target_yaw_rad': scenario['target_sim']['yaw_rad']}}
    cfg = apply_m0_contract(cfg, contract, {'source_frame': 'm15_virtual_tabletop_m',
        'status': 'declared_virtual_layout_mapping_not_physical_calibration',
        'rotation_rad': 0., 'translation_xy_m': [0., 0.], 'scale': 1.})
    cfg['object']['dimensions_status'] = 'catalogue_envelope_proxy_unmeasured'
    cfg['object']['mass_status'] = 'catalogue_value_not_weighed_user_piece'
    cfg['m15_mapping'] = mapping
    return cfg


def snapshot(m0, m1):
    files = []
    for label, root, directories in [
        ('m0', m0, ['data/personal/m0', 'results/personal-m0-20261009']),
        ('m1', m1, ['results/personal-m1-20261009', 'supervisor/feeding_loading', 'configs/feeding_loading'])]:
        for directory in directories:
            for p in sorted((root / directory).rglob('*')):
                if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('._'):
                    files.append({'root': label, 'path': str(p.relative_to(root)), 'bytes': p.stat().st_size,
                                  'sha256': sha(p)})
    return files


def prepare(m0, m1, output):
    if output.exists():
        raise FileExistsError('Use a fresh output directory: ' + str(output))
    dataset = json.loads((m0 / 'data/personal/m0/dataset_manifest.json').read_text())
    calibration = json.loads((m0 / 'results/personal-m0-20261009/tracking_sources/calibration.json').read_text())
    counts = Counter(d['task_category'] for d in dataset['demonstrations'])
    assert dict(counts) == {'normal_feeding': 7, 'awkward_pause_resume_feeding': 4,
                            'orientation_correction': 3, 'already_completed': 2}, counts
    assert 'demo_0017' not in {d['demo_id'] for d in dataset['demonstrations']}
    assert (m0 / 'data/personal/m0/excluded_demonstrations/demo_0017/metadata.json').exists()
    output.mkdir(parents=True)
    h0 = calibration['recordings']['recording_0005']['image_to_target_unit_square_homography']
    mapping = {'schema_version': 1, 'source_coordinate_schema': 'raw pixels; normalized image; dimensionless target-unit homography',
        'physical_metric_calibration_available': False, 'physical_to_sim_transform': None,
        'sim_target_side_m': .1, 'target_origin_sim_xy_m': [.125, -.12],
        'world_rotation_rad': -heading(h0, [280.25, 479.], 0),
        'position_equation': 'sim_xy = origin + 0.1 * R(world_rotation) * [target_v - 0.5, target_u - 0.5]',
        'yaw_equation': 'project numeral-up vector through local homography; swap unit axes; yaw=atan2(-dx,dy)+world_rotation; retain directed 2pi angle',
        'scale_status': 'engineering_layout_scale: one target-square unit represents M1 100mm staging square, not a measurement of real target',
        'origin_status': 'supplied translation chosen before execution to fit M1 workspace; common to all personal examples',
        'plane_assumption': 'resting endpoint visible centroid approximates tabletop position; object top is elevated and centroid is not contact centre',
        'endpoint_rule': 'two supported consecutive samples within first/last1.75s; dt0.3-0.75s, displacement<=0.12targetunits, yawchange<=10deg',
        'missing_observations': 'reject missing initial/final position or yaw; never replace initial with later post-action pose',
        'block': {'identity': 'DUPLO31110pb074 orange brick with blue2',
            'identity_confidence': 'user_confirmed_catalogue_identification',
            'user_confirmation': 'yeap thats he one',
            'catalogue_url': 'https://www.bricklink.com/v2/catalog/catalogitem.page?P=31110pb074',
            'catalogue_upright_envelope_m': [.032, .032, .043], 'catalogue_mass_kg': .0127,
            'sim_proxy_width_depth_height_m': [.032, .043, .032], 'sim_proxy_mass_kg': .0127,
            'orientation': 'printed face upward; original upright brick height runs along local+y toward studs',
            'measurement_status': 'catalogue estimate, user piece not measured/weighed',
            'm1_original_provisional_dimensions_m': [.05, .035, .03], 'm1_original_provisional_mass_kg': .08,
            'limitations': ['solid bounding-box collision/inertia surrogate', 'studs/hollows not modelled',
                           'catalogue packing envelope is not a precision CAD drawing', 'friction remains M1 supplied assumption']},
        'controller': 'unchanged M1 privileged geometric expert; simulator truth used for execution and independent scoring',
        'camera_adjustment': 'M15 overhead raised to2.2m,36degree field of view; both views512px. Debug-run visual review found original overhead obscured. Common settings for both cohorts; M1 unchanged.',
        'synthetic_control': {'seed': 151509, 'matching': 'one independent draw per usable personal episode, matched category and robot/settings',
            'initial_xy_domain_m': [[-.28, .28], [-.28, .25]], 'target_xy_domain_m': [[.10, .14], [-.14, -.10]],
            'yaw_domain_rad': [-math.pi, math.pi], 'personal_pose_values_used_by_sampler': False},
        'frozen_before_execution': True}
    scenarios, provenance, checks = [], [], []
    for demo in dataset['demonstrations']:
        did = demo['demo_id']; camera = demo['camera_a']; rid = camera['recording_id']
        tracking = m0 / 'data/personal/m0/demonstrations' / did / 'tracking.csv'
        rows = [r for r in csv.DictReader(tracking.open()) if r['view'] == 'camera_a']
        cal = calibration['recordings'][rid]; h = cal['image_to_target_unit_square_homography']
        for corner, expected in zip(cal['corner_pixels'], [[0, 0], [1, 0], [1, 1], [0, 1]]):
            assert max(abs(a-b) for a,b in zip(project(h, corner), expected)) < 1e-5
        for row in rows:
            if row['detected'] == 'True':
                uv = project(h, [float(row['centre_x_px']), float(row['centre_y_px'])])
                assert max(abs(uv[i] - float(row[['target_unit_x_apparent','target_unit_y_apparent'][i]])) for i in (0,1)) < 1e-4
        first, first_pair = stable_endpoint(rows, camera['start_s'], camera['end_s'], 'initial')
        last, last_pair = stable_endpoint(rows, camera['start_s'], camera['end_s'], 'final')
        item = {'scenario_id': 'personal_' + did, 'cohort': 'personal', 'source_demo_id': did,
                'category': demo['task_category'], 'pair_id': demo['pair_id'], 'seed': 151000 + int(did[-4:]),
                'reconstruction_status': 'usable' if first and last else 'unreconstructable',
                'reconstruction_confidence': 'medium_layout_only' if first and last else 'insufficient',
                'unusable_reasons': ([] if first else ['no reliable stationary initial position-and-yaw pair within1.75s']) +
                                    ([] if last else ['no reliable stationary final position-and-yaw pair within1.75s']),
                'pause_annotation': {'user_intended_pause_resume': demo['task_category']=='awkward_pause_resume_feeding',
                    'duration_s': None, 'timing_status': 'not quantified in M0', 'robot_pause_command_generated': False},
                'initial_evidence': first_pair, 'final_evidence': last_pair,
                'target_sim': {'position_xy_m': mapping['target_origin_sim_xy_m'], 'yaw_rad': 0., 'goal_source': 'observed target square and user upright2 convention'},
                'source_clips': {v: demo[v] for v in ['camera_a','camera_b']}}
        if first and last:
            item['initial_sim'] = map_pose(first, h, mapping)
            item['observed_final_sim'] = map_pose(last, h, mapping)
            item['observable_correction'] = (demo['task_category'] == 'orientation_correction' and
                abs(wrap(item['initial_sim']['yaw_rad'])) > math.radians(10) and
                abs(wrap(item['observed_final_sim']['yaw_rad'])) <= math.radians(15))
            try:
                cfg = resolved_config(item, mapping)
                dump(output / 'configs' / (item['scenario_id'] + '.json'), cfg)
            except ValueError as exc:
                item['reconstruction_status'] = 'unreconstructable'
                item['unusable_reasons'].append('M1 geometry/workspace validation: ' + str(exc))
            # Inverse virtual mapping, independent of image frame indexing.
            delta = rotate([(item['initial_sim']['position_xy_m'][i]-mapping['target_origin_sim_xy_m'][i])/.1
                            for i in (0,1)], -mapping['world_rotation_rad'])
            assert max(abs(a-b) for a,b in zip([delta[1]+.5,delta[0]+.5],item['initial_sim']['target_unit_xy'])) < 1e-10
            a0 = heading(h, [float(first['centre_x_px']),float(first['centre_y_px'])], 0)
            a180 = heading(h, [float(first['centre_x_px']),float(first['centre_y_px'])], 180)
            assert abs(abs(wrap(a180-a0))-math.pi) < 1e-8
        scenarios.append(item)
        provenance.append({'source_demo_id': did, 'scenario_id': item['scenario_id'], 'category': item['category'],
            'take_id': demo['source_trial_index'], 'source_recordings': item['source_clips'],
            'tracking_path': str(tracking.relative_to(m0)), 'tracking_sha256': sha(tracking),
            'metadata_sha256': sha(tracking.parent/'metadata.json'),
            'observation_policy': 'only M0 camera_a observed rows supply reconstruction; human trajectory is not robot action data'})
        checks.append({'source_demo_id': did, 'homography_corners_and_csv_agree': True,
                       'status': item['reconstruction_status'], 'reasons': item['unusable_reasons']})
    rng = random.Random(mapping['synthetic_control']['seed'])
    controls = []
    for personal in scenarios:
        if personal['reconstruction_status'] != 'usable':
            continue
        target = [rng.uniform(.10,.14), rng.uniform(-.14,-.10)]
        goal_yaw = rng.uniform(-math.pi,math.pi)
        category = personal['category']
        if category == 'already_completed':
            xy = [v+rng.uniform(-.005,.005) for v in target]; yaw = wrap(goal_yaw+rng.uniform(-.08,.08))
        elif category == 'orientation_correction':
            xy = [v+rng.uniform(-.007,.007) for v in target]
            yaw = wrap(goal_yaw+rng.choice([-1,1])*rng.uniform(math.pi/4,math.pi))
        else:
            xy = [rng.uniform(-.28,.28), rng.uniform(-.28,.25)]
            while math.dist(xy,target) < .08:
                xy = [rng.uniform(-.28,.28), rng.uniform(-.28,.25)]
            yaw = rng.uniform(-math.pi,math.pi)
        control = {'scenario_id': 'synthetic_' + personal['source_demo_id'], 'cohort':'synthetic',
            'source_demo_id': None, 'matched_personal_demo_id': personal['source_demo_id'], 'category':category,
            'seed':personal['seed'], 'reconstruction_status':'usable', 'reconstruction_confidence':'synthetic_exact',
            'initial_sim':{'position_xy_m':xy,'yaw_rad':yaw},
            'target_sim':{'position_xy_m':target,'yaw_rad':goal_yaw},
            'sampling_provenance':'independent fixed RNG, common predeclared workspace, no personal pose sampling/copying'}
        dump(output/'configs'/(control['scenario_id']+'.json'),resolved_config(control,mapping));controls.append(control)
    dump(output/'mapping_config.json',mapping)
    dump(output/'reconstructed_scenarios.json',{'personal':scenarios,'synthetic':controls,'excluded_source_ids':['demo_0017']})
    dump(output/'source_provenance.json',{'curated_sources':provenance,
        'excluded_example':{'demo_id':'demo_0017','reason':'user excluded short pause; retained unchanged in M0',
                            'metadata_sha256':sha(m0/'data/personal/m0/excluded_demonstrations/demo_0017/metadata.json')}})
    dump(output/'mapping_validation.json',{'passed':True,'checks':checks,'category_counts':dict(counts),
        'usable_personal':len(controls),'unreconstructable_personal':len(scenarios)-len(controls),
        'physical_metres_claimed':False,'directed_180_degree_yaw_preserved':True})
    dump(output/'input_integrity_before.json',{'files':snapshot(m0,m1)})
    dump(output/'input_locations.json',{'m0_root':str(m0.resolve()),'m1_root':str(m1.resolve())})
    print(json.dumps({'usable':len(controls),'unusable':[(x['source_demo_id'],x['unusable_reasons']) for x in scenarios if x['reconstruction_status']!='usable']}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--m0-root',type=Path,required=True);p.add_argument('--m1-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('results/personal-m15-20261009'))
    a=p.parse_args();prepare(a.m0_root.resolve(),a.m1_root.resolve(),a.output.resolve())
