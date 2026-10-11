"""Frozen v2 offline Luna visual critic; simulator counterfactuals, never robot control."""
import argparse, base64, collections, datetime, gzip, hashlib, importlib.util, json, math, os, pickle, shutil, statistics, time, urllib.error, urllib.request
from pathlib import Path

V1_SHA = '42e1465ea1c6548222c3b876a3b019f028afdab7a21465b56f58f63b11d86343'
V1 = Path(__file__).with_name('exp20261011_ghost_critic.py')
if hashlib.sha256(V1.read_bytes()).hexdigest() != V1_SHA:
    raise RuntimeError('Frozen v1 rendering dependency changed')
spec = importlib.util.spec_from_file_location('ghost_v1', V1)
v1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v1)
read, save, sha, stamp = v1.read, v1.save, v1.sha, v1.stamp
MODEL, ENDPOINT, IDS = v1.MODEL, v1.ENDPOINT, v1.IDS
ROOT = Path('/var/tmp/typhon-research-20261011/ghost-critic-v2')
REJECTS = ['NONE_ADEQUATE', 'INSUFFICIENT_EVIDENCE']
CHOICES = IDS + REJECTS
GOAL = ('Move the orange block into the green outlined staging square. Its blue numeral 2 must finish upright in the overhead view, '
        'with the upper edge pointing toward the short green orientation tick above the square. Grasp, lift, orient, place, release and withdraw. '
        'If visibly already complete, preserve that state and avoid unnecessary manipulation.')
RUBRIC = ('Choose the best useful NEXT subgoal without losing achieved progress. For a free block: approach, align the fingers, securely grasp, then lift. '
          'For an already-held block: preserve control of it, orient and transport toward the goal, then lower to supported placement, release and withdraw. '
          'Do not infer block attachment from finger closure or wrist motion. Arm/wrist geometry alone cannot prove object rotation, support, release or completion. '
          'If adequate progress is visibly available, select the strongest supported candidate; labels are randomized and are not rankings. '
          'Use NONE_ADEQUATE only when the supplied visual evidence is sufficient to establish that every proposal lacks useful next-subgoal progress or loses achieved progress. '
          'Use INSUFFICIENT_EVIDENCE when occlusion, absent object outcomes, ambiguous views or indistinguishable evidence prevents a reliable choice or rejection of all candidates.')
REASONS = {
    'APPROACH_ALIGN': 'Visible approach or finger alignment toward a free block, without visible loss of progress.',
    'GRASP_LIFT': 'Visible object-following evidence supports a new grasp and lift, not just wrist motion.',
    'RETAIN_ORIENT_TRANSPORT': 'An already-held block remains controlled and visibly progresses in orientation or transport.',
    'SUPPORTED_PLACE_RELEASE': 'Visible supported placement or release at the required target and orientation.',
    'PRESERVE_COMPLETE': 'The visible completed state is preserved without needless manipulation.',
    'NO_USEFUL_PROGRESS': 'Sufficient evidence shows that none of the candidates makes useful next-subgoal progress.',
    'REGRESSION': 'Sufficient evidence shows that all candidates lose achieved progress.',
    'OCCLUDED_OR_AMBIGUOUS': 'The evidence is too occluded, ambiguous or indistinguishable to decide.',
    'OBJECT_OUTCOME_NOT_SHOWN': 'Arm geometry is visible but object attachment, orientation, support or release is not established.'}
THRESHOLDS = {'approach_progress_mm': 10., 'free_object_displacement_allowance_mm': 20.,
              'lift_above_start_mm': 10., 'held_position_progress_mm': 10., 'held_yaw_progress_deg': 5.,
              'held_lowering_progress_mm': 10., 'allowed_position_regression_mm': 10., 'allowed_yaw_regression_deg': 5.}


def prepare(root, candidates, base_root=None, supplement=False):
    if (root / 'protocol.json').exists():
        assert read(root / 'protocol.json')['schema'] == 'luna-ghost-critic-v2'
        return
    root.mkdir(parents=True, exist_ok=True)
    states = sorted(p.name for p in (candidates / 'inputs').iterdir() if (p / 'payload.pkl.gz').exists())
    protocol = {
        'schema': 'luna-ghost-critic-v2', 'created_utc': stamp(), 'model': MODEL, 'endpoint': ENDPOINT,
        'goal': GOAL, 'visual_rubric': RUBRIC, 'choices': CHOICES, 'reason_codes': REASONS,
        'states': states, 'supplement': supplement, 'candidate_count': 8, 'presentation_repeats': 2,
        'modes': ['ghost_only', 'future_rgb'], 'primary_mode': 'ghost_only', 'presentation_seed': 2026101107,
        'times_s': [.5, 1., 1.5], 'role': 'External offline visual critic, not native X-VLA reasoning or robot control.',
        'lookahead_scope': 'BOTH modes use privileged physically executed simulator counterfactual futures. Actual executed robot ghost geometry is also lookahead, not a learned world model or deployable prediction.',
        'input_boundary': 'Raw current camera RGB, rendered candidate cards and fixed visual task/rubric only. No held cue, truth pose, action vector, oracle score, source ID, seed, numeric task coordinate or metadata reaches the API.',
        'current_image_requirement': 'Must match unannotated env.frame RGB before policy-specific held-marker preprocessing. Supplement requires saved raw-image audit before any API request.',
        'ghost_limitation': 'Arm/wrist geometry alone cannot establish block orientation, attachment, support or release. The full arm mask does not remove this limitation.',
        'errors': 'Refusal, transport error and invalid response are separate non-decisions; never converted to either model rejection class.',
        'comparison': {'oracle_role': 'Offline comparator only; never visual critic input.', 'thresholds': THRESHOLDS,
            'availability': 'Per state, fraction/count of the eight candidates meeting the operational next-subgoal and retention proxy.',
            'selection_quality': 'Selected adequate fraction conditional on availability, selected rank/ties, and per-state uniform-random expectation vs per-state oracle best.',
            'physical_outcome': 'Report selected contact, new lift (only if initially not held), retained grasp, position/yaw error, release/stability and full task success separately.',
            'timing': 'Progress/retention at 30 executed controls (1.5s); official physical task outcome also reported after the documented extra 24-step settle/hold diagnostic.',
            'oracle_proxy': 'A declared diagnostic ordering, not a universal reward or visual ground-truth label. It cannot adjudicate whether the images themselves are sufficient.',
            'phase': 'Offline start truth derives free, held or apparent-complete phase. Initial complete uses single-state pose/release/velocity proxy, not a verified stability history.',
            'free_order': 'Full task success, retained new lift, retained bilateral grasp, then >=10mm approach without >20mm object displacement; reject lost grasp after a transient lift.',
            'held_order': 'Full task success first; otherwise retain bilateral grasp, improve position >=10mm or yaw >=5deg or lower >=10mm while aligned, with <=10mm position and <=5deg yaw regression.',
            'complete_order': 'Preserve full task success; never award unnecessary displacement.',
            'random': 'Exact uniform expectation over the same eight proposals for EACH state; state-weighted aggregates only after per-state values.',
            'oracle': 'Best lexicographic adequate/subgoal/progress candidate per state; report all tied best IDs.',
            'rejections': 'Separate NONE_ADEQUATE from INSUFFICIENT_EVIDENCE; cross-tab with oracle availability but do not call insufficiency incorrect just because an oracle action exists.'},
        'freeze_scope': 'Frozen before any model API call. The primary simulator outcomes were already available and two summaries were inspected to reconcile schema; this is a development diagnostic, not blind preregistration.',
        'limitations': ['Finite archived development states and finite eight-proposal coverage.', 'Exact simulator lookahead unavailable in a physical deployment without another predictive mechanism.', 'No classifier accuracy, latency or model benefit measured until real API calls.', 'Supplement begins with held objects; do not report initial grasp as a new pickup.'],
        'v1_preserved_sha256': V1_SHA, 'code_sha256': sha(__file__),
        'sources': ['https://developers.openai.com/api/docs/guides/decisions', 'https://developers.openai.com/api/docs/models/gpt-6-luna', 'https://arxiv.org/abs/2407.09829', 'https://arxiv.org/abs/2504.05225']}
    save(root / 'protocol.json', protocol)
    if base_root:
        for name in ['current', 'slates']:
            if (base_root / name).exists(): shutil.copytree(base_root / name, root / name)
        for p in (root / 'slates').glob('*/*/*/manifest.json'):
            m = read(p); m['current_image'] = str(root / Path(m['current_image']).relative_to(base_root))
            for x in m['candidates'].values(): x['image'] = str(root / Path(x['image']).relative_to(base_root))
            m['protocol_sha256'] = sha(root / 'protocol.json'); save(p, m)
    else:
        for state in states:
            packet_path = candidates / 'inputs' / state / 'payload.pkl.gz'
            with gzip.open(packet_path, 'rb') as f: packet = pickle.load(f)
            pixels = v1.np.concatenate(packet['images'], axis=1)
            assert pixels.shape == (512, 1024, 3) and pixels.dtype == v1.np.uint8
            dst = root / 'current' / state; dst.mkdir(parents=True, exist_ok=True)
            v1.Image.fromarray(pixels).save(dst / 'current.png')
            save(dst / 'provenance.json', {'source_payload_sha256': sha(packet_path), 'image_sha256': sha(dst / 'current.png'), 'pixel_exact_to_payload': True, 'packet_fields_used': ['images']})
    save(root / 'api_blocker.json', {'at_utc': stamp(), 'actual_model_api_calls': 0, 'reason': 'Authorized Luna credential/service location pending; no substitute model.'})
    print(json.dumps({'prepared': str(root), 'states': states, 'actual_model_api_calls': 0}))


def artifact_image(value, root):
    path = Path(value)
    if path.is_relative_to(root): return path
    # Artifact manifests retain canonical Powerglide roots when mirrored to Mint.
    for canonical in [ROOT, ROOT.with_name('ghost-critic-supplement-v2')]:
        if path.is_relative_to(canonical): return root / path.relative_to(canonical)
    raise ValueError('Image outside declared v2 artifact root')


def request_payload(manifest, root, chosen=None):
    assert manifest['mode'] in ['ghost_only', 'future_rgb']
    text = GOAL + ' ' + RUBRIC + ' The first image is unchanged CURRENT RGB: overhead left, oblique right. '
    text += 'These eight alternatives each represent 1.5 seconds, not a sequential plan. Blue arms are overlaid executed future robot poses, not simultaneous real objects. '
    text += 'BOTH modes use privileged exact simulator lookahead, not a learned prediction. '
    text += ('Only future arm geometry is overlaid; future object outcomes are NOT shown.' if manifest['mode'] == 'ghost_only' else 'Each candidate also includes raw simulator future RGB at 1.5 seconds. Judge only what it visibly establishes.')
    content = [{'type': 'input_text', 'text': text}, v1.image_part(artifact_image(manifest['current_image'], root))]
    labels = IDS if chosen is None or chosen in REJECTS else [chosen]
    for label in labels:
        content += [{'type': 'input_text', 'text': 'Candidate ' + label}, v1.image_part(artifact_image(manifest['candidates'][label]['image'], root))]
    if chosen is None:
        question = {'type': 'choice', 'name': 'best_candidate', 'instructions': RUBRIC,
            'choices': [{'value': i, 'description': 'Select visual candidate ' + i} for i in IDS] + [
                {'value': 'NONE_ADEQUATE', 'description': 'Sufficient visual evidence establishes that every candidate lacks adequate next-subgoal progress or loses achieved progress.'},
                {'value': 'INSUFFICIENT_EVIDENCE', 'description': 'Visual evidence is missing or ambiguous; cannot reliably select a candidate or reject all as inadequate.'}]}
    else:
        question = {'type': 'choice', 'name': 'visual_reason', 'instructions': 'Independently identify the strongest visible reason for the preceding decision ' + chosen + '. Do not assume that decision was correct. Use an evidence-limit reason if attachment, orientation or support is not shown.',
                    'choices': [{'value': k, 'description': v} for k, v in REASONS.items()]}
    return {'model': MODEL, 'input': [{'role': 'user', 'content': content}], 'questions': [question]}


def parse_choice(response, allowed, name):
    if response.get('model') != MODEL: raise ValueError('Unexpected or absent model identity')
    answers = [a for a in response.get('answers', []) if a.get('name') == name]
    if len(answers) != 1: raise ValueError('Expected one named answer')
    a = answers[0]
    if a.get('type') == 'refusal': return {'choice': None, 'status': 'refused', 'confidence': None}
    if a.get('type') != 'choice' or a.get('choice') not in allowed: raise ValueError('Invalid discrete choice')
    confidence = float(a['confidence'])
    probs = a.get('probabilities', [])
    if not math.isfinite(confidence) or not 0 <= confidence <= 1: raise ValueError('Invalid confidence')
    if len(probs) != len(allowed) or {p['value'] for p in probs} != set(allowed): raise ValueError('Incomplete or duplicated choices')
    if not all(math.isfinite(p['probability']) and 0 <= p['probability'] <= 1 for p in probs): raise ValueError('Invalid probability')
    if abs(sum(p['probability'] for p in probs) - 1) >= .05: raise ValueError('Probabilities do not sum to one')
    return {'choice': a['choice'], 'status': 'valid', 'confidence': confidence, 'probabilities': probs}


def call(payload, key, folder, name, timeout=45):
    save(folder / (name + '-request.json'), payload); start = time.perf_counter()
    try:
        req = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=timeout) as response: answer = json.load(response)
        save(folder / (name + '-raw.json'), answer)
    except Exception as e:
        result = {'choice': None, 'status': 'transport_error', 'error_type': type(e).__name__, 'detail': str(e).replace(key, '[REDACTED]')}
    else:
        try:
            result = parse_choice(answer, CHOICES if name == 'choice' else list(REASONS), 'best_candidate' if name == 'choice' else 'visual_reason')
            result.update(usage=answer.get('usage'), model=answer.get('model'))
        except Exception as e:
            result = {'choice': None, 'status': 'invalid_response', 'error_type': type(e).__name__, 'detail': str(e).replace(key, '[REDACTED]')}
    result.update(wall_seconds=time.perf_counter() - start, at_utc=stamp()); save(folder / (name + '.json'), result); return result


def run(root, mode, limit):
    protocol = read(root / 'protocol.json')
    audit = read(root / 'current_raw_audit.json')
    if audit.get('all_current_raw_verified') is not True: raise RuntimeError('Raw-current-image audit not verified')
    if protocol['code_sha256'] != sha(__file__): raise RuntimeError('Critic source differs from frozen protocol')
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not key:
        save(root / 'api_blocker.json', {'at_utc': stamp(), 'reason': 'OPENAI_API_KEY unavailable; no model request made', 'actual_model_api_calls': 0})
        raise SystemExit('Authorized Luna API credential unavailable; artifacts preserved')
    count = 0; reasons = 0
    for p in sorted((root / 'slates').glob('*/*/*/manifest.json')):
        if count >= limit: break
        m = read(p)
        if mode != 'both' and m['mode'] != mode: continue
        if m['protocol_sha256'] != sha(root / 'protocol.json'): raise RuntimeError('Unfrozen slate protocol')
        out = p.parent / 'luna'
        if (out / 'choice.json').exists(): continue
        result = call(request_payload(m, root), key, out, 'choice'); count += 1
        if result['status'] == 'valid': call(request_payload(m, root, result['choice']), key, out, 'reason'); reasons += 1
        print(json.dumps({'state': m['state'], 'mode': m['mode'], 'repeat': m['order_repeat'], **result}), flush=True)
        if result['status'] in ['transport_error', 'invalid_response']: break
    save(root / 'api_run_status.json', {'at_utc': stamp(), 'new_choice_requests': count, 'new_reason_requests': reasons, 'model': MODEL})


def truth_features(t, cfg):
    target = cfg['task']['target_position_xy_m']; pos = t['object_position_m']; yaw = t['object_yaw_rad']
    error = abs((yaw - cfg['task']['target_yaw_rad'] + math.pi) % (2 * math.pi) - math.pi)
    height = cfg['workspace']['table_z_m'] + cfg['object']['height_m'] / 2
    pos_error = math.dist(pos[:2], target); height_error = abs(pos[2] - height)
    q = t['object_quat_wxyz']; upright = math.acos(max(-1., min(1., 1 - 2 * (q[1] ** 2 + q[2] ** 2))))
    c, s = math.cos(yaw), math.sin(yaw); half = cfg['task']['target_region_size_m'] / 2
    footprint = all(abs(pos[0] - target[0] + c*x - s*y) <= half and abs(pos[1] - target[1] + s*x + c*y) <= half for x in [-cfg['object']['width_m']/2, cfg['object']['width_m']/2] for y in [-cfg['object']['depth_m']/2, cfg['object']['depth_m']/2])
    ev = cfg['evaluation']; released = bool(t['gripper_open'] and not t['grasp_contact'])
    pose_ok = pos_error <= ev['position_tolerance_m'] and error <= ev['yaw_tolerance_rad'] and height_error <= ev['height_tolerance_m'] and upright <= ev['upright_tolerance_rad'] and footprint
    velocity_ok = math.sqrt(sum(x*x for x in t['object_linear_velocity_m_s'])) <= ev['linear_velocity_tolerance_m_s'] and math.sqrt(sum(x*x for x in t['object_angular_velocity_rad_s'])) <= ev['angular_velocity_tolerance_rad_s']
    return {'position_error_mm': 1000*pos_error, 'yaw_error_deg': math.degrees(error), 'height_error_mm': 1000*height_error, 'pose_ok': pose_ok, 'released': released, 'bilateral_contact': bool(t['bilateral_grasp_contact']), 'apparent_complete_no_stability_history': bool(pose_ok and released and velocity_ok)}


def oracle_row(summary, start, cfg):
    a = truth_features(start, cfg); b = truth_features(summary['chunk_end_truth'], cfg)
    phase = 'apparent_complete' if a['apparent_complete_no_stability_history'] else ('held' if a['bilateral_contact'] else 'free')
    position_gain = a['position_error_mm'] - b['position_error_mm']; yaw_gain = a['yaw_error_deg'] - b['yaw_error_deg']; height_gain = a['height_error_mm'] - b['height_error_mm']
    official = summary['evaluation_after24step_settle']; success = bool(official['success']); retained = b['bilateral_contact']
    end_lift = 1000 * (summary['chunk_end_truth']['object_position_m'][2] - start['object_position_m'][2])
    moved = 1000 * math.dist(summary['chunk_end_truth']['object_position_m'], start['object_position_m'])
    progress = 0.; tier = -1; reason = 'no_adequate_progress'
    if success: tier = 4; progress = 0.; reason = 'full_task_success_after_settle'
    elif phase == 'free':
        if retained and end_lift >= THRESHOLDS['lift_above_start_mm']: tier = 3; progress = end_lift; reason = 'new_lift_retained_at_chunk_end'
        elif retained: tier = 2; progress = summary['approach_progress_mm']; reason = 'new_bilateral_grasp_retained'
        elif summary['approach_progress_mm'] >= THRESHOLDS['approach_progress_mm'] and moved <= THRESHOLDS['free_object_displacement_allowance_mm'] and not summary['physical_verified_lift']:
            tier = 1; progress = summary['approach_progress_mm']; reason = 'approach_without_large_object_displacement'
    elif phase == 'held':
        no_regression = position_gain >= -THRESHOLDS['allowed_position_regression_mm'] and yaw_gain >= -THRESHOLDS['allowed_yaw_regression_deg']
        aligning_lower = b['position_error_mm'] <= 1000*cfg['evaluation']['position_tolerance_m'] and b['yaw_error_deg'] <= math.degrees(cfg['evaluation']['yaw_tolerance_rad']) and height_gain >= THRESHOLDS['held_lowering_progress_mm']
        if retained and no_regression and (position_gain >= THRESHOLDS['held_position_progress_mm'] or yaw_gain >= THRESHOLDS['held_yaw_progress_deg'] or aligning_lower):
            tier = 2; progress = max(position_gain / THRESHOLDS['held_position_progress_mm'], yaw_gain / THRESHOLDS['held_yaw_progress_deg'], height_gain / THRESHOLDS['held_lowering_progress_mm'] if aligning_lower else 0.); reason = 'held_progress_without_position_or_yaw_regression'
    adequate = tier >= 1
    return {'phase': phase, 'adequate_proxy': adequate, 'subgoal_tier': tier, 'progress_within_tier': progress, 'proxy_reason': reason,
        'rank': [int(adequate), tier, progress], 'start_features': a, 'chunk_end_features': b, 'position_progress_mm': position_gain, 'yaw_progress_deg': yaw_gain,
        'approach_progress_mm': summary['approach_progress_mm'], 'initially_held': a['bilateral_contact'], 'new_verified_lift': bool(summary['physical_verified_lift'] and not a['bilateral_contact']),
        'retained_bilateral_contact_at_chunk_end': retained, 'physical_outcome_after24step_settle': official}


def compare(root, candidates):
    state_data = {}; rows = []
    for p in sorted((root / 'slates').glob('*/*/*/manifest.json')):
        m = read(p); state = m['state']
        if state not in state_data:
            cfg = read(candidates / 'inputs' / state / 'metadata.json')['scenario']['config']; values = {}
            for item in m['candidates'].values():
                folder = candidates / 'futures' / state / item['branch']
                values[item['branch']] = oracle_row(read(folder / 'summary.json'), read(folder / 'start-state.json')['truth'], cfg)
            n = len(values); best = max(tuple(x['rank']) for x in values.values()); adequate = sum(x['adequate_proxy'] for x in values.values())
            state_data[state] = {'candidate_count': n, 'adequate_candidate_count': adequate, 'any_adequate': adequate > 0,
                'uniform_random_expectation': {'adequate_probability': adequate/n, 'full_success_probability': sum(x['physical_outcome_after24step_settle']['success'] for x in values.values())/n,
                    'new_verified_lift_probability': sum(x['new_verified_lift'] for x in values.values())/n, 'mean_approach_progress_mm': statistics.mean(x['approach_progress_mm'] for x in values.values())},
                'oracle_best_candidates': [k for k,v in values.items() if tuple(v['rank']) == best], 'oracle_rank': list(best), 'candidate_oracle': values}
        result_file = p.parent / 'luna/choice.json'
        if not result_file.exists(): continue
        result = read(result_file); choice = result.get('choice'); branch = m['candidates'].get(choice, {}).get('branch'); data = state_data[state]; selected = data['candidate_oracle'].get(branch)
        rows.append({'state': state, 'mode': m['mode'], 'repeat': m['order_repeat'], 'choice': choice, 'status': result['status'], 'branch': branch,
            'availability': {'any_adequate': data['any_adequate'], 'adequate_candidate_count': data['adequate_candidate_count']},
            'selection_quality': None if selected is None else {'adequate': selected['adequate_proxy'], 'oracle_best': branch in data['oracle_best_candidates'], 'rank': selected['rank'], 'rank_regret_in_tier': data['oracle_rank'][2]-selected['rank'][2] if selected['rank'][:2] == data['oracle_rank'][:2] else None},
            'selected_physical_outcome': selected, 'uniform_random_expectation': data['uniform_random_expectation'], 'wall_seconds': result.get('wall_seconds')})
    modes = {}
    for mode in ['ghost_only','future_rgb']:
        subset = [x for x in rows if x['mode'] == mode]; selected = [x for x in subset if x['branch']]; available = [x for x in selected if x['availability']['any_adequate']]
        rejection = {kind: {'count': sum(x['choice'] == kind for x in subset), 'with_oracle_adequate_available': sum(x['choice'] == kind and x['availability']['any_adequate'] for x in subset)} for kind in REJECTS}
        order = []
        for state in sorted({x['state'] for x in subset}):
            pair = [x for x in subset if x['state'] == state]
            if len(pair) == 2: order.append({'state': state, 'same_decision': (pair[0]['branch'] or pair[0]['choice'], pair[0]['status']) == (pair[1]['branch'] or pair[1]['choice'], pair[1]['status'])})
        modes[mode] = {'model_decisions': len(subset), 'selected_candidates': len(selected), 'selected_adequate_given_availability': {'numerator': sum(x['selection_quality']['adequate'] for x in available), 'denominator': len(available)},
            'rejection_classes': rejection, 'nondecision_errors': dict(collections.Counter(x['status'] for x in subset if x['status'] != 'valid')), 'ordering_consistency': order,
            'median_wall_seconds': statistics.median(x['wall_seconds'] for x in subset) if subset else None,
            'state_uniform_random_adequate_probability_mean': statistics.mean(v['uniform_random_expectation']['adequate_probability'] for v in state_data.values()) if state_data else None,
            'state_oracle_any_adequate_fraction': statistics.mean(v['any_adequate'] for v in state_data.values()) if state_data else None}
    save(root / 'comparison_v2.json', {'at_utc': stamp(), 'scope': 'Offline oracle comparison; never critic inputs; development operational proxy.', 'protocol_sha256': sha(root / 'protocol.json'), 'states': state_data, 'rows': rows, 'by_mode': modes, 'actual_choice_responses': len(rows)})
    print(json.dumps({'states': len(state_data), 'actual_choice_responses': len(rows), 'by_mode': modes}), flush=True)


def selfcheck(root):
    checks = []
    def response(choice): return {'model': MODEL, 'answers': [{'type': 'choice', 'name': 'best_candidate', 'choice': choice, 'confidence': .8, 'probabilities': [{'value': x, 'probability': 1. if x == choice else 0.} for x in CHOICES]}]}
    for choice in REJECTS: assert parse_choice(response(choice), CHOICES, 'best_candidate')['choice'] == choice
    checks.append('two_rejection_classes_preserved')
    refusal = {'model': MODEL, 'answers': [{'type': 'refusal', 'name': 'best_candidate'}]}
    assert parse_choice(refusal, CHOICES, 'best_candidate')['choice'] is None; checks.append('refusal_is_not_model_rejection')
    for bad in ['UNKNOWN', 'seed752001', 'I']:
        try: parse_choice(response(bad), CHOICES, 'best_candidate'); raise AssertionError('Invalid label accepted')
        except ValueError: pass
    r = response('A'); r['model'] = 'different-model'
    try: parse_choice(r, CHOICES, 'best_candidate'); raise AssertionError('Substitute model accepted')
    except ValueError: pass
    checks.append('invalid_id_and_model_rejected')
    cfg = {'task': {'target_position_xy_m': [0.,0.], 'target_yaw_rad': 0., 'target_region_size_m': .1}, 'workspace': {'table_z_m': .8}, 'object': {'height_m': .03, 'width_m': .05, 'depth_m': .035}, 'evaluation': {'position_tolerance_m': .015, 'yaw_tolerance_rad': .17453, 'height_tolerance_m': .002, 'upright_tolerance_rad': .17453, 'linear_velocity_tolerance_m_s': .01, 'angular_velocity_tolerance_rad_s': .1}}
    def truth(x, yaw=0., held=True, z=.9):
        return {'object_position_m': [x,0.,z], 'object_yaw_rad': yaw, 'object_quat_wxyz': [math.cos(yaw/2),0.,0.,math.sin(yaw/2)], 'gripper_open': not held, 'grasp_contact': held, 'bilateral_grasp_contact': held, 'object_linear_velocity_m_s': [0.,0.,0.], 'object_angular_velocity_rad_s': [0.,0.,0.]}
    def candidate(end):
        return {'chunk_end_truth': end, 'evaluation_after24step_settle': {'success': False}, 'approach_progress_mm': 30., 'physical_verified_lift': True}
    start = truth(-.2)
    good = oracle_row(candidate(truth(-.17)), start, cfg)
    assert good['adequate_proxy'] and not good['new_verified_lift']
    assert not oracle_row(candidate(truth(-.17, held=False)), start, cfg)['adequate_proxy']
    assert not oracle_row(candidate(truth(-.17, yaw=math.radians(30))), start, cfg)['adequate_proxy']
    checks.append('held_progress_requires_retention_and_no_orientation_regression')
    checks.append('initially_held_never_credited_as_new_pickup')
    n = 0
    for p in (root / 'slates').glob('*/*/*/manifest.json'):
        m = read(p); baseline = request_payload(m, root); poisoned = json.loads(json.dumps(m)); poisoned['oracle'] = {'held': True, 'success': True, 'best': 'A', 'pose': [1, 2, 3]}
        assert request_payload(poisoned, root) == baseline
        for key in ['seed752', 'film-demo_', 'object_position', '/var/tmp', 'bilateral_grasp_contact']:
            assert key not in json.dumps(baseline)
        n += 1
    checks.append('allowlisted_request_immune_to_injected_oracle_metadata')
    save(root / 'validation_v2.json', {'at_utc': stamp(), 'checks': checks, 'slates_checked': n, 'actual_model_api_calls': 0, 'status': 'prepared_verified_pending_authorized_api_credential', 'code_sha256': sha(__file__), 'v1_dependency_sha256': sha(V1)})
    print(json.dumps({'checks': checks, 'slates_checked': n, 'actual_model_api_calls': 0}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare','build','run','compare','selfcheck']); p.add_argument('--root', type=Path, default=ROOT); p.add_argument('--candidates', type=Path, default=v1.CANDIDATES); p.add_argument('--base-root', type=Path); p.add_argument('--supplement', action='store_true'); p.add_argument('--mode', choices=['ghost_only','future_rgb','both'], default='ghost_only'); p.add_argument('--limit', type=int, default=12); a = p.parse_args()
    if a.command == 'prepare': prepare(a.root, a.candidates, a.base_root, a.supplement)
    elif a.command == 'build': v1.build(a.root, a.candidates)
    elif a.command == 'run': run(a.root, a.mode, a.limit)
    elif a.command == 'compare': compare(a.root, a.candidates)
    else: selfcheck(a.root)
