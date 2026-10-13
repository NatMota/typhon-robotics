"""Private M7 checkpoint inference pipe; robot/RGB/request inputs only."""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time
import traceback


def prepare_inputs(processor, payload, config, device):
    import torch
    from .m7_training import request_text, validate_request
    from .m7_visual import apply_held_crosshair
    from .observations import views
    from .m6_contract import reorder
    allowed = {'op', 'images', 'instruction', 'proprio', 'request', 'history', 'annotation', 'camera_variant'}
    if set(payload) - allowed:
        raise ValueError('Unapproved worker input fields: ' + str(set(payload) - allowed))
    request = validate_request(payload['request'], inference=True)
    primary, secondary = payload['images']
    rgb = {'primary': primary, 'secondary': secondary}
    annotation = payload['annotation']
    if config['condition'] == 'crosshair':
        if annotation['source'] != 'explicit_annotation_conditioned_v2':
            raise ValueError('Crosshair evaluation requires explicit annotation-conditioned V2 mode')
        rgb, metadata = apply_held_crosshair(rgb, annotation)
        held, held_valid = metadata['held'], metadata['held_valid']
    else:
        held, held_valid = None, False
        metadata = {'condition': 'raw', 'held': None, 'held_valid': False}
    main, auxiliary = ('primary', 'secondary') if payload['camera_variant'] == 'overhead_main' else ('secondary', 'primary')
    inputs = processor(images=views({'primary': rgb[main], 'secondary': rgb[auxiliary]}, 'O2'),
                       language_instruction=request_text(payload['instruction'], request))
    proprio = reorder(torch.as_tensor(payload['proprio'], dtype=torch.float32)).unsqueeze(0)
    proprio[:, [9, 19]] = 0.
    inputs.update(proprio=proprio, domain_id=torch.tensor([29], dtype=torch.long))
    inputs = {key: value.to(device) for key, value in inputs.items()}
    history_images, times, cameras = [], [], []
    for frame in payload['history']:
        if not frame['time_s'] < 0:
            raise ValueError('History must be strictly past')
        past = dict(zip(['primary', 'secondary'], frame['images']))
        if config['condition'] == 'crosshair':
            if frame['annotation']['source'] != 'explicit_annotation_conditioned_v2':
                raise ValueError('Unlabelled crosshair history')
            past, _ = apply_held_crosshair(past, frame['annotation'])
        for camera in (main, auxiliary):
            history_images.append(past[camera]); times.append(frame['time_s'])
            cameras.append(0 if camera == 'primary' else 1)
    if history_images:
        if config['history_mode'] == 'none':
            raise ValueError('History supplied to a no-history checkpoint')
        inputs['history_pixels'] = processor.image_processor(history_images, return_tensors='pt')['pixel_values'].to(device)
        inputs['history_times_s'] = torch.tensor(times, dtype=torch.float32, device=device)
        inputs['history_camera_ids'] = torch.tensor(cameras, dtype=torch.long, device=device)
    return inputs, held, held_valid, metadata


def processor_smoke(output):
    """Actual pinned processor/masks/history on CPU; no checkpoint or GPU model load."""
    from pathlib import Path
    import json
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from .m7_visual import apply_held_crosshair, self_check
    from .observations import views
    output = Path(output)
    request = json.loads((output.parent / 'protocol.json').read_text())['requested_action']
    source_root = Path('frozen_results/personal-m3-20261009/data-canonical')
    entries = json.loads((source_root / 'dataset.json').read_text())['episodes']
    entry = next(entry for entry in entries if entry['id'] == 'personal_demo_0006')
    with np.load(source_root / entry['path']) as source:
        images = [source[key][4].copy() for key in ['primary', 'secondary']]
        past = [[source[key][i].copy() for key in ['primary', 'secondary']] for i in range(4)]
        proprio = source['proprio'][4].copy()
    before = [image.copy() for image in images]
    processor = AutoProcessor.from_pretrained('.cache-xvla/checkpoint', trust_remote_code=True, local_files_only=True)
    processor.language_max_length = 256
    records = []
    for condition, label in [('raw', None), ('crosshair', False), ('crosshair', None), ('crosshair', True)]:
        for camera in ['overhead_main', 'oblique_main']:
            annotation = {'held': label, 'source': 'explicit_annotation_conditioned_v2', 'runtime_available': False}
            payload = {'op': 'predict', 'images': images, 'instruction': entry['instruction'], 'proprio': proprio, 'request': request,
                'history': [{'images': rgb, 'time_s': (i-4)/5, 'annotation': annotation} for i, rgb in enumerate(past)],
                'annotation': annotation, 'camera_variant': camera}
            inputs, held, valid, metadata = prepare_inputs(processor, payload, {'condition': condition, 'history_mode': 'dense'}, 'cpu')
            assert inputs['proprio'].shape == (1,20) and inputs['proprio'][0,9] == 0 and inputs['proprio'][0,19] == 0
            assert inputs['history_pixels'].shape == (8,3,224,224) and (inputs['history_times_s'] < 0).all()
            expected_ids = [0,1]*4 if camera == 'overhead_main' else [1,0]*4
            assert inputs['history_camera_ids'].tolist() == expected_ids
            assert held == label and valid is (condition == 'crosshair' and label is not None)
            expected_rgb = dict(zip(['primary', 'secondary'], images))
            if condition == 'crosshair':
                expected_rgb, _ = apply_held_crosshair(expected_rgb, annotation)
            if camera == 'oblique_main':
                expected_rgb = {'primary': expected_rgb['secondary'], 'secondary': expected_rgb['primary']}
            expected = processor(images=views(expected_rgb, 'O2'), language_instruction='check')
            assert torch.equal(expected['image_input'], inputs['image_input']) and torch.equal(expected['image_mask'], inputs['image_mask'])
            records.append({'condition': condition, 'held': label, 'camera': camera, 'image_mask': inputs['image_mask'].tolist(),
                'input_shapes': {key: list(value.shape) for key, value in inputs.items()}, 'history_camera_ids': expected_ids})
    for key, value in [('targets', [1.]), ('truth', {'object_pose': [1.]})]:
        try:
            prepare_inputs(processor, {**payload, key: value}, {'condition': 'raw', 'history_mode': 'dense'}, 'cpu')
        except ValueError:
            pass
        else:
            raise AssertionError('Unapproved labels admitted')
    assert all(np.array_equal(a, b) for a, b in zip(before, images)) and self_check()
    report = {'passed': True, 'evidence_type': 'CPU native processor, no model loaded', 'source': 'personal_demo_0006 current/past RGB only',
        'cases': records, 'native_image_mask_preserved': True, 'held_unknown_validity_preserved': True,
        'truth_and_target_payload_rejected': True, 'raw_rgb_unchanged': True, 'history_camera_order_matches_training': True}
    output.write_text(json.dumps(report, indent=2) + '\n')
    return {'passed': True, 'cases': len(records), 'output': str(output)}


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--audit-dir', required=True)
    parser.add_argument('--check-processor')
    parser.add_argument('--annotation-conditioned-v2', action='store_true')
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()
    if args.check_processor:
        import json
        print(json.dumps(processor_smoke(args.check_processor))); return
    if not args.checkpoint:
        parser.error('--checkpoint or --check-processor required')
    wire = sys.stdout.buffer
    sys.stdout = sys.stderr
    def reply(value):
        pickle.dump(value, wire, protocol=4); wire.flush()
    try:
        import torch
        import numpy as np
        audit_dir=Path(args.audit_dir);audit_dir.mkdir(parents=True,exist_ok=False);audit_index=0
        captured_noise=[];native_randn=torch.randn
        def audit_randn(*a,**kw):
            value=native_randn(*a,**kw);captured_noise.append(value.detach().cpu().numpy().copy());return value
        torch.randn=audit_randn
        from transformers import AutoProcessor
        from .m7_training import generate_requested, HISTORY_AUDIT, load_checkpoint
        start = time.perf_counter()
        model, projection, restored = load_checkpoint(args.checkpoint, Path('.cache-xvla/checkpoint'))
        if restored['domain_id'] != 29:
            raise ValueError('Unsupported M7 domain')
        config = restored['config']
        if (config['condition'] == 'crosshair') != args.annotation_conditioned_v2:
            raise ValueError('Crosshair checkpoint requires explicit annotation-conditioned V2; raw forbids it')
        metadata = {key: restored[key] for key in ['schema', 'base_revision', 'step', 'config']}
        metadata.update(model_state_strict_loaded=True, physical_time_projection_strict_loaded=True,
                        history_adapter='fixed timestamp/camera encoding; same native feature weights and append_causal_history as training')
        # The simulator runtime intentionally has no torch. TorchVersion is a
        # str subclass whose pickle import otherwise requires torch there.
        metadata = json.loads(json.dumps(metadata, default=str))
        del restored
        model.eval(); projection.eval()
        processor = AutoProcessor.from_pretrained('.cache-xvla/checkpoint', trust_remote_code=True, local_files_only=True)
        processor.language_max_length = 256
        reply({'ready': True, 'load_time_s': time.perf_counter()-start, 'checkpoint': args.checkpoint, **metadata})
        while True:
            payload = pickle.load(sys.stdin.buffer)
            if payload['op'] == 'close':
                break
            if payload['op'] == 'reset':
                torch.manual_seed(payload['seed']); torch.cuda.manual_seed_all(payload['seed'])
                HISTORY_AUDIT.clear(); reply({'reset': True}); continue
            try:
                start = time.perf_counter()
                inputs, held, valid, annotation = prepare_inputs(processor, payload, config, 'cuda')
                captured_noise.clear()
                tensors={k:v.detach().cpu().numpy().copy() for k,v in inputs.items()}
                tensors['rng_cpu_before']=torch.get_rng_state().numpy().copy()
                tensors['rng_cuda_before']=torch.cuda.get_rng_state().cpu().numpy().copy()
                torch.cuda.synchronize(); inference_start = time.perf_counter()
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    timed = generate_requested(model, projection, inputs, payload['request'], held, valid, steps=args.steps)
                torch.cuda.synchronize()
                tensors.update({f'noise_{i}':v for i,v in enumerate(captured_noise)})
                tensors['rng_cpu_after']=torch.get_rng_state().numpy().copy()
                tensors['rng_cuda_after']=torch.cuda.get_rng_state().cpu().numpy().copy()
                np.savez_compressed(audit_dir/f'{audit_index:04d}.npz',**tensors);audit_index+=1
                reply({'timed': timed, 'inference_s': time.perf_counter()-inference_start,
                       'preprocess_and_inference_s': time.perf_counter()-start,
                       'image_mask': inputs['image_mask'].cpu().tolist(),
                       'input_shapes': {key: list(value.shape) for key, value in inputs.items()},
                       'history_camera_ids': inputs.get('history_camera_ids', torch.tensor([])).cpu().tolist(),
                       'history_times_s': inputs.get('history_times_s', torch.tensor([])).cpu().tolist(),
                       'history_audit': list(HISTORY_AUDIT), 'annotation': annotation})
            except Exception:
                reply({'error': traceback.format_exc()})
    except Exception:
        reply({'error': traceback.format_exc()}); raise SystemExit(1)


if __name__ == '__main__':
    main()
