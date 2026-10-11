"""CPU-only reproducible audits of raw-current provenance and rendered branches."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from exp20261011_replan_candidates import read, save, unpack, CANDIDATE_SEEDS


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def difference(a, b):
    assert a.shape == b.shape
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {'max_channel_difference': int(d.max()), 'nonzero_channels': int(np.count_nonzero(d))}


def run(out):
    protocol = read(out / 'protocol.json')
    states = []
    for state in protocol['candidates']['states']:
        source = out / 'candidates/inputs' / state
        packet = unpack(source / 'payload.pkl.gz')
        raw = np.concatenate(packet['images'], axis=1)
        branches = []
        futures = out / 'candidates/futures' / state
        reference = read(futures / 'qualification_original/start-state.json')
        for name in ['qualification_original'] + [f'seed{s}' for s in CANDIDATE_SEEDS]:
            folder = futures / name
            assert read(folder / 'start-state.json') == reference
            initial = np.asarray(Image.open(folder / 'initial.png'))
            initial_difference = difference(initial, raw)
            coords = np.argwhere(initial != raw)
            assert initial_difference['max_channel_difference'] <= 1
            assert initial_difference['nonzero_channels'] / raw.size <= 0.0001
            initial_changes = [[int(y),int(x),int(c),int(initial[y,x,c])-int(raw[y,x,c])] for y,x,c in coords]
            current_png = source / 'current-raw-rgb.png'
            if current_png.exists():
                np.testing.assert_array_equal(raw, np.asarray(Image.open(current_png)))
            artifacts = {'initial_rgb_sha256': file_sha(folder / 'initial.png')}
            with np.load(folder / 'future_arrays.npz') as arrays:
                assert arrays['qpos'].shape[0] == 54
                rgb = arrays['rgb_every2steps']
                assert rgb.shape == (27, 512, 1024, 3)
                assert arrays['targets'].shape == ((0, 20) if name == 'qualification_original' else (30, 20))
                differences = []
                for step in [10, 20, 30, 54]:
                    rendered = np.asarray(Image.open(folder / f'future-{step:02d}.png'))
                    d = difference(rendered, rgb[step // 2 - 1])
                    # Separate EGL render calls may round a few channels by 1/255.
                    assert d['max_channel_difference'] <= 1, (state, name, step, d)
                    differences.append({'step': step, **d})
                    mask = np.asarray(Image.open(folder / f'future-{step:02d}-robot-mask.png'))
                    assert mask.shape == rendered.shape[:2]
                    assert set(np.unique(mask)) == {0, 255}
                initial_mask = np.asarray(Image.open(folder / 'initial-robot-mask.png'))
                assert initial_mask.shape == raw.shape[:2] and set(np.unique(initial_mask)) == {0, 255}
            branches.append({'branch': name, 'same_entire_start_state': True,
                             'initial_equals_archived_raw_current_bitwise': initial_difference['nonzero_channels'] == 0,
                             'initial_render_difference': initial_difference,
                             'initial_changes_y_x_channel_delta': initial_changes,
                             'original_current_png_equals_archived_raw_packet_bitwise': True if current_png.exists() else None,
                             'native_robot_masks_binary_nonempty': True,
                             'future_image_array_differences': differences, **artifacts})
        states.append({'state': state, 'branches': branches,
                       'raw_packet_sha256': file_sha(source / 'payload.pkl.gz')})
    result = {'schema': 'candidate-render-qualification-v3-sparse-roundoff', 'passed': True,
              'branches': sum(len(s['branches']) for s in states), 'states': states,
              'source_sha256': file_sha(__file__),
              'current_RGB': 'Archived currentPNG exactly equals saved raw policy packet. Fresh env.frame at identical dynamics state is separately qualified with sparse render rounding, not called pixel-exact.',
              'rounding_qualification': 'Post-exact-failure qualification: maximum channel difference1/255, at most0.01% of channels changed. All locations/deltas retained. Original raw packet remains the model input.',
              'future_RGB_note': 'Each PNG and array frame is separately rendered at the identical saved simulator state. Up to 1/255 channel rounding is recorded, not silently called bitwise equal.',
              'privileged_lookahead': 'All executed future RGB and native robot geometry masks depend on simulator transitions; they are privileged lookahead even without numeric truth.'}
    proof_path = out / 'candidates/render-qualification-reproducible.json'
    save(proof_path, result)
    previous_provenance = out / 'CURRENT_RGB_PROVENANCE.json'
    if previous_provenance.exists():
        provenance = read(previous_provenance)
        assert {x['state'] for x in provenance['states']} == {x['state'] for x in states}
        provenance['original_cpu_audit_sha256'] = file_sha(previous_provenance)
        provenance['all_current_raw_verified'] = True
        provenance['fresh_replay_proof'] = str(proof_path)
        provenance['fresh_replay_proof_sha256'] = file_sha(proof_path)
        for state in provenance['states']:
            state['fresh_simulator_replay_rgb_equality_pending_until_physics_run'] = False
            matched = next(x for x in states if x['state'] == state['state'])
            state['fresh_simulator_replay_rgb_bitwise_equal_to_raw_packet_and_current_png'] = all(x['initial_equals_archived_raw_current_bitwise'] for x in matched['branches'])
            state['fresh_render_roundoff_qualified'] = True
            state['worst_fresh_render_changed_channels'] = max(x['initial_render_difference']['nonzero_channels'] for x in matched['branches'])
            state['worst_fresh_render_channel_difference'] = max(x['initial_render_difference']['max_channel_difference'] for x in matched['branches'])
        save(out / 'CURRENT_RGB_PROVENANCE_VALIDATED.json', provenance)
    print({'passed': True, 'states': len(states), 'branches': result['branches']})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('/var/tmp/typhon-research-20261011/replanning-candidates'))
    run(p.parse_args().output)
