"""Data-only semantic packing; deliberately not a stock X-VLA training batch.

Valid anchors preserve source values and timing. Zero padding is never a target.
Future scope, intent and duration stay under supervision, outside observations.
NumPy/Torch are imported only when reading RGB or materializing a native batch.
"""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

HORIZON = 30
CONTROL_HZ = 20
ROTATION_ORDER = (0, 3, 1, 4, 2, 5)  # Exact M6 concatenated -> generic interleaved permutation.
CAMERAS = {'overhead_main': ('primary', 'secondary'), 'oblique_main': ('secondary', 'primary')}
INSTRUCTIONS = {
    'approach_and_align': 'Approach the block and align the gripper.',
    'descend_to_grasp': 'Lower the gripper to the grasp pose.',
    'grasp_close': 'Close the gripper on the block.',
    'lift': 'Lift the block.',
    'transport_and_reorient': 'Move the block toward the destination while reorienting it.',
    'place_lower': 'Lower the block into the target region.',
    'release_open': 'Open the gripper to release the block.',
    'retreat': 'Move the gripper away from the placed block.',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_gzip(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def reorder(values):
    result = list(values)
    for start in (3, 13):
        result[start:start+6] = [values[start+i] for i in ROTATION_ORDER]
    return result


class M7Dataset:
    def __init__(self, export_dir, source_snapshot, split=None):
        self.export_dir = Path(export_dir)
        self.source_snapshot = Path(source_snapshot)
        self.source_hashes = {}
        for line in (self.export_dir/'SHA256SUMS').read_text().splitlines():
            expected, name = line.split('  ', 1)
            path = self.export_dir/name
            if sha(path) != expected:
                raise ValueError(f'Frozen semantic export hash mismatch: {path}')
            self.source_hashes[str(path)] = expected
        report = json.loads((self.export_dir/'report.json').read_text())
        expected = [v for k, v in report['source_files_before'].items() if k.endswith('/temporal/source_snapshot.json.gz')]
        if len(expected) != 1 or sha(self.source_snapshot) != expected[0]:
            raise ValueError('Source snapshot does not match semantic export provenance')
        self.source_hashes[str(self.source_snapshot)] = expected[0]
        snapshot = read_gzip(self.source_snapshot)
        self.sources = {e['id']: e for e in snapshot['episodes']}
        with (self.export_dir/'index.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        if split is not None and split not in ['P_train', 'P_validation']:
            raise ValueError('Only frozen P_train or P_validation splits are supported')
        self.rows = [r for r in rows if split is None or r['split'] == split]
        self.by_id = {r['sample_id']: r for r in self.rows}
        if len(self.by_id) != len(self.rows):
            raise ValueError('Duplicate semantic sample identifiers')
        self.target_arrays = read_gzip(self.export_dir/'targets.json.gz')
        self._rgb_cache = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.pack(self.rows[index]['sample_id'])

    def pack(self, sample_id, camera_variant='overhead_main'):
        if camera_variant not in CAMERAS:
            raise ValueError(f'Unknown declared camera variant: {camera_variant}')
        row = self.by_id[sample_id]
        source = self.sources[row['recording_id']]
        if row['split'] != source['split'] or row['source_npz_sha256'] != source['sha256']:
            raise ValueError('Source split or NPZ hash mismatch')
        frame, start = int(row['frame_index']), int(row['action_start'])
        semantic_start, end = int(row['semantic_start']), int(row['action_end_exclusive'])
        indices = [int(i) for i in row['anchor_action_indices'].split()]
        native = self.target_arrays[int(row['targets_row'])]
        if source['frame_action_indices'][frame] != start or not 0 <= semantic_start-start <= 3:
            raise ValueError('Observation is not aligned before first exported command')
        if not 0 <= start <= semantic_start < end <= source['actions']:
            raise ValueError('Invalid same-take semantic range')
        if len(indices) != len(native) or len(indices) != min(HORIZON, end-start):
            raise ValueError('Anchor count does not match the declared horizon')
        if indices != sorted(set(indices)) or indices[0] != start or indices[-1] != end-1:
            raise ValueError('Invalid anchor index order or endpoints')
        if native != [source['targets'][i] for i in indices]:
            raise ValueError('Export anchors differ from original targets')
        count = len(native)
        valid = [True]*count+[False]*(HORIZON-count)
        action_source = [list(a) for a in native]+[[0.0]*20 for _ in range(HORIZON-count)]
        action = [reorder(a) for a in action_source]
        proprio_source = list(source['proprio'][frame])
        proprio = reorder(proprio_source)
        proprio[9] = proprio[19] = 0.0  # Native X-VLA input contract, not action labels.
        main, auxiliary = CAMERAS[camera_variant]
        camera_ref = {'source_npz_uri': source['source_npz_uri'], 'source_npz_name': source['path'],
                      'source_npz_sha256': source['sha256'], 'frame_index': frame,
                      'action_start': start, 'main_array': main, 'auxiliary_array': auxiliary,
                      'crop_source_array': main, 'camera_variant': camera_variant,
                      'observation_timing': 'captured before action_start',
                      'geometry': 'unchanged original pixel axes and world action coordinates'}
        return {
            'observation': {'proprio': proprio, 'task_instruction': source['instruction'], 'camera_ref': camera_ref},
            'supervision': {
                'action': action, 'action_source': action_source, 'valid_mask': valid,
                'semantic_mask': [i >= semantic_start for i in indices]+[False]*(HORIZON-count),
                'timestamps_s': [(i-start)/CONTROL_HZ for i in indices]+[0.0]*(HORIZON-count),
                'source_action_indices': indices+[-1]*(HORIZON-count), 'true_length': count,
                'duration_seconds': (end-semantic_start)/CONTROL_HZ,
                'aligned_duration_seconds': (end-start)/CONTROL_HZ,
                'scope': row['representation'], 'primitive_count': int(row['primitive_count']),
                'semantic_instruction': ' '.join(INSTRUCTIONS[label] for label in row['component_labels'].split('|')),
                'component_labels': row['component_labels'].split('|'),
                'component_primitive_ids': row['component_primitive_ids'].split('|'),
                'conditioning_source': 'offline demonstration labels; not observed state or inference oracle'},
            'provenance': {
                'sample_id': sample_id, 'example_id': sample_id+':'+camera_variant,
                'recording_id': row['recording_id'], 'split': row['split'], 'frame_index': frame,
                'action_start': start, 'semantic_start': semantic_start, 'end': end,
                'context_prefix_commands': semantic_start-start, 'source_npz_sha256': source['sha256'],
                'source_action_sha256': row['source_action_sha256'], 'source_episode_sha256': row['source_episode_sha256'],
                'proprio_source': proprio_source, 'rotation_source': 'concatenated_columns',
                'rotation_model': 'generic_interleaved', 'source_control_hz': CONTROL_HZ},
            'integration': {'stock_trainer_compatible': False, 'direct_20hz_execution_allowed': False,
                            'required': ['unreduced masked loss', 'padding-aware model integration',
                                         'non-oracle intent/horizon interface', 'timestamp-aware reconstruction']}}

    def images(self, packed, source_root):
        """Resolve the hash-referenced NPZ locally; no network or source writes."""
        import numpy as np
        from .observations import object_crop
        ref = packed['observation']['camera_ref']
        path = Path(source_root)/ref['source_npz_name']
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, ref['source_npz_sha256'])
        # ponytail: one take cached in RAM; streaming/mmap is a future training throughput concern.
        if self._rgb_cache is None or self._rgb_cache[0] != key:
            if sha(path) != ref['source_npz_sha256']:
                raise ValueError(f'Canonical RGB source hash mismatch: {path}')
            with np.load(path, allow_pickle=False) as archive:
                indices = archive['frame_action_indices']
                rgb = {camera: archive[camera] for camera in ['primary', 'secondary']}
            if path.stat().st_size != stat.st_size or path.stat().st_mtime_ns != stat.st_mtime_ns:
                raise ValueError('Canonical RGB source changed while reading')
            self._rgb_cache = (key, indices, rgb)
        _, indices, rgb = self._rgb_cache
        frame = ref['frame_index']
        if int(indices[frame]) != ref['action_start']:
            raise ValueError('Camera frame does not precede the expected source action')
        images = [rgb[name][frame].copy() for name in [ref['main_array'], ref['auxiliary_array']]]
        if any(x.shape != (512, 512, 3) or x.dtype != np.uint8 for x in images):
            raise ValueError('Expected synchronized native512x512 RGB views')
        crop, _ = object_crop(images[0])
        return images if crop is None else images+[crop]

    def collate(self, samples, processor, source_root, *, domain_id, inspection_only=False):
        """Produce explicit nested CPU tensors; does not call a model or flatten labels into observations."""
        if inspection_only is not True:
            raise RuntimeError('M7 batches are inspection-only: pass inspection_only=True. Training requires explicit legal requested-intent/horizon conditioning plus mask/time-aware integration; task-only mixed scopes are ambiguous.')
        import torch
        if not samples or len({p['provenance']['split'] for p in samples}) != 1:
            raise ValueError('A batch must contain exactly one nonempty frozen split')
        if not isinstance(domain_id, int) or not 0 <= domain_id < 30:
            raise ValueError('Provide an explicit domain table row0..29 for inspection')
        observation = dict(processor(images=[self.images(p, source_root) for p in samples],
            language_instruction=[p['observation']['task_instruction'] for p in samples]))
        observation['proprio'] = torch.tensor([p['observation']['proprio'] for p in samples], dtype=torch.float32)
        observation['domain_id'] = torch.full((len(samples),), domain_id, dtype=torch.long)
        dtypes = {'action': torch.float32, 'action_source': torch.float32,
                  'valid_mask': torch.bool, 'semantic_mask': torch.bool,
                  'timestamps_s': torch.float64, 'source_action_indices': torch.long}
        supervision = {key: torch.tensor([p['supervision'][key] for p in samples], dtype=dtype) for key, dtype in dtypes.items()}
        metadata = [{**p['provenance'], 'semantic_supervision': {k: v for k, v in p['supervision'].items() if k not in dtypes},
                     'integration': p['integration']} for p in samples]
        return {'observation': observation, 'supervision': supervision, 'metadata': metadata}


def training_batch(*args, **kwargs):
    raise RuntimeError('Stock training is unsupported: mixed semantic scopes need legal intent/horizon conditioning, and native forward/loss/decoder need masks and timestamps.')


def masked_mean_loss(unreduced_loss, valid_mask):
    """Exclude invalid slots BEFORE reduction; cannot mask stock already-reduced losses."""
    import torch
    if valid_mask.dtype != torch.bool or unreduced_loss.shape[:valid_mask.ndim] != valid_mask.shape:
        raise ValueError('Boolean valid mask must match the leading unreduced loss dimensions')
    mask = valid_mask
    while mask.ndim < unreduced_loss.ndim:
        mask = mask.unsqueeze(-1)
    selected = unreduced_loss.masked_select(mask.expand_as(unreduced_loss))
    if selected.numel() == 0:
        raise ValueError('Cannot reduce a batch with no valid targets')
    return selected.mean()


def export_package(export_dir, snapshot, output):
    dataset = M7Dataset(export_dir, snapshot)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = [dataset.pack(row['sample_id'], variant) for row in dataset.rows for variant in CAMERAS]
    padded = sum(not m for p in records for m in p['supervision']['valid_mask'])
    for record in records:
        s = record['supervision']
        for key in ['action', 'action_source', 'valid_mask', 'semantic_mask', 'timestamps_s', 'source_action_indices']:
            assert len(s[key]) == HORIZON
        for i, valid in enumerate(s['valid_mask']):
            if not valid:
                assert s['action'][i] == s['action_source'][i] == [0]*20
                assert not s['semantic_mask'][i] and s['timestamps_s'][i] == 0 and s['source_action_indices'][i] == -1
        assert set(record['observation']) == {'proprio', 'task_instruction', 'camera_ref'}
    with (output/'packed_records.json.gz').open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding='utf-8') as stream:
                json.dump(records, stream, separators=(',', ':'), allow_nan=False)
    assert read_gzip(output/'packed_records.json.gz') == records
    assert all(sha(path) == value for path, value in dataset.source_hashes.items())
    report = {'schema': 'm7-model-input-v1', 'data_only': True, 'training_started': False,
              'records': len(records), 'semantic_samples': len(dataset), 'slots': HORIZON, 'action_dimensions': 20,
              'records_by_split': {split: sum(p['provenance']['split'] == split for p in records) for split in ['P_train', 'P_validation']},
              'padded_slots': padded, 'ragged_records_padded': sum(p['supervision']['true_length'] < HORIZON for p in records),
              'source_hashes': dataset.source_hashes, 'loader_sha256': sha(Path(__file__)),
              'rotation': 'action_source is exact unchanged source concatenated columns; action/proprio use one exact M6 permutation to generic interleaved. No scaling or second conversion.',
              'proprio': 'Pre-action first9 pose follows source measured-first/previous-command-history contract. Input grippers9/19 explicitly zero. Source proprio retained in provenance.',
              'padding': 'Exactly30 slots. Invalid action/source rows zero, source indices-1, timestamps0, masksfalse. valid_mask is mandatory for both target and timestamp use; zero is not a real padded command.',
              'timing': 'Valid timestamps are (source_action_index-action_start)/20 seconds. Sparse anchors are irregular; never consecutive20Hz motor commands.',
              'labels_not_observations': 'Semantic instruction, scope, primitive count and demonstrated duration are supervision/metadata only. Observation contains task text, current camera reference and source proprio. Inference intent/horizon must be externally requested or predicted from allowed current/history inputs.',
              'cameras': 'Declared main/aux routes; both same frame before action_start. O2 crop derives only from selected main. Up to3 source context commands precede exact semantic_start.',
              'conditional_task_boundary': 'The same current observation can label multiple semantic scopes. Task-only training is ambiguous. Training could condition on annotated requested intent under an explicitly conditional task, but inference requests must come from user/planner/predictor, not evaluator futures. collate requires inspection_only=True; training_batch always refuses.',
              'stock_trainer_compatible': False,
              'integration_boundary': 'Stock forward/reduced loss does not consume masks or anchor times. masked_mean_loss applies only to unreduced element losses; future integration must also handle padding in the model, non-oracle horizon conditioning and timing-aware reconstruction. No training or decoder integration performed.',
              'verification': {'all_packed_valid_values_equal_frozen_export_and_original_snapshot': True,
                               'padding_and_masks_verified': True, 'saved_pack_roundtrip_equal': True,
                               'frozen_files_unchanged': True, 'no_rgb_duplicated': True},
              'native_batch_verification': 'Performed separately by scripts/m7_inspect_batches.py; this report does not claim a processor/model run.'}
    (output/'data_contract.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'example_short.json').write_text(json.dumps(next(p for p in records if p['supervision']['true_length'] < HORIZON), indent=2)+'\n')
    (output/'README.txt').write_text(
        'M7 data package:396 declared camera examples from198 semantic sequences. DATA ONLY.\n'
        'Read packed_records.json.gz or import M7Dataset from supervisor.adaptation.m7_data.\n'
        'M7Dataset(export_dir, source_snapshot, split=None).pack(sample_id, camera_variant) returns nested observation/supervision/provenance.\n'
        'dataset.images(record, canonical_npz_root) verifies the source hash and returns current coherent O2 camera inputs. No images are copied into this package.\n'
        'dataset.collate(records, native_processor, canonical_npz_root, domain_id=29, inspection_only=True) materializes nested CPU tensors. Domain29 here is an explicit inspection argument, not a training decision.\n'
        'Default collate and training_batch refuse training use. Mixed scopes need an explicit legal requested-intent/horizon interface; task-only conditioning is insufficient.\n'
        'Targets have30 slots; valid_mask excludes zero padding. semantic_mask additionally excludes camera-alignment context.\n'
        'timestamps_s are relative to the observed action_start. Semantic intent/scope and demonstrated durations remain supervision, not policy observations.\n'
        'Action/source rotation orders are labelled separately. Current M6/X-VLA trainer and executor require explicit mask/timing/horizon integration before using this package.\n'
        'masked_mean_loss accepts unreduced losses only; masking an already-reduced native loss cannot remove padding contributions.\n')
    files = sorted(p for p in output.iterdir() if p.is_file())
    (output/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files))
    print(json.dumps({k: report[k] for k in ['records', 'semantic_samples', 'records_by_split', 'padded_slots', 'ragged_records_padded']}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--source-snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_package(args.export_dir, args.source_snapshot, args.output)
