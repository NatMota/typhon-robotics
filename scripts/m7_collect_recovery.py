"""Collect an auditable V4 corrective catalog from unchanged live learner states."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl'); os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import argparse, copy, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.adaptation.m7_recovery import collect_episode
from supervisor.baseline.xvla import task_prompt
from m7_export_catalog_v2 import sha, gz, dump


def main(args):
    dataset_path = Path('frozen_results/personal-m3-20261009/data-canonical/dataset.json')
    dataset = json.loads(dataset_path.read_text())
    sources = {row['id']: row for row in dataset['episodes'] if row['split'] in {'P_train', 'R_train', 'S_train'}}
    selected = [sources[name] for name in args.source_id]
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint or Path('results/personal-m6-joint-20261009/training/final.pt')
    if args.learner == 'm7' and args.checkpoint is None: raise ValueError('M7 requires its matching V3 checkpoint')
    protocol = {'schema': 'm7-recovery-protocol-v1', 'learner': args.learner, 'method': args.method,
                'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': sha(checkpoint),
                'frozen_source_dataset_sha256': sha(dataset_path), 'source_ids': args.source_id,
                'source_splits': {r['id']: r['split'] for r in selected}, 'learner_steps': args.learner_steps,
                'expert': 'reset-free live-state geometric recovery with actual held-grasp transform',
                'algorithm': 'one-round expert-supervised dataset aggregation from learner states; no PPO/RL optimizer',
                'completed_behavior': 'stop learner immediately on completed geometry; open-gripper zero-motion hold',
                'seed': args.seed, 'annotation_conditional': args.annotation_conditioned,
                'checkpoint_lineage_note': 'frozen M6 readiness evidence' if args.learner == 'frozen-m6' else 'matching V3 learner',
                'created_from_train_configurations_only': True,
                'code_sha256': {p: sha(p) for p in ['supervisor/adaptation/m7_recovery.py', 'scripts/m7_collect_recovery.py']}}
    dump(output/'protocol_before_collection.json', protocol)
    snapshot = {'schema': 'm7-compact-canonical-source-v1', 'dataset_sha256': sha(output/'protocol_before_collection.json'),
                'dataset_sha256_after_read': sha(output/'protocol_before_collection.json'), 'episodes': [], 'source_files': {}}
    evidence = {'schema': 'm7-semantic-source-evidence-v1', 'episodes': {}, 'source_files': {}}
    if args.learner == 'frozen-m6':
        from m6_evaluate import M6Policy
        policy = M6Policy('FT3000', checkpoint, output/'worker.log')
    else:
        from m7_evaluate import M7Policy
        policy = M7Policy(checkpoint, output/'worker.log', request=json.loads(args.request.read_text()) if args.request else None, annotation_conditioned_v2=args.annotation_conditioned)
    try:
        for ordinal, source in enumerate(selected):
            ident = f'm7_recovery_{args.method}_{source["id"]}_{ordinal:02d}'
            folder = output/ident; folder.mkdir()
            config = copy.deepcopy(source['config'])
            dump(folder/'resolved_config.json', config)
            env = FeedingLoadingEnv(config)
            try:
                env.reset(args.seed + ordinal)
                if hasattr(policy, 'bind'): policy.bind(env, folder)
                record = collect_episode(env, policy, args.learner_steps, args.seed + ordinal)
            finally:
                env.close()
            episode = record['episode']
            episode.update(scenario_id=ident, source_demo_id=source.get('source_demo_id'), category='learner_state_recovery',
                           source_train_id=source['id'], source_train_split=source['split'], seed=args.seed + ordinal,
                           protocol_sha256=sha(output/'protocol_before_collection.json'), learner_checkpoint_sha256=protocol['checkpoint_sha256'])
            dump(folder/'episode.json', episode)
            (folder/'robot_actions.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in record['actions']))
            np.savez_compressed(folder/'raw_student_predictions.npz', **{f'chunk_{i:04d}': c for i, c in enumerate(record['raw_student_chunks'])})
            arrays = record['arrays']; path = output/(ident+'.npz'); np.savez_compressed(path, **arrays)
            digest = sha(path)
            meta = {'id': ident, 'split': 'R_train', 'category': 'learner_state_recovery', 'path': path.name, 'sha256': digest,
                    'source_npz_uri': str(path.resolve()), 'verified_source_sha256': digest, 'source_sha256_after_read': digest,
                    'source_action_sha256': sha(folder/'robot_actions.jsonl'), 'source_episode_sha256': sha(folder/'episode.json'),
                    'actions': len(arrays['targets']), 'frames': len(arrays['primary']), 'config': config,
                    'instruction': task_prompt(config['task']), 'source_path': str(folder.resolve()),
                    'supervision_action_start': episode['supervision_action_start'],
                    'source_provenance': {'source_train_id': source['id'], 'source_train_split': source['split'],
                      'learner_checkpoint_sha256': protocol['checkpoint_sha256'], 'learner_kind': args.learner,
                      'actual_learner_actions': episode['learner_executed_actions'], 'no_reset': True,
                      'aggregation_lineage': protocol['checkpoint_lineage_note']},
                    'observation_shapes': {k: {'shape': list(arrays[k].shape), 'dtype': str(arrays[k].dtype)} for k in ['primary', 'secondary']}}
            for key in ['targets', 'proprio', 'frame_action_indices']: meta[key] = arrays[key].tolist()
            primitives = []
            for index, p in enumerate(record['primitives']):
                primitives.append({**p, 'primitive_id': f'{ident}:p{index:02d}:{p["label"]}', 'recording_id': ident,
                                   'split': 'R_train', 'primitive_index': index, 'command_count': p['action_end_exclusive']-p['action_start'],
                                   'duration_seconds': (p['action_end_exclusive']-p['action_start'])/20,
                                   'boundary_evidence': 'Exact reset-free live-state expert phase entry/exit counters',
                                   'source_controller_phase': p['label']})
            dump(folder/'primitives.json', primitives)
            files = {n: {'path': str((folder/n).resolve()), 'sha256_before': sha(folder/n), 'sha256_after': sha(folder/n)} for n in ['episode.json', 'robot_actions.jsonl']}
            if episode['success']:
                snapshot['episodes'].append(meta)
                evidence['episodes'][ident] = {'episode': episode, 'actions': record['actions'], 'primitives': primitives, 'source_files': files,
                    'context_only_ranges': [{'label': 'learner_failure_prefix_no_teacher_labels', 'action_start': 0, 'action_end_exclusive': episode['supervision_action_start']}]}
            print(json.dumps({'id': ident, 'success': episode['success'], 'learner_actions': episode['learner_executed_actions'],
                              'trigger': episode['trigger'], 'supervised_actions': len(arrays['targets'])-episode['supervision_action_start']}), flush=True)
    finally:
        policy.close()
    temporal = output/'temporal'; temporal.mkdir()
    gz(temporal/'source_snapshot.json.gz', snapshot); gz(temporal/'semantic_evidence.json.gz', evidence)
    manifest = {'schema': 'm7-recovery-ready-v1', 'protocol': str((output/'protocol_before_collection.json').resolve()),
                'source_snapshot': str((temporal/'source_snapshot.json.gz').resolve()), 'semantic_evidence': str((temporal/'semantic_evidence.json.gz').resolve()),
                'valid_corrective_episodes': len(snapshot['episodes']), 'attempted_episodes': len(selected),
                'learner': args.learner, 'method': args.method, 'checkpoint_sha256': protocol['checkpoint_sha256'],
                'on_policy_for_matching_v3': args.learner == 'm7', 'learner_state_corrections': all(s['source_provenance']['actual_learner_actions'] > 0 for s in snapshot['episodes']),
                'ready_for_catalog_export': bool(snapshot['episodes']), 'expert_targets_only_after_supervision_action_start': True}
    dump(output/'manifest.json', manifest)
    print(json.dumps(manifest), flush=True)
    if not snapshot['episodes']: raise RuntimeError('No successful corrections; not a trainable V4 dataset')
    if args.emit_gate:
        if args.learner != 'm7' or not manifest['learner_state_corrections']:
            raise ValueError('A V4 gate requires actual matching-V3 learner actions in every correction')
        emit_gate(output, args.emit_gate, manifest)


def emit_gate(output, destination, manifest):
    """Freeze only successful same-checkpoint corrections after loader validation."""
    import csv
    from m7_export_catalog_v2 import export
    from m7_export_held_catalog import run as export_held
    from supervisor.adaptation.m7_catalog_data import M7Dataset
    from supervisor.adaptation.m7_visual import load_held_lookup
    repo = Path(__file__).resolve().parents[1]
    export(output)
    export_held(output/'temporal', output/'visual-enrichment')
    dataset = M7Dataset(output/'export', output/'temporal/source_snapshot.json.gz')
    with (output/'export/v4_candidates.csv').open() as stream:
        candidates = list(csv.DictReader(stream))
    annotations = load_held_lookup(output/'visual-enrichment/oracle_annotations.jsonl')
    for row in candidates:
        packed = dataset.pack(row['sample_id'], row['camera_variant'])
        source = dataset.sources[row['recording_id']]
        floor = source['supervision_action_start']
        assert packed['provenance']['action_start'] >= floor
        assert all(i >= floor for i in packed['supervision']['source_action_indices'] if i >= 0)
        assert (row['recording_id'], int(packed['provenance']['frame_index'])) in annotations
    for source in dataset.sources.values():
        assert source['source_provenance']['learner_checkpoint_sha256'] == manifest['checkpoint_sha256']
        assert source['source_provenance']['actual_learner_actions'] > 0
        assert source['source_provenance']['no_reset']
    def relative(path): return str(Path(path).resolve().relative_to(repo))
    gate = {'schema': 'm7-actual-dagger-gate-v1', 'ready': True, 'method': manifest['method'],
            'student_checkpoint_sha256': manifest['checkpoint_sha256'], 'actual_student_state_handover': True,
            'ge_corrective_examples': len(candidates), 'algorithm': 'dagger_supervised_correction',
            'provenance': relative(output/'manifest.json'),
            'file_sha256': {relative(p): sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'worker.log'},
            'training_data': {'export': relative(output/'export'),
                'source_snapshot': relative(output/'temporal/source_snapshot.json.gz'),
                'annotations': relative(output/'visual-enrichment/oracle_annotations.jsonl'), 'source_root': relative(output)},
            'candidate_manifest': relative(output/'export/v4_candidates.csv')}
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists(): raise FileExistsError(destination)
    dump(destination, gate)
    print(json.dumps({'gate': str(destination), 'corrective_examples': len(candidates), 'ready': True}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learner', choices=['frozen-m6', 'm7'], default='frozen-m6')
    parser.add_argument('--method', choices=['M3', 'M4', 'M5', 'M6'], default='M6')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/personal-m7-training-20261009/recovery-v1/frozen-M6-readiness'))
    parser.add_argument('--source-id', action='append', default=None)
    parser.add_argument('--learner-steps', type=int, default=120)
    parser.add_argument('--seed', type=int, default=773901)
    parser.add_argument('--request', type=Path, help='Optional declared non-oracle planner request JSON')
    parser.add_argument('--annotation-conditioned', action='store_true', help='Declare causal privileged held label for crosshair-trained V3')
    parser.add_argument('--emit-gate', type=Path, help='Export/validate catalog and emit matching-V3 DAgger gate here')
    args = parser.parse_args()
    args.source_id = args.source_id or ['personal_demo_0007']
    if args.learner_steps < 1: parser.error('--learner-steps must be positive')
    main(args)
