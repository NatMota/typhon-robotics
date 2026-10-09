"""Portable all-category M7 catalog loader. Actual video integration is a trainer responsibility."""
from .m7_data import M7Dataset as BaseDataset, CAMERAS, CONTROL_HZ, HORIZON, INSTRUCTIONS as BASE_INSTRUCTIONS, reorder
INSTRUCTIONS = {**BASE_INSTRUCTIONS, 'continue_task': 'Continue the supplied feeding task for the requested native control horizon.',
                'maintain_completed_state': 'Keep the completed block placement undisturbed and hold the robot still with the gripper open.'}
class M7Dataset(BaseDataset):
    def __init__(self, export_dir, source_snapshot, split=None):
        super().__init__(export_dir, source_snapshot, split=None)
        allowed={'P_train','R_train','S_train','C_train'}
        if any(r['split'] not in allowed for r in self.rows):
            raise ValueError('Training catalog contains a non-training split')
        if split is not None:
            if split not in allowed: raise ValueError('Unknown train split')
            self.rows=[r for r in self.rows if r['split']==split]
            self.by_id={r['sample_id']:r for r in self.rows}

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
        packed = {
            'observation': {'proprio': proprio, 'task_instruction': source['instruction'], 'camera_ref': camera_ref},
            'supervision': {
                'action': action, 'action_source': action_source, 'valid_mask': valid,
                'semantic_mask': [i >= semantic_start for i in indices]+[False]*(HORIZON-count),
                'timestamps_s': [(i-start)/CONTROL_HZ for i in indices]+[0.0]*(HORIZON-count),
                'source_action_indices': indices+[-1]*(HORIZON-count), 'true_length': count,
                'duration_seconds': (end-semantic_start)/CONTROL_HZ,
                'aligned_duration_seconds': (end-start)/CONTROL_HZ,
                'scope': row['representation'], 'primitive_count': int(row['primitive_count']),
                'semantic_instruction': row.get('semantic_instruction') or ' '.join(INSTRUCTIONS[label] for label in row['component_labels'].split('|')),
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


        histories = {}
        for mode in ('dense', 'sparse'):
            frames = [int(v) for v in row[mode+'_history_frame_indices'].split()]
            if frames != sorted(set(frames)) or frames[-1] != frame:
                raise ValueError('Causal history must be ordered and end at the current frame')
            action_indices = [source['frame_action_indices'][i] for i in frames]
            if any(i > start for i in action_indices):
                raise ValueError('Future camera frame in model history')
            histories[mode] = {'frame_indices': frames, 'action_indices': action_indices,
                               'relative_timestamps_s': [(i-start)/CONTROL_HZ for i in action_indices],
                               'source_npz_sha256': source['sha256'],
                               'observation_timing': 'Every frame captured before its own source action; current frame included'}
        packed['observation']['history'] = histories
        packed['provenance']['category'] = row['category']
        packed['provenance']['target_kind'] = row['target_kind']
        packed['integration']['direct_20hz_execution_allowed'] = row['target_kind'] == 'native_contiguous'
        return packed


    def history_images(self, packed, source_root, mode, frame_budget=None):
        """Return time-ordered coherent native camera contexts and exact causal timestamps.

        An explicit budget picks the most recent dense frames or a spaced sparse subset.
        The full candidate history remains in pack metadata; truncation is disclosed here.
        """
        import copy
        if mode not in ('dense','sparse'):
            raise ValueError('Declare dense or sparse history')
        history=packed['observation']['history'][mode]
        frames=list(history['frame_indices'])
        if frame_budget is not None:
            if not isinstance(frame_budget,int) or frame_budget<1: raise ValueError('Positive frame budget required')
            if len(frames)>frame_budget:
                if mode=='dense': frames=frames[-frame_budget:]
                elif frame_budget==1: frames=frames[-1:]
                else: frames=[frames[round(i*(len(frames)-1)/(frame_budget-1))] for i in range(frame_budget)]
        source=self.sources[packed['provenance']['recording_id']]
        contexts=[]
        for frame in frames:
            record=copy.deepcopy(packed)
            ref=record['observation']['camera_ref']
            ref['frame_index']=frame;ref['action_start']=source['frame_action_indices'][frame]
            contexts.append(self.images(record,source_root))
        return {'contexts':contexts,'frame_indices':frames,
                'relative_timestamps_s':[(source['frame_action_indices'][i]-packed['provenance']['action_start'])/CONTROL_HZ for i in frames],
                'mode':mode,'candidate_frames':len(history['frame_indices']),
                'explicit_frame_budget':frame_budget,'truncated':len(frames)<len(history['frame_indices'])}
