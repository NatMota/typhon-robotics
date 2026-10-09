"""Create Run3 <=4s temporal abstraction from unchanged paired simulator RGB/expert targets."""
import argparse, csv, gzip, hashlib, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from m7_export_catalog_v2 import semantic_anchors
from m7_prepare_sampling import balanced, summarize
from supervisor.adaptation.m7_catalog_data import M7Dataset, INSTRUCTIONS
from supervisor.adaptation.m7_timing import reconstruct_20hz


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def csvwrite(p, rows):
    with Path(p).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def gzwrite(p, obj):
    with Path(p).open('xb') as f:
        with gzip.GzipFile(fileobj=f, mode='wb', filename='', mtime=0) as z:
            z.write(json.dumps(obj, separators=(',', ':'), allow_nan=False).encode())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--catalog', type=Path, default=ROOT/'results/personal-m7-data-prep-20261009/training-catalog-v2-complete')
    ap.add_argument('--output', type=Path, default=ROOT/'results/personal-m7-data-prep-20261009/run3-temporal-v3_1')
    ap.add_argument('--source-root', type=Path)
    a = ap.parse_args(); catalog = a.catalog.resolve(); out = a.output.resolve()
    if out.exists():
        raise SystemExit('Create-only output exists; choose a new output directory.')
    source_root = (a.source_root or catalog/'rgb').resolve()
    snapshot_path = catalog/'temporal/source_snapshot.json.gz'
    protected = [p for p in catalog.rglob('*') if p.is_file() and 'rgb' not in p.relative_to(catalog).parts]
    protected += [ROOT/'scripts/m7_train.py', ROOT/'supervisor/adaptation/m7_catalog_data.py', ROOT/'supervisor/adaptation/m7_timing.py']
    before = {str(p): sha(p) for p in protected}
    snapshot = json.loads(gzip.decompress(snapshot_path.read_bytes()))
    sources = {s['id']: s for s in snapshot['episodes']}
    parents = [r for r in csv.DictReader((catalog/'export/index.csv').open()) if r['target_kind']=='semantic']
    primitives = json.loads((catalog/'export/primitives.json').read_text())
    parts = {p['primitive_id']: p for p in primitives}
    canonical_path = ROOT/'frozen_results/personal-m3-20261009/data-canonical/dataset.json'
    canonical = json.loads(canonical_path.read_text())
    heldout = [e for e in canonical['episodes'] if e['split'].endswith('_validation')]
    assert not set(sources) & {e['id'] for e in heldout}
    assert all(s['split'] in {'P_train','R_train','S_train','C_train'} for s in sources.values())
    for field in ['sha256','source_action_sha256','source_episode_sha256']:
        assert not {s[field] for s in sources.values()} & {e[field] for e in heldout}
    source_audit = []
    for s in sources.values():
        path = source_root/s['path']; digest = sha(path); assert digest == s['sha256']
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z['targets'], np.asarray(s['targets']))
            assert np.array_equal(z['proprio'], np.asarray(s['proprio']))
            assert np.array_equal(z['frame_action_indices'], s['frame_action_indices'])
            assert z['primary'].shape == z['secondary'].shape == (s['frames'],512,512,3)
            assert z['primary'].dtype == z['secondary'].dtype == np.uint8
        assert s['frame_action_indices'] == list(range(0,s['actions'],4))
        source_audit.append({'recording_id':s['id'],'path':str(path),'sha256':digest,'split':s['split'],'actions':s['actions'],'rgb_frames_per_camera':s['frames']})
    out.mkdir(parents=True); export = out/'export'; export.mkdir()
    rows=[]; arrays=[]; pair_records=[]; parent_coverage=[]
    for p in parents:
        s=sources[p['recording_id']]; first=int(p['action_start']); last=int(p['action_end_exclusive'])
        selected=[parts[x] for x in p['component_primitive_ids'].split('|')]
        windows=list(range(first,last,80)); covered=set()
        for ordinal,start in enumerate(windows):
            end=min(start+80,last); begin=max(int(p['semantic_start']),start)
            active=[dict(x,action_start=max(x['action_start'],begin),action_end_exclusive=min(x['action_end_exclusive'],end)) for x in selected if x['action_start']<end and x['action_end_exclusive']>begin]
            assert active and start<=begin<end and begin-start<4
            anchor_parts=([{'action_start':start,'action_end_exclusive':begin}] if begin>start else [])+active
            indices,mandatory=semantic_anchors(start,end,anchor_parts,s['targets'])
            frame=s['frame_action_indices'].index(start)
            dense=list(range(max(0,frame-20),frame+1))
            sparse=list(reversed(list(range(frame,max(-1,frame-21),-5))))
            row=dict(p, sample_id=p['sample_id']+f':4s:w{ordinal:02d}', targets_row=len(rows), frame_index=frame,
                action_start=start,semantic_start=begin,action_end_exclusive=end,context_prefix_commands=begin-start,
                true_anchor_count=len(indices),anchor_action_indices=' '.join(map(str,indices)),
                parent_sample_id=p['sample_id'],parent_action_start=first,parent_semantic_start=int(p['semantic_start']),
                parent_action_end_exclusive=last,window_index=ordinal,window_count=len(windows),
                parent_aligned_duration_seconds=(last-first)/20,window_aligned_duration_seconds=(end-start)/20,
                semantic_instruction='Continue the requested sequence: '+' '.join(INSTRUCTIONS[x] for x in p['component_labels'].split('|'))+' Execute only the requested next portion of this sequence.',
                active_component_primitive_ids='|'.join(x['primitive_id'] for x in active),
                active_component_labels='|'.join(x['label'] for x in active),
                partial_parent_chunk=len(windows)>1,mandatory_anchor_action_indices=' '.join(map(str,mandatory)),
                temporal_representation='event_aware_source_tick_anchors_within_4s',input_history_seconds=4)
            for mode,frames in [('dense',dense),('sparse',sparse)]:
                ii=[s['frame_action_indices'][f] for f in frames]
                row[mode+'_history_frame_indices']=' '.join(map(str,frames))
                row[mode+'_history_action_indices']=' '.join(map(str,ii))
                row[mode+'_history_relative_timestamps_s']=' '.join(f'{(i-start)/20:.2f}' for i in ii)
            rows.append(row); arrays.append([s['targets'][i] for i in indices]); covered.update(range(start,end))
            future_frames=[max(j for j,idx in enumerate(s['frame_action_indices']) if idx<=i) for i in indices]
            pair_records.append({'sample_id':row['sample_id'],'recording_id':s['id'],'source_npz_sha256':s['sha256'],
                'future_anchor_action_indices':indices,'future_anchor_timestamps_s':[i/20 for i in indices],
                'paired_preceding_rgb_frame_indices':future_frames,
                'paired_preceding_rgb_action_indices':[s['frame_action_indices'][j] for j in future_frames],
                'rgb_to_anchor_lag_s':[(i-s['frame_action_indices'][j])/20 for i,j in zip(indices,future_frames)],
                'role':'Future RGB/action pairing is supervision/QA metadata only, never observation input.',
                'camera_arrays':['primary','secondary'],'no_frame_interpolation':True})
        assert covered==set(range(first,last))
        parent_coverage.append({'parent_sample_id':p['sample_id'],'windows':len(windows),'start':first,'end_exclusive':last,'all_source_ticks_covered':True})
    assert len({r['sample_id'] for r in rows})==len(rows)
    csvwrite(export/'index.csv',rows); gzwrite(export/'targets.json.gz',arrays)
    dump(export/'primitives.json',primitives)
    candidates=[{'sample_id':r['sample_id'],'recording_id':r['recording_id'],'split':r['split'],'camera_variant':view,
        'representation':r['representation'],'history_mode':'sparse','green_held_marker':True,'data_variant':'v3',
        'example_id':r['sample_id']+':'+view,'category':r['category']} for r in rows for view in ['overhead_main','oblique_main']]
    csvwrite(export/'v3_candidates.csv',candidates)
    report={'schema':'m7-run3-temporal-4s-v3','source_files_before':{str(snapshot_path):sha(snapshot_path)},
        'parent_semantic_chunks':len(parents),'windowed_samples':len(rows),'camera_examples':len(candidates),'source_episodes':len(sources),
        'stock_trainer_compatible':False,'loader_module':'m7_catalog_data','train_only':True,
        'human_alignment':'Simulator camera frames pair with generated geometric-expert actions. Personal phone videos supply source scenario provenance, not framewise action supervision.',
        'timing':'Unchanged20Hz control ticks; <=4s/80tick windows; <=30 exact source anchors. Both sides of context/primitive boundaries and binary gripper transitions retained. No time compression or smoothing of labels.',
        'video':'Synchronized5Hz source captures. Model observes causal1Hz history over at most4s ending at current capture. Future anchor-to-RGB pairing is QA/supervision only.',
        'long_intent':'Parent requested sequence and all its source time ranges retained across contiguous4s windows; full17.8s examples are not squeezed into4s.',
        'camera_variants':'Nativeoverhead/oblique exchange main/auxiliary roles with unchanged world targets and pixels.',
        'green_marker':'Simulator current/past grasp-contact annotation; privileged channel; requires runtime sensor or labelled oracle experiment.',
        'proprio_contract':'First sample uses measured end-effector pose; later pose slots use the preceding commanded target. Not a claim of continuous measured robot pose.',
        'primitive_limits':'Approach+align and transport+reorient remain composite expert phases; no invented rotation-only phase or human pause action.',
        'short_windows':'Fewer than30 native ticks produce masked shorter targets; zero padding is invalid, not an action.',
        'conditioning':'Training request intent/horizon comes from offline labels. Runtime must use a user/planner/predicted request; no future evaluator labels.'}
    dump(export/'report.json',report)
    (export/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(export.iterdir()) if p.is_file()))
    ds=M7Dataset(export,snapshot_path)
    approximate=[]; padded=0; anchors_total=0; min_times=[]; view_checks=0
    for row,pair in zip(rows,pair_records):
        s=sources[row['recording_id']]
        for view in ['overhead_main','oblique_main']:
            packed=ds.pack(row['sample_id'],view); su=packed['supervision']; history=packed['observation']['history']['sparse']
            assert all(-4<=t<=0 for t in history['relative_timestamps_s']) and history['relative_timestamps_s'][-1]==0
            assert history['frame_indices'][-1]==int(row['frame_index'])
            assert max(su['timestamps_s'])<su['aligned_duration_seconds']<=4
            assert np.array_equal(np.asarray(su['action_source'])[su['valid_mask']],np.asarray(s['targets'])[pair['future_anchor_action_indices']])
            assert all(0<=lag<=.15 for lag in pair['rgb_to_anchor_lag_s'])
            view_checks+=1
        request={'source':'training_demonstration_supervision','scope':su['scope'],
            'instruction':su['semantic_instruction'],'anchor_times_s':[t for t,v in zip(su['timestamps_s'],su['valid_mask']) if v],
            'duration_s':su['duration_seconds'],'aligned_duration_s':su['aligned_duration_seconds'],
            'context_prefix_s':packed['provenance']['context_prefix_commands']/20}
        payload={'request':request,'waypoints':[{'time_s':t,'target_generic_interleaved':x} for t,x,v in zip(su['timestamps_s'],su['action'],su['valid_mask']) if v]}
        reconstructed=reconstruct_20hz(payload,oracle_replay=True)['targets']
        native=np.asarray(s['targets'][int(row['action_start']):int(row['action_end_exclusive'])])
        assert len(native)==len(reconstructed)
        anchors=pair['future_anchor_action_indices']; local=np.asarray(anchors)-int(row['action_start'])
        assert np.array_equal(native[local],reconstructed[local])
        assert np.array_equal(native[:,[9,19]],reconstructed[:,[9,19]])
        try: reconstruct_20hz(payload)
        except ValueError: pass
        else: raise AssertionError('Future training timing accepted as inference input')
        approximate.append({'sample_id':row['sample_id'],'position_max_m':float(np.linalg.norm(native[:,:3]-reconstructed[:,:3],axis=1).max()),'source_target_rot6_max_abs':float(np.max(np.abs(native[:,3:9]-reconstructed[:,3:9]))),'gripper_timing_exact':True})
        padded+=30-sum(su['valid_mask']); anchors_total+=sum(su['valid_mask']); min_times.append(min(history['relative_timestamps_s']))
    plan,quota=balanced(candidates,17); replay,_=balanced(candidates,17); alternative,quota2=balanced(candidates,29)
    assert plan==replay and plan!=alternative and quota==quota2
    assert {r['example_id'] for r in plan}=={r['example_id'] for r in candidates}
    csvwrite(out/'train_epoch_seed17.csv',plan); csvwrite(out/'train_epoch_seed29.csv',alternative)
    csvwrite(out/'parent_coverage.csv',parent_coverage); csvwrite(out/'reconstruction_diagnostics.csv',approximate)
    (out/'trajectory_anchor_pairs.jsonl').write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in pair_records))
    dump(out/'source_audit.json',source_audit)
    assert all(sha(p)==value for p,value in before.items()), 'A protected original input changed'
    validation={**report,'status':'PASS','training_started':False,'image_model_forward_tested':False,
        'rows_by_scope':dict(Counter(r['representation'] for r in rows)), 'parents_by_scope':dict(Counter(r['representation'] for r in parents)),
        'rows_by_split':dict(Counter(r['split'] for r in rows)), 'rows_by_category':dict(Counter(r['category'] for r in rows)),
        'full4s_windows':sum(float(r['window_aligned_duration_seconds'])==4 for r in rows),
        'windows_under4s':sum(float(r['window_aligned_duration_seconds'])<4 for r in rows),
        'valid_action_anchors':anchors_total,'invalid_padding_slots':padded,'validated_view_packs':view_checks,
        'source_counts':dict(Counter(s['split'] for s in sources.values())), 'balanced_sampling':summarize(plan),
        'leakage':{'heldout_episode_ids_excluded':[e['id'] for e in heldout],'npz_action_episode_hash_overlap':False,'session_independence':'Not established; some original human recordings contain both training and held-out takes.'},
        'checks':{'all_parent_chunks_covered':True,'no_missing_source_time_ranges':True,'all_source_hashes_and_paired_arrays_verified':True,'exact_source_action_anchor_values':True,'exact_gripper_transition_reconstruction':True,'history_past_only':True,'history_within4s':True,'same_timeline_views_and_action_labels':True,'no_human_to_robot_time_stretch':True,'source_and_v1_v2_unchanged':True,'seed_reproducibility_and_complete_coverage':True,'training_truth_rejected_at_inference':True},
        'approximation':{'max_cartesian_target_reconstruction_error_m':max(x['position_max_m'] for x in approximate),'mean_window_max_cartesian_error_m':float(np.mean([x['position_max_m'] for x in approximate])),'meaning':'Interpolation diagnostic against original commanded targets, not new simulator execution or measured success.'},
        'ready_gate':'DATA_PASS. Before Run3 training: native image/history batch+finite-loss smoke on target runtime; use freshvariant3 checkpoint/config; explicit non-oracle runtime intent/horizon and held-marker handling for evaluation.',
        'protected_sha256':before,'script_sha256':sha(Path(__file__))}
    dump(out/'validation_summary.json',validation)
    config={'variant':3,'loader_module':'m7_catalog_data','export_dir':str(export),'source_snapshot':str(snapshot_path),'source_root':str(source_root),
        'candidate_manifest':str(export/'v3_candidates.csv'),'annotations':str(catalog/'visual-enrichment/oracle_annotations.jsonl'),
        'history_mode':'sparse','history_seconds':4,'condition':'crosshair','max_future_seconds':4,'max_action_anchors':30,
        'source_control_hz':20,'source_camera_hz':5,'sparse_input_history_hz':1,'task_splits':['P_train','R_train','S_train','C_train'],
        'start_training':False,'data_claim':report['human_alignment'],'required_prelaunch_gate':validation['ready_gate']}
    dump(out/'training_data_config.json',config)
    (out/'REPRODUCE.txt').write_text('.venv-sim/bin/python scripts/m7_prepare_run3_temporal.py --output results/personal-m7-data-prep-20261009/run3-temporal-v3-reproduction\n')
    lines=['# Run3 temporal abstraction data','',f"**Data validation PASS; no training started.** {len(parents)} original semantic chunks become {len(rows)} windows / {len(candidates)} camera arrangements from {len(sources)} training episodes.",'',
        '| Scope | Original chunks | Four-second-or-shorter windows |','|---|---:|---:|']
    for scope in ['single','pair','triple','intent']:
        lines.append(f"| {scope} | {validation['parents_by_scope'].get(scope,0)} | {validation['rows_by_scope'].get(scope,0)} |")
    lines += ['', 'All parent chunks, categories and source timeline intervals remain represented. Windows tile the original commands without changing their speed. Up to30 event-aware anchors retain exact source values; mandatory anchors preserve primitive endpoints and both sides of gripper transitions. Short tails are masked instead of invented.', '',
        'RGB observations are actual synchronized robot-camera captures, causally subsampled to1Hz over the previous4s. They stay on the same20Hz action clock; overhead/oblique views use the same frame index. Future target anchors point to the preceding available5Hz RGB capture with0–0.15s lag for QA only. These future images are never fed as policy history.', '',
        'Full intent remains a parent sequence linked across adjacent windows. Pair/triple examples retain their original requested intent even if a window covers only part of it. Per-window active primitive IDs are explicit. Completed holds have single/intent examples; no fictional pair/triple primitives are created.', '',
        'The green held marker uses declared privileged simulator contact annotations. The pipeline does not claim raw human frame-to-robot-action supervision, human pause imitation, independently demonstrated rotation-only primitives, or reduced physical motion noise. Temporal decimation removes intermediate labels; smoothing effects are unproven.', '',
        f"A balanced shuffled epoch has {len(plan)} draws and covers every camera example. Seeds17/29 change ordering with identical group counts. Six held-out episodes and their hashes are excluded; original recording-session independence is not established.", '',
        f"Timestamp-aware interpolation has worst commanded Cartesian deviation {validation['approximation']['max_cartesian_target_reconstruction_error_m']:.6f}m from the original20Hz labels; all gripper transitions are exact. This is a numerical check, not a rollout success result.", '',
        'Proprioception retains the legacy interface: first sample measured, later pose slots are previous commands, not new measurements.\n\nRun3 requires a fresh config/checkpoint and a target-runtime image/history+finite-loss smoke before training. This export works with the existing timed/masked catalog loader; stock X-VLA loss alone is insufficient. Runtime request timing must come from user/planner/policy, not held-out demonstration truth.', '',
        'Reproduce with `REPRODUCE.txt`; inspect `validation_summary.json`, `training_data_config.json`, `parent_coverage.csv`, `trajectory_anchor_pairs.jsonl`, and the export manifest. V1/V2, source snapshots, raw data and frozen experiments were not modified.']
    (out/'Run3_report.md').write_text('\n'.join(lines)+'\n')
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'SHA256SUMS'))
    print(json.dumps({k:validation[k] for k in ['status','parent_semantic_chunks','windowed_samples','camera_examples','rows_by_scope','rows_by_split','valid_action_anchors','invalid_padding_slots','approximation','ready_gate']},indent=2))

if __name__=='__main__': main()
