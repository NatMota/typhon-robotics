"""Create-only matching-student correction aggregation; never collects or trains."""
import argparse, copy, csv, gzip, hashlib, json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from m7_export_catalog_v2 import semantic_anchors
from m7_prepare_sampling import balanced,summarize
from m7_export_held_catalog import run as export_held
from supervisor.adaptation.m7_catalog_data import M7Dataset,INSTRUCTIONS
from supervisor.adaptation.m7_timing import reconstruct_20hz

TRAIN={'P_train','R_train','S_train','C_train'}
METHODS={'M3':'m3_prompt','M4':'m4_recovery','M5':'m5_lora','M6':'m6_joint'}

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(Path(p).read_text())
def readgz(p):return json.loads(gzip.decompress(Path(p).read_bytes()))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def gz(p,x):
    with Path(p).open('xb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0,filename='') as z:z.write(json.dumps(x,separators=(',',':'),allow_nan=False).encode())
def writecsv(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def history(indices,start):
    dense=[j for j,t in enumerate(indices) if start-80<=t<=start]
    past=[j for j in dense if indices[j]<start]
    sparse=sorted({max([j for j in past if indices[j]<=start-20*k],default=-1) for k in range(1,5)}-{-1})+[indices.index(start)]
    return dense,sparse

def correction_rows(source,parts):
    """Window real source times; handle an extra off-grid teacher-handover frame."""
    floor=source['supervision_action_start'];frames=source['frame_action_indices'];rows=[];arrays=[];coverage=[]
    assert floor in frames and floor>0 and frames==sorted(set(frames))
    assert all(p['action_start']>=floor for p in parts)
    for scope,size in [('single',1),('pair',2),('triple',3),('intent',len(parts))]:
        for first in range(len(parts)-size+1):
            selected=parts[first:first+size];begin=selected[0]['action_start'];end=selected[-1]['action_end_exclusive']
            assert all(x['action_end_exclusive']==y['action_start'] for x,y in zip(selected,selected[1:]))
            parent=f"{source['id']}:{scope}:p{first:02d}-p{first+size-1:02d}";cursor=begin;ordinal=0;covered=set()
            while cursor<end:
                start=max(t for t in frames if floor<=t<=cursor);stop=min(start+80,end);assert start<=cursor<stop and cursor-start<=3
                active=[dict(p,action_start=max(p['action_start'],cursor),action_end_exclusive=min(p['action_end_exclusive'],stop)) for p in selected if p['action_start']<stop and p['action_end_exclusive']>cursor]
                context=[{'action_start':start,'action_end_exclusive':cursor}] if cursor>start else []
                anchors,mandatory=semantic_anchors(start,stop,context+active,source['targets'])
                dense,sparse=history(frames,start)
                row={'sample_id':parent+f':4s:w{ordinal:02d}','recording_id':source['id'],'split':'R_train','category':source['category'],
                    'representation':scope,'primitive_count':len(selected),'component_primitive_ids':'|'.join(p['primitive_id'] for p in selected),
                    'component_labels':'|'.join(p['label'] for p in selected),'frame_index':frames.index(start),'action_start':start,
                    'semantic_start':cursor,'action_end_exclusive':stop,'context_prefix_commands':cursor-start,'true_anchor_count':len(anchors),
                    'source_npz_sha256':source['sha256'],'source_action_sha256':source['source_action_sha256'],'source_episode_sha256':source['source_episode_sha256'],
                    'anchor_action_indices':' '.join(map(str,anchors)),'target_kind':'semantic','parent_sample_id':parent,
                    'parent_semantic_start':begin,'parent_action_end_exclusive':end,'window_index':ordinal,
                    'window_aligned_duration_seconds':(stop-start)/20,'semantic_instruction':'Continue the requested sequence: '+' '.join(INSTRUCTIONS[p['label']] for p in selected)+' Execute only the requested next portion of this sequence.',
                    'active_component_primitive_ids':'|'.join(p['primitive_id'] for p in active),'mandatory_anchor_action_indices':' '.join(map(str,mandatory)),
                    'source_cohort':('matching_student_completed_hold' if source['category']=='learner_completed_hold' else 'matching_student_correction'),'teacher_supervision_floor':floor}
                for mode,inds in [('dense',dense),('sparse',sparse)]:
                    row[mode+'_history_frame_indices']=' '.join(map(str,inds));row[mode+'_history_action_indices']=' '.join(str(frames[j]) for j in inds)
                    row[mode+'_history_relative_timestamps_s']=' '.join(f'{(frames[j]-start)/20:.2f}' for j in inds)
                rows.append(row);arrays.append([source['targets'][i] for i in anchors]);covered.update(range(cursor,stop));cursor=stop;ordinal+=1
            assert covered==set(range(begin,end))
            coverage.append({'parent_sample_id':parent,'windows':ordinal,'semantic_start':begin,'end_exclusive':end,'no_lost_teacher_ticks':True})
    return rows,arrays,coverage

def validate_identity(manifest,protocol,method,checkpoint_sha,train_ids):
    assert manifest['learner']==protocol['learner']=='m7','Frozen-M6 diagnostic corrections are forbidden'
    assert manifest['method']==protocol['method']==method
    assert manifest['checkpoint_sha256']==protocol['checkpoint_sha256']==checkpoint_sha,'Wrong student checkpoint'
    assert manifest['on_policy_for_matching_v3'] and manifest['learner_state_corrections']
    assert set(protocol['source_ids'])<=train_ids,'Held-out or unknown collection scenario'
    assert all(s in TRAIN for s in protocol['source_splits'].values()),'Non-training collection scenario'

def main(a):
    base=a.base_data.resolve();catalog=a.base_catalog.resolve();recovery=a.recovery.resolve();out=a.output.resolve();checkpoint=a.student_checkpoint.resolve();base_rgb=(a.base_rgb_root or catalog/'rgb').resolve()
    assert not out.exists(),'Create-only output already exists'
    base_snap=catalog/'temporal/source_snapshot.json.gz';base_evidence=catalog/'temporal/semantic_evidence.json.gz'
    protected=[base/'export/index.csv',base/'export/targets.json.gz',base/'export/v3_candidates.csv',base/'export/SHA256SUMS',base_snap,base_evidence]
    before={str(p):sha(p) for p in protected};checkpoint_sha=sha(checkpoint)
    cfg=load(checkpoint.parent/'config.json');training=load(checkpoint.parent/'training_report.json')
    assert training['status']=='PASS' and training['stop_reason']=='configured_updates_complete'
    assert training['checkpoint_sha256']==checkpoint_sha and cfg['training_mode']==METHODS[a.method]
    assert cfg['history_mode']=='sparse' and cfg['history_seconds']==4 and cfg['condition']=='crosshair'
    assert cfg['catalog_input_hashes']['candidate_manifest']==sha(base/'export/v3_candidates.csv'),'Student did not train on this Run3 export'
    assert cfg['catalog_input_hashes']['source_snapshot']==sha(base_snap)
    bs=readgz(base_snap);be=readgz(base_evidence);original={s['id']:s for s in bs['episodes']};assert len(original)==18 and Counter(s['split'] for s in original.values())==Counter(P_train=6,R_train=2,S_train=6,C_train=4)
    canonical_path=ROOT/'frozen_results/personal-m3-20261009/data-canonical/dataset.json';canonical=load(canonical_path)
    heldout={e['id'] for e in canonical['episodes'] if e['split'].endswith('_validation')};assert not set(original)&heldout
    manifest=load(recovery/'manifest.json');protocol=load(recovery/'protocol_before_collection.json');receipt=load(a.precollection)
    validate_identity(manifest,protocol,a.method,checkpoint_sha,set(original))
    assert receipt['checkpoint_sha256']==checkpoint_sha and receipt['method']==a.method
    assert set(receipt['source_ids'])==set(protocol['source_ids'])
    assert receipt['source_dataset_sha256']==protocol['frozen_source_dataset_sha256']==sha(canonical_path)
    request=receipt['request'];assert request['source']=='planner' and request['scope']=='intent' and request['duration_s']==request['aligned_duration_s']==4
    assert request['anchor_times_s']==[round(i*79/29)/20 for i in range(30)] and request['context_prefix_s']==0
    assert '--emit-gate' not in receipt['command']
    reqpath=Path(receipt['command'][receipt['command'].index('--request')+1]);reqpath=reqpath if reqpath.is_absolute() else ROOT/reqpath
    assert sha(reqpath)==receipt['request_sha256'] and load(reqpath)==request
    rs=readgz(recovery/'temporal/source_snapshot.json.gz');re=readgz(recovery/'temporal/semantic_evidence.json.gz');accepted=[];attempts=[];actual_corrections=[];completed_holds=[];quarantined=[]
    for folder in sorted(p for p in recovery.iterdir() if p.is_dir() and (p/'episode.json').exists()):
        ep=load(folder/'episode.json');attempts.append({'id':ep['scenario_id'],'success':ep['success'],'learner_actions':ep['learner_executed_actions'],'trigger':ep['trigger'],'included':ep['success']})
    assert len(attempts)==manifest['attempted_episodes'],'Not all attempted episodes have preserved results'
    for s in rs['episodes']:
        ev=re['episodes'][s['id']];ep=ev['episode'];floor=s['supervision_action_start'];actions=ev['actions'];provenance=s['source_provenance']
        assert ep['success'] and ep['evaluation']['success'] and floor==ep['learner_executed_actions']>0
        trigger=ep['trigger'];assert trigger=='completed_hold' or trigger=='failed_grasp_empty_closed_gripper' or trigger.startswith('learner_budget_exhausted:')
        s=copy.deepcopy(s)
        if trigger=='completed_hold':
            assert ep['completed_stop'];s['category']='learner_completed_hold'
        else:
            assert not ep['completed_stop']
        assert ep['no_reset_or_state_relocation_after_learner'] and ep['handover_sim_state_sha256']==ep['first_expert_pre_action_sim_state_sha256']
        assert ep['learner_checkpoint_sha256']==provenance['learner_checkpoint_sha256']==checkpoint_sha
        assert provenance['learner_kind']=='m7' and provenance['source_train_id'] in original and provenance['no_reset']
        assert s['config']==original[provenance['source_train_id']]['config'],'Geometry differs from declared train source'
        assert all(x['controller']=='learner' and not x['teacher_supervised'] for x in actions[:floor])
        assert all(x['controller']=='geometric_expert' and x['teacher_supervised'] for x in actions[floor:])
        failed_primitives=[p for p in ev['primitives'] if p['converged'] is False]
        path=recovery/s['path'];assert sha(path)==s['sha256']
        with np.load(path,allow_pickle=False) as z:
            assert np.array_equal(z['teacher_mask'],np.arange(s['actions'])>=floor)
            assert np.array_equal(z['targets'],np.asarray(s['targets'])) and np.array_equal(z['frame_action_indices'],s['frame_action_indices'])
            assert z['primary'].shape==z['secondary'].shape==(s['frames'],512,512,3)
        assert sha(recovery/s['id']/'episode.json')==s['source_episode_sha256'] and sha(recovery/s['id']/'robot_actions.jsonl')==s['source_action_sha256']
        assert ep['protocol_sha256']==sha(recovery/'protocol_before_collection.json')
        if failed_primitives:
            reason={'id':s['id'],'reason':'Successful final state but expert primitive failed convergence; excluded pending review','failed_primitives':failed_primitives}
            quarantined.append(reason)
            for outcome in attempts:
                if outcome['id']==s['id']:outcome.update(included=False,quarantine_reason=reason['reason'])
            continue
        (completed_holds if trigger=='completed_hold' else actual_corrections).append(s['id'])
        accepted.append(s)
    assert accepted and len(accepted)+len(quarantined)==manifest['valid_corrective_episodes']
    assert {s['id'] for s in accepted}=={r['id'] for r in attempts if r['success'] and r['included']}
    out.mkdir(parents=True);(out/'temporal').mkdir();(out/'export').mkdir();(out/'rgb').mkdir()
    merged=copy.deepcopy(bs);merged['episodes']+=accepted;merged['dataset_sha256']=merged['dataset_sha256_after_read']=sha(a.precollection)
    evidence=copy.deepcopy(be);evidence['episodes'].update({s['id']:re['episodes'][s['id']] for s in accepted})
    gz(out/'temporal/source_snapshot.json.gz',merged);gz(out/'temporal/semantic_evidence.json.gz',evidence)
    for s in merged['episodes']:
        original_file=(base_rgb/s['path']) if s['id'] in original else (recovery/s['path'])
        assert sha(original_file)==s['sha256'];os.symlink(os.path.relpath(original_file.resolve(),out/'rgb'),out/'rgb'/s['path'])
    rows=list(csv.DictReader((base/'export/index.csv').open()));original_arrays=readgz(base/'export/targets.json.gz');arrays=list(original_arrays);primitives=load(base/'export/primitives.json');parent_coverage=[]
    original_ids={r['sample_id'] for r in rows}
    for r in rows:r['source_cohort']='original_run3'
    for s in accepted:
        parts=re['episodes'][s['id']]['primitives'];new_rows,new_arrays,coverage=correction_rows(s,parts)
        rows+=new_rows;arrays+=new_arrays;primitives+=parts;parent_coverage+=coverage
    assert len(rows)==len(arrays) and len({r['sample_id'] for r in rows})==len(rows)
    for i,r in enumerate(rows):r['targets_row']=i
    writecsv(out/'export/index.csv',rows);gz(out/'export/targets.json.gz',arrays);dump(out/'export/primitives.json',primitives)
    candidates=[{'sample_id':r['sample_id'],'recording_id':r['recording_id'],'split':r['split'],'camera_variant':view,'representation':r['representation'],'history_mode':'sparse','green_held_marker':True,'data_variant':'v4','example_id':r['sample_id']+':'+view,'category':r['category'],'source_cohort':r['source_cohort']} for r in rows for view in ['overhead_main','oblique_main']]
    writecsv(out/'export/v4_candidates.csv',candidates)
    dump(out/'export/report.json',{'schema':'m7-run4-actual-student-aggregate-v1','source_files_before':{str(out/'temporal/source_snapshot.json.gz'):sha(out/'temporal/source_snapshot.json.gz')},'original_run3_samples':len(original_ids),'all_original_18_sources_retained':True,'actual_matching_student_sha256':checkpoint_sha,'successful_teacher_suffixes':len(accepted),'expert_failure_correction_episodes':len(actual_corrections),'learner_completed_hold_episodes':len(completed_holds),'timing':'At most4s on original20Hz clock;30 exact anchors maximum; masks exclude learner prefix. Sparse causal1Hz history plus current.','scope':'One-round expert-supervised dataset aggregation, not PPO or Q-learning.','original_data_sha256':before})
    (out/'export/SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted((out/'export').iterdir()) if p.is_file()))
    export_held(out/'temporal',out/'visual-enrichment')
    ds=M7Dataset(out/'export',out/'temporal/source_snapshot.json.gz');checks=[]
    for r in rows:
        p=ds.pack(r['sample_id']);su=p['supervision'];floor=ds.sources[r['recording_id']].get('supervision_action_start',0)
        assert int(r['action_start'])>=floor and all(i>=floor for i in su['source_action_indices'] if i>=0)
        assert su['aligned_duration_seconds']<=4 and all(-4<=t<=0 for t in p['observation']['history']['sparse']['relative_timestamps_s'])
        request={'source':'training_demonstration_supervision','scope':su['scope'],'instruction':su['semantic_instruction'],'aligned_duration_s':su['aligned_duration_seconds'],'anchor_times_s':[t for t,v in zip(su['timestamps_s'],su['valid_mask']) if v]}
        payload={'request':request,'waypoints':[{'time_s':t,'target_generic_interleaved':x} for t,x,v in zip(su['timestamps_s'],su['action'],su['valid_mask']) if v]}
        reconstructed=reconstruct_20hz(payload,oracle_replay=True)['targets'];native=np.asarray(ds.sources[r['recording_id']]['targets'][int(r['action_start']):int(r['action_end_exclusive'])])
        assert len(reconstructed)==len(native) and np.array_equal(reconstructed[:,[9,19]],native[:,[9,19]])
        checks.append({'sample_id':r['sample_id'],'source_cohort':r['source_cohort'],'max_position_interpolation_error_m':float(np.linalg.norm(reconstructed[:,:3]-native[:,:3],axis=1).max())})
        ds.pack(r['sample_id'],'oblique_main')
    plan,quota=balanced(candidates,17);other,other_quota=balanced(candidates,29);assert plan!=other and quota==other_quota
    assert {r['example_id'] for r in plan}=={r['example_id'] for r in candidates}
    assert {r['sample_id'] for r in rows if r['source_cohort']=='original_run3'}==original_ids
    assert arrays[:len(original_ids)]==original_arrays
    writecsv(out/'train_epoch_seed17.csv',plan);writecsv(out/'train_epoch_seed29.csv',other);writecsv(out/'correction_parent_coverage.csv',parent_coverage);writecsv(out/'timing_reconstruction.csv',checks)
    dump(out/'collection_outcomes.json',attempts);dump(out/'quarantined_successful_teacher_episodes.json',quarantined)
    assert all(sha(p)==h for p,h in before.items()) and sha(checkpoint)==checkpoint_sha
    gate={'schema':'m7-run4-matching-student-mixed-gate-v1','ready':bool(actual_corrections),'method':a.method,'algorithm':'one_round_expert_supervised_dataset_aggregation','student_checkpoint_sha256':checkpoint_sha,'student_checkpoint':str(checkpoint),
        'original_source_episodes':18,'original_semantic_windows':len(original_ids),'successful_teacher_suffixes':len(accepted),'correction_episodes':len(actual_corrections),'completed_hold_episodes':len(completed_holds),'failure_correction_ids':actual_corrections,'learner_completed_hold_ids':completed_holds,'all_source_episodes':len(ds.sources),'all_semantic_windows':len(rows),'camera_examples':len(candidates),
        'all_original_targets_unchanged':True,'all_original_splits_retained':True,'all_attempt_outcomes_preserved':True,'quarantined_successful_teacher_episodes':len(quarantined),'quarantine':quarantined,'failed_expert_episodes_in_training':0,'learner_prefix_in_positive_labels':False,
        'training_data':{'export_dir':str(out/'export'),'source_snapshot':str(out/'temporal/source_snapshot.json.gz'),'candidate_manifest':str(out/'export/v4_candidates.csv'),'annotations':str(out/'visual-enrichment/oracle_annotations.jsonl'),'source_root':str(out/'rgb'),'history_mode':'sparse','history_seconds':4,'condition':'crosshair','scope':'mixed','train_splits':['P_train','R_train','S_train','C_train']},
        'balanced_sampling':summarize(plan),'minimum_updates_for_complete_balanced_epoch':(len(plan)+3)//4,'effective_batch':4,'collection_protocol_sha256':sha(recovery/'protocol_before_collection.json'),'precollection_receipt_sha256':sha(a.precollection),'source_inputs_sha256':before,
        'learner_prefix_history':'Past learner images may remain as observation history; only post-handover teacher commands are supervised.',
        'limits':['One collection round from this exact student; continuing updates are not claimed to have newly on-policy data.','Synthetic robot RGB and teacher actions, not raw phone frame/action supervision.','Causal held annotation and independent training failure trigger use declared privileged simulator truth.','Four-second windows retain full intent provenance; longest full intent is not squeezed into one4s action.','No reward optimizer, PPO or Q-learning is implemented.','Learner-completed cases are retained as completion holds and never counted as failure corrections.','Relative RGB symlinks require the original catalog and collection directories to remain available.'],
        'training_started':False,'requires_training_owner_launch':True,'script_sha256':sha(__file__),'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/m7_export_catalog_v2.py',ROOT/'scripts/m7_export_held_catalog.py',ROOT/'scripts/m7_prepare_visual_enrichment.py',ROOT/'supervisor/adaptation/m7_catalog_data.py',ROOT/'supervisor/adaptation/m7_timing.py']}}
    dump(out/'RUN4_GATE.json',gate)
    dump(out/'validation_summary.json',{'status':('PASS' if actual_corrections else 'DATA_VALID_NO_FAILURE_CORRECTION_GATE_CLOSED'),'original_sources':18,'corrective_sources':len(actual_corrections),'learner_completed_hold_sources':len(completed_holds),'rows':len(rows),'max_position_interpolation_error_m':max(r['max_position_interpolation_error_m'] for r in checks),'gate':gate})
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in sorted(out.rglob('*')) if p.is_file() and 'rgb' not in p.relative_to(out).parts))
    print(json.dumps({k:gate[k] for k in ['ready','method','original_source_episodes','correction_episodes','completed_hold_episodes','minimum_updates_for_complete_balanced_epoch','all_semantic_windows','camera_examples','learner_prefix_in_positive_labels']},indent=2))

def self_check():
    targets=np.zeros((205,20));targets[:,0]=np.arange(205)*.001;targets[17:100,9]=1
    frames=sorted(set(range(0,205,4))|{5})
    s={'id':'TEST_ONLY','split':'R_train','category':'learner_state_recovery','supervision_action_start':5,'frame_action_indices':frames,'targets':targets.tolist(),'sha256':'test','source_action_sha256':'test','source_episode_sha256':'test'}
    parts=[{'primitive_id':'t1','label':'grasp_close','action_start':5,'action_end_exclusive':100},{'primitive_id':'t2','label':'lift','action_start':100,'action_end_exclusive':205}]
    rows,arrays,cov=correction_rows(s,parts)
    assert all(int(r['action_start'])>=5 and int(r['action_end_exclusive'])-int(r['action_start'])<=80 for r in rows)
    assert all(int(r['action_start']) in frames for r in rows)
    assert all(all(i>=5 for i in map(int,r['anchor_action_indices'].split())) for r in rows)
    assert any(r['window_index']>0 for r in rows)
    assert all(all(-4<=float(t)<=0 for t in r['sparse_history_relative_timestamps_s'].split()) for r in rows)
    manifest={'learner':'m7','method':'M3','checkpoint_sha256':'right','on_policy_for_matching_v3':True,'learner_state_corrections':True}
    protocol={'learner':'m7','method':'M3','checkpoint_sha256':'right','source_ids':['train'],'source_splits':{'train':'P_train'}}
    validate_identity(manifest,protocol,'M3','right',{'train'})
    for m,p,h in [(dict(manifest,learner='frozen-m6'),protocol,'right'),(manifest,protocol,'wrong'),(manifest,dict(protocol,source_ids=['heldout']),'right')]:
        try:validate_identity(m,p,'M3',h,{'train'})
        except AssertionError:pass
        else:raise AssertionError('Rejected lineage was accepted')
    print(json.dumps({'status':'PASS','test_only_not_collection':True,'off_grid_handover':5,'no_learner_labels':True,'all_parent_intervals_preserved':True,'wrong_checkpoint_frozenM6_heldout_rejected':True,'test_rows':len(rows)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-check',action='store_true');p.add_argument('--base-data',type=Path,default=ROOT/'results/personal-m7-data-prep-20261009/run3-temporal-v3_1');p.add_argument('--base-catalog',type=Path,default=ROOT/'results/personal-m7-data-prep-20261009/training-catalog-v2-complete');p.add_argument('--base-rgb-root',type=Path);p.add_argument('--recovery',type=Path);p.add_argument('--precollection',type=Path);p.add_argument('--student-checkpoint',type=Path);p.add_argument('--method',choices=list(METHODS));p.add_argument('--output',type=Path);a=p.parse_args()
    if a.self_check:self_check()
    else:
        if not all([a.recovery,a.precollection,a.student_checkpoint,a.method,a.output]):p.error('Collection, receipt, checkpoint, method and new output are required')
        main(a)
