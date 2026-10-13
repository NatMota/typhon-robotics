"""Actor-free acceptance review using existing observations, features and explicit agent notes."""
import argparse, ast, csv, hashlib, json, struct, zipfile
from pathlib import Path
from datetime import datetime, timezone

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def npy(z,name):
    b=z.read(name);ver=b[6];n=struct.unpack('<H' if ver==1 else '<I',b[8:10 if ver==1 else 12])[0]
    off=10 if ver==1 else 12
    return ast.literal_eval(b[off:off+n].decode().strip()),b[off+n:]
def main(root,archive):
    notes=json.loads((root/'agent_visual_review.json').read_text());inputs=json.loads((root/'source_ledger_inputs.json').read_text())
    lock=json.loads((root/'candidate_lock.json').read_text()); selected={x['demo_id']:x for x in lock['examples']}
    dense=archive/'dense-v2';ix=json.loads((dense/'source-index.json').read_text());samples=ix['samples']
    z=zipfile.ZipFile(dense/'features/features.npz');ih,ib=npy(z,'sample_ids.npy')
    width=int(ih['descr'].split('U')[1]); ids=[ib[i:i+width*4].decode('utf-32le').rstrip('\0') for i in range(0,len(ib),width*4)]
    assert ids==[s['sample_id'] for s in samples]
    arrays={}
    for name in ['full_rgb','current_observed_object_square_crop']:
        h,b=npy(z,name+'.npy');assert h['shape']==(2644,1024) and h['descr']=='<f4';arrays[name]=(h,b)
    trajectories=json.loads((dense/'analysis-v2/trajectory-summary.json').read_text())['sequences']
    goals=json.loads((dense/'goal-probe/results.json').read_text())['goal_similarity']
    comparisons=json.loads((dense/'analysis-v2/feature-comparison.json').read_text())
    authorities={
      'pickup_subgoal':{'source':'User conversation: explicit three subgoals (pickup, placement in square, orientation), plus M0 demonstration specification grasp/lift/place/release.',
                       'requirement':'Under the requested pickup-place assessment, reaching the same endpoint by pushing does not satisfy the pickup subgoal. No exact human route, grasp shape, speed or lift height is required.'},
      'preserve_completed':{'source':'User M1.5 requirement for already-completed examples.',
                       'quote':'Initialize the block inside the goal tolerance and verify that task-completion logic recognises the completed state without unnecessary manipulation.',
                       'requirement':'Avoid unnecessary pickup/replacement when the goal is already satisfied. Initial-state recognition plus this supplied rule can suffice; a full video is not proven necessary for that decision.'},
      'orientation_goal':{'source':'User M0 task convention and confirmed correct example.',
                       'requirement':'Finish with blue 2 upright and studs toward the top of the primary view. Exact intermediate orientation path is unspecified.'},
      'pause_intent':{'source':'User labelled four takes awkward/pause-resume; M1.5 explicitly says annotate pauses and do not assume the human pause itself is a robot action.',
                       'requirement':'Preserve the demonstration intent and annotation. No independently imposed mandatory wait, obstacle or external intervention follows from the label.'}}
    save(root/'requirement_authorities.json',authorities)
    ledger=[];examples=[]
    for d in inputs:
        did=d['demo_id'];n=notes['notes'][did];a=d['annotations'];cat=d['manual_category'];refs=d['reviewed_frame_requests']
        isfeed=cat in ['normal_feeding','awkward_pause_resume_feeding'];complete=cat=='already_completed'
        if isfeed:
            authority=['pickup_subgoal','orientation_goal']+(['pause_intent'] if cat=='awkward_pause_resume_feeding' else [])
            requirement='Pickup/lift then release matters under the requested subgoal assessment; exact observed route, pace and rotation timing are not independently required.'
            counterfactual='Robot could push the block to the same correct endpoint; this would fail the explicitly requested pickup subgoal. This does not require imitation of the full human path.'
            consequence='Track a qualified held/lifted/released phase, keep unknown contact explicit, and verify placement; do not copy pauses as a stop command.'
        elif complete:
            authority=['preserve_completed','orientation_goal'];requirement='Preserve an already-satisfied goal without unnecessary manipulation.'
            counterfactual='Robot could pick up and replace the block at identical endpoints; this violates the supplied no-unnecessary-manipulation requirement.'
            consequence='Recognise the current goal and avoid a new manipulation. Temporal observation can audit preservation, but its necessity for the initial decision is unproven.'
        else:
            authority=['orientation_goal'];requirement='Final orientation is required; no independently required intermediate path was found in reviewed source evidence.'
            counterfactual='Different rotations or paths can reach the same upright goal. No independent requirement Y was established that their path would violate.'
            consequence='Use the observable incorrect start and desired final yaw; preserve occlusion uncertainty. Do not demand the exact human intermediate path.'
        dense_rows=[(i,s) for i,s in enumerate(samples) if s['demo_id']==did]
        counts={v:{'samples':sum(s['view']==v for _,s in dense_rows),
                   'valid_xy':sum(s['view']==v and s['valid_xy'] for _,s in dense_rows),
                   'known_yaw':sum(s['view']==v and s['yaw_clockwise_deg'] is not None for _,s in dense_rows),
                   'with_hand':sum(s['view']==v and bool(s['hands']) for _,s in dense_rows)} for v in ['camera_a','camera_b']}
        row={'demo_id':did,'category':cat,'initial_final_information':{'initial_position_annotation':a.get('initial_position_relative_to_target'),'initial_orientation_annotation':a.get('initial_orientation_visual'),'final_outcome_annotation':a.get('outcome'),'provenance':'Frozen M0 annotations, separately checked only at the listed sampled review instants.'},
             'additional_temporal_information':n['observation'],'information_class':n['kind'],'raw_frame_time_evidence':refs,
             'sync_uncertainty_s':d['sync_uncertainty_s'],'boundary_uncertainty_s':d['boundary_uncertainty_s'],
             'task_relevance':requirement,'requirement_authority_ids':authority,'same_endpoint_counterfactual':counterfactual,
             'robot_consequence':consequence,
             'representation_support':{'retained':'Ordered source frames/times and confidence-masked 2D object/yaw observations; supported hand landmarks, not physical contact.',
                  'suggested':'Existing VIDEO03-05 probes support held versus unheld on selected labelled interiors, but known approach false positives prevent autonomous grasp/release authority.',
                  'missing':'Qualified metric depth, support/contact, precise grasp/release transitions and human-to-robot action targets.',
                  'untested':'Whether frozen descriptors and a causal policy use this intermediate distinction to improve robot decisions over endpoint/current-state controls.'},
             'existing_dense_counts':counts,'review_limitations':[n['limit'],'Seven instants per view; absence between samples is not established. Original paired-view synchronization uncertainty remains.','Agent visual observations are not new human approval or model semantics.'],
             'negative_control':did=='demo_0007'}
        ledger.append(row)
        if did not in selected:continue
        matches=[]
        for r in refs:
            if r['sample_index'] not in selected[did]['sample_indices']:continue
            i,s=min([(i,s) for i,s in dense_rows if s['view']==r['view']],key=lambda x:abs(x[1]['source_time_s']-r['source_time_decoded_s']))
            delta=s['source_time_s']-r['source_time_decoded_s']
            matches.append({'review_view':r['view'],'review_sample_index':r['sample_index'],'raw_review_frame_index':r['source_frame_index'],
                'raw_review_pts_s':r['source_time_decoded_s'],'frozen_sample_id':s['sample_id'],'frozen_source_frame_index':s['source_frame_index'],
                'frozen_pts_s':s['source_time_s'],'frozen_minus_review_time_s':delta,'same_native_frame':s['source_frame_index']==r['source_frame_index'],
                'feature_row_index':i,'feature_row_sha256':{name:hashlib.sha256(b[i*4096:(i+1)*4096]).hexdigest() for name,(h,b) in arrays.items()},
                'is_grasped':s['is_grasped'],'metric_xyz':s['metric_xyz'],'valid_xy':s['valid_xy'],'known_yaw':s['yaw_clockwise_deg'] is not None,'hand_detected':bool(s['hands'])})
        examples.append({'demo_id':did,'selection':selected[did],'source_review':n,'frame_feature_correspondence':matches,
                         'existing_trajectory_summary':[x for x in trajectories if x['demo_id']==did],
                         'existing_goal_similarity':[x for x in goals if x['demo_id']==did],
                         'existing_timing_retrieval':[x for x in comparisons['within_known_pair'] if x['demo_id']==did],
                         'interpretation':'Existing retrieval/goal-similarity quantities are descriptive offline results, not causal phase accuracy. No new similarity metric, fit or threshold selection.'})
    assert len(ledger)==16 and len(examples)==3 and all(r['demo_id']!='demo_0017' for r in ledger)
    save(root/'temporal_information_ledger.json',ledger);save(root/'selected_examples.json',examples)
    with (root/'temporal_information_ledger.csv').open('w',newline='') as f:
        fields=['demo_id','category','initial_final_information','additional_temporal_information','information_class','raw_frame_time_evidence','sync_uncertainty_s','task_relevance','requirement_authority_ids','same_endpoint_counterfactual','robot_consequence','representation_support','review_limitations']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in ledger:
            vals={k:row[k] for k in fields};vals['raw_frame_time_evidence']=[{'view':r['view'],'recording_id':r['recording_id'],'frame':r['source_frame_index'],'pts_s':r['source_time_decoded_s']} for r in row['raw_frame_time_evidence']]
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in vals.items()})
    verify={'schema':'video01-c-review-verification-v1','created_utc':datetime.now(timezone.utc).isoformat(),'raw_recordings_verified_by_preparer':8,
        'curated_takes':16,'paired_review_instants':112,'view_samples':224,'excluded_demo_0017_absent':True,'selected_example_ids':[x['demo_id'] for x in examples],
        'frozen_feature_index_order_verified':len(ids),'frozen_feature_shapes':{n:list(h['shape']) for n,(h,b) in arrays.items()},
        'all_dense_physical_contact_unknown':all(s['is_grasped'] is None for s in samples),
        'all_dense_metric_xyz_unknown':all(s['metric_xyz'] is None for s in samples),
        'new_model_fits':0,'new_physics_rollouts':0,'new_classifier_or_similarity_thresholds':0,
        'input_hashes':{str(p.relative_to(archive)):sha(p) for p in [dense/'source-index.json',dense/'features/features.npz',dense/'analysis-v2/trajectory-summary.json',dense/'goal-probe/results.json',dense/'analysis-v2/feature-comparison.json']}}
    save(root/'representation_audit.json',verify);print(json.dumps(verify))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);a=p.parse_args();main(a.root,a.archive)
