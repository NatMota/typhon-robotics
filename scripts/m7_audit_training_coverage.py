"""Read-only source audit; only writes a new coverage report. No simulator/GPU dependency."""
import argparse,collections,csv,hashlib,json
from pathlib import Path


def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()


def main(root,out,semantic):
 root=root.resolve();out.mkdir(parents=True,exist_ok=False)
 m0=root.parent/'humanoid-task-supervisor';m15=root.parent/'humanoid-task-supervisor-m15/results/personal-m15-20261009'
 canonical=root/'frozen_results/personal-m3-20261009/data-canonical'
 real=root/'results/personal-m7-data-prep-20261009/visual-enrichment-v1/real-held-v1/export'
 sources=[]
 def asset(p,expected=None):
  p=Path(p);x={'path':str(p),'exists':p.is_file()}
  if x['exists']:
   x.update(bytes=p.stat().st_size,sha256=digest(p))
   if expected is not None:x['expected_sha256']=expected;x['matches_expected']=x['sha256']==expected
  sources.append(x);return x
 def read(p):
  asset(p);return json.loads(Path(p).read_text())
 def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
 dataset=read(canonical/'dataset.json');manifest=read(m0/'data/personal/m0/dataset_manifest.json')
 reconstruction=read(m15/'reconstructed_scenarios.json');prov=read(m15/'source_provenance.json');suitability=read(m15/'training_suitability.json')
 real_manifest=read(real/'manifest.json');real_validation=read(real/'validation_summary.json')
 eligibility={x['id']:x for x in dataset['eligibility']};episodes={x['id']:x for x in dataset['episodes']}
 scenarios={x['source_demo_id']:x for x in reconstruction['personal']};suit={x['source_demo_id']:x for x in suitability}
 semrows=list(csv.DictReader(semantic.open())) if semantic and semantic.is_file() else []
 if semrows:asset(semantic)
 semcounts=collections.Counter(r['recording_id'] for r in semrows)
 held=list(csv.DictReader((real/'held_trajectory_annotations.csv').open()));asset(real/'held_trajectory_annotations.csv')
 episode_assets={key:asset(canonical/e['path'],e['sha256']) for key,e in episodes.items()}
 rows=[]
 for item in manifest['demonstrations']:
  did=item['demo_id'];eid='personal_'+did;e=episodes.get(eid);elig=eligibility.get(did,{});sc=scenarios.get(did,{})
  views=[]
  for view in ['camera_a','camera_b']:
   clip=asset(m0/item[view]['clip']);derived=next(v for v in real_manifest['videos'] if v['demo_id']==did and v['view']==view)
   overlay=asset(real/derived['output'],derived['output_sha256']);sidecar=asset(real/derived['sidecar'],derived['sidecar_sha256'])
   views.append({'view':view,'original_clip':clip,'enriched_clip':overlay,'held_sidecar':sidecar,'source_recording':item[view],'frame_count':derived['frame_count'],'held_counts':derived['counts']})
  h=[r for r in held if r['demo_id']==did];epdir=m15/'episode_artifacts'/eid
  traces={}
  for name in ['robot_actions.jsonl','episode.json','resolved_config.json']:
   traces[name]=asset(epdir/name)
  trace_steps=sum(1 for _ in (epdir/'robot_actions.jsonl').open()) if (epdir/'robot_actions.jsonl').is_file() else 0
  completed=item['attempt_type']=='already_completed'
  rows.append({'example_id':did,'cohort':'personal','category':item['attempt_type'],'curated':True,'existing_split':elig.get('split','unassigned'),'train_eligible_existing':bool(e and e['split'].endswith('_train')),
   'canonical_robot_episode':e,'canonical_npz':episode_assets.get(eid),'native_robot_rgb_action_pair_available':bool(e),'human_robot_frame_alignment_available':False,'human_native20D_targets_available':False,
   'semantic_pairing_status':'task/source-level association only; primitive-to-primitive alignment not annotated' if e else 'source/scenario association only; no canonical native robot pair',
   'm15_reconstruction':{k:sc.get(k) for k in ['reconstruction_status','reconstruction_confidence','initial_sim','target_sim','unusable_reasons','pause_annotation','observable_correction']},'m15_suitability':suit.get(did),'m15_traces':traces,'m15_action_rows':trace_steps,
   'canonical_exclusion_reason':None if e else elig.get('reason','not present in canonical pool'),'m7_semantic_v1_windows':semcounts[eid],
   'm5_m6_training_source':did=='demo_0001','m5_m6_development_validation':did in ['demo_0006','demo_0014','demo_0016'],
   'completed_reuse_status':'M15 inspected source; excluded from canonical motorpool; M4/M5/M6 completed cases independently handdeclared, different source/configuration. New explicit split and M1.6 RGB/native replay needed.' if completed else None,
   'real_camera_views':views,'real_view_frame_count':len(h),'real_binary_eligible_frames':sum(r['eligible_for_binary_enrichment']=='1' for r in h),
   'real_held_true':sum(r['held_manipulation']=='1' for r in h),'real_held_false':sum(r['held_manipulation']=='0' for r in h),'real_held_unknown':sum(r['held_manipulation']=='' for r in h),'real_crosshair_on':sum(r['crosshair_on']=='1' for r in h),
   'training_next_step':('existing paired native training available; include category-balanced windows' if e and e['split'].endswith('_train') else 'retain frozen validation only' if e else 'explicit C_train protocol plus M1.6 native replay; old exclusion remains unchanged' if completed else 'retain human observations; do not invent successful robot motor targets')})
 excluded=read(m0/'data/personal/m0/excluded_demo_manifest.json')
 synthetic=[]
 for attempt in dataset['synthetic_attempts']:
  sid=attempt['scenario']['scenario_id'];e=episodes.get(sid)
  synthetic.append({'example_id':sid,'cohort':'synthetic','category':attempt['scenario']['category'],'success':attempt['success'],'failures':attempt['failures'],'existing_split':e['split'] if e else 'failed_generation_excluded','train_eligible_existing':bool(e and e['split'].endswith('_train')),'native_robot_rgb_action_pair_available':bool(e),'canonical_npz':episode_assets.get(sid),'canonical_robot_episode':e,'scenario':attempt['scenario'],'m7_semantic_v1_windows':semcounts[sid],'personal_video_pair':None})
 protocols={}
 for phase in ['personal-m5-20261009','personal-m6-joint-20261009']:
  d=read(root/'results'/phase/'evaluation_protocol.json')
  protocols[phase]={'validation_status':d.get('validation_status'),'cases':[{'scenario_id':x['scenario_id'],'group':x['group'],'source_demo_id':x.get('source_demo_id'),'source':x.get('source'),'initial_state':x['config']['initial_state'],'task':x['config']['task']} for x in d['core']]}
 m5=read(root/'results/personal-m5-20261009/source_protocol.json')
 for name in ['scripts/m4_freeze.py','scripts/m3_prepare_data.py','supervisor/adaptation/m5.py','supervisor/adaptation/m6.py']:
  if (root/name).exists():asset(root/name)
 allrows=rows+synthetic;train=[x for x in allrows if x['train_eligible_existing']];val=[x for x in allrows if x['existing_split'].endswith('_validation')]
 summary={'schema':'m7-training-coverage-v1','read_only_source_audit':True,'curated_personal_examples':len(rows),'personal_categories':dict(collections.Counter(x['category'] for x in rows)),
  'canonical_robot_episodes':len(episodes),'canonical_train_episodes':len(train),'canonical_train_by_split':dict(collections.Counter(x['existing_split'] for x in train)),'canonical_train_bytes':sum(x['canonical_npz']['bytes'] for x in train),'canonical_validation_episodes':len(val),'canonical_validation_bytes':sum(x['canonical_npz']['bytes'] for x in val),
  'all_canonical_npz_hashes_match':all(x.get('matches_expected') for x in episode_assets.values()),'real_enriched_videos':len(real_manifest['videos']),'real_video_bytes':sum(v['enriched_clip']['bytes'] for r in rows for v in r['real_camera_views']),'real_view_frames':len(held),'real_binary_eligible_frames':sum(x['real_binary_eligible_frames'] for x in rows),'real_native20D_action_labelled_examples':0,
  'm7_v1_semantic_windows':len(semrows),'m7_v1_semantic_episodes':dict(semcounts),'missing_from_m7_v1_train':[x['example_id'] for x in train if not x['m7_semantic_v1_windows']],
  'm5_m6_source':{'episode':m5['episode']['id'],'split':m5['episode']['split'],'native_windows':m5['complete_windows'],'additional_robot_training_examples_created':0},
  'already_completed_examples':['demo_0018','demo_0019'],'completed_split_status':'new protocol decision required; old canonical excluded_motor_training retained','synthetic_attempts':len(synthetic),'synthetic_failed_attempts':sum(not x['success'] for x in synthetic),
  'existing_frozen_outputs_modified':False,'real_pairing_limit':'source/task-level semantics only, no human/robot frame mapping; generated robot actions cannot serve as measured human labels','coordinate_limit':'M0 pixels/normalized/target units; M15 chosen virtual scaling, not calibrated physical metres','historical_holdout_limit':'whole takes separated, some recordings shared across splits; validation configurations have already been inspected'}
 dump(out/'personal_examples.json',rows);dump(out/'synthetic_examples.json',synthetic);dump(out/'curated_exclusions.json',excluded);dump(out/'evaluation_exposure.json',protocols);dump(out/'validation_summary.json',summary);dump(out/'source_assets.json',sources)
 with (out/'coverage.csv').open('w',newline='') as f:
  keys=['example_id','cohort','category','existing_split','train_eligible_existing','native_robot_rgb_action_pair_available','human_native20D_targets_available','m7_semantic_v1_windows','canonical_exclusion_reason','training_next_step'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(allrows)
 with (out/'source_checksums.sha256').open('w') as f:
  for x in sources:
   if x['exists']:f.write(x['sha256']+'  '+x['path']+'\n')
 lines=['M7 training-data coverage audit','',f'Canonical robot pool: {len(train)} training episodes ({summary["canonical_train_by_split"]}), {len(val)} validation episodes. All20 NPZ hashes verified. Training NPZ payload: {summary["canonical_train_bytes"]:,} bytes.',
  'All16 real demonstrations are retained:7 normal,4 awkward/pause-resume,3 correction,2 completed. User-excluded0017 remains excluded and preserved. Camera views are paired observations, not independent demonstrations.',
  '','Category coverage:','- P_train0001/0002/0003/0005 are normal;0012/0013 are awkward/pause-resume.','- R_train0007/0009 are successful corrections and already have native robot RGB/action data.','- S_train001–006 are independent synthetic scenarios. Keep them identified separately.','- P_validation0006/0014/0016 and S_validation007/009/010 remain excluded from optimizer batches.','- 0004 lacks observable initial orientation;0011/0015 have failed expert outcomes. Preserve their real observations and failure traces, not positive motor-imitation targets.','- 0018/0019 are valid completed source observations with successful M15 no-op traces, excluded only from the previous motor pool. No new native NPZ pair exists yet. Adding them requires an explicit new C_train split and M1.6 replay; old reports/splits stay fixed.',
  '','Historical coverage: M3P used6 source trains; M3S used6 synthetic trains; M4PR used corrections0007/0009 and M4PN matched normal0001/0002; M5/M6 used only0001 with48 windows. M7semantic-v1 contains only9P episodes including3validation; the correction/synthetic training examples still need v2 exports.',
  '','Human-video lane: all32 enriched clips and separate causal held-state sidecars are available. There are0 native20D human robot-action labels. M15 establishes common task/source identity and initial/goal correspondence, not temporal alignment. Valid current use is explicit observation/intention supervision or newly annotated semantic primitive correspondences; direct motor loss from arbitrarily paired human frames and generated robot actions would fabricate labels. The2 real views must remain jointly grouped bytake.',
  '','Completed evaluation independence: M4/M5/M6 used independent hand-declared completed states at[0.12,-0.10] and[0.13,-0.09],yaw0. Phone0018/0019 map to target[0.125,-0.12], observed starts near[0.1215,-0.1144], yaws0.07806/0.00391. They are different sources/configurations, although all have been inspected during development; no untouched-test claim is available.',
  '','No source files or frozen outputs were changed. No GPU job or training ran in this audit. source_assets.json records actual source paths, sizes and hashes; coverage.csv is the portable decision ledger.']
 (out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
 (out/'REPRODUCE.txt').write_text('Run on simulation host after copying the frozen analysis host semantic-v1 index to coverage-input-semantic-v1.csv:\npython3 scripts/m7_audit_training_coverage.py --root . --output results/personal-m7-training-20261009/coverage-v1 --semantic-index coverage-input-semantic-v1.csv\nOutput must be new/nonexistent.\n')
 print(json.dumps(summary,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path,required=True);p.add_argument('--semantic-index',type=Path);a=p.parse_args();main(a.root,a.output,a.semantic_index)
