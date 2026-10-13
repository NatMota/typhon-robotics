from pathlib import Path
import json,hashlib,subprocess,datetime,os
Q=Path('/media/nm/Fast Drive/Typhon/research-20261012');R=Q.parent
W=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
BASE='19aada67454b28821d3ed8660f3e02ed87a0286e'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
repo_specs={
 'eval_report':'experiment-history/2026-10-13-eval02-review/CLAIM_STATUS.md',
 'eval_ledger':'experiment-history/2026-10-13-eval02-review/comparison_ledger.json',
 'data07_report':'experiment-history/2026-10-13-data07-review/REPORT.md',
 'data07_summary':'experiment-history/2026-10-13-data07-review/summary.json',
 'data07_ledger':'experiment-history/2026-10-13-data07-review/episode_ledger.json',
 'del01_manifest':'experiment-history/2026-10-13-del01-review/media_claim_manifest.json',
 'del01_report':'experiment-history/2026-10-13-del01-review/CORRECTIONS.md',
 'del01_validation':'experiment-history/2026-10-13-del01-review/validation_summary.json',
 'model_registry':'experiment-history/2026-10-12-ost-sweep/MODEL_REGISTRY.md',
 'confirm_protocol':'experiment-history/2026-10-12-ost-sweep/protocols/confirmatory-v2/protocol.json',
 'goal_protocol':'experiment-history/2026-10-12-ost-sweep/protocols/goal-evaluation/evaluation-protocol.json',
 'goal_train_protocol':'experiment-history/2026-10-12-ost-sweep/protocols/goal-grounding/training-protocol.json',
 'recovery_protocol':'experiment-history/2026-10-12-ost-sweep/protocols/recovery-evaluation/evaluation-protocol.json',
 'goal_text_probes':'experiment-history/2026-10-12-ost-sweep/goal-text-probes.json',
 'original_subgoal_code':'scripts/film_feeding_subgoal_audit_20261011.py',
 'video10_report':'experiment-history/2026-10-13-video10/REPORT.md',
 'video11_report':'experiment-history/2026-10-13-video11/REPORT.md',
 'video08_report':'experiment-history/2026-10-13-video08/REPORT.md'
}
arc=R/'research-20261011'
archive_specs={
 'suite775':R/'evaluations/film-feeding-775-20261011/suite.json',
 'broad_source_report':arc/'broad-pt/artifacts/source-competence/SOURCE_COMPETENCE_REPORT.md',
 'training_audit':Q/'ost-sweep-20261012/audits/training-data-audit.json',
 'release_report':arc/'release-prompts/release-audit/REPORT.md',
 'release_provenance':arc/'release-prompts/release-audit/source-provenance.json',
 'release_summary':arc/'release-prompts/release-audit/summary.json',
 'blue_report':arc/'blue2-causal-20261011-v1/REPORT.md',
 'blue_protocol':arc/'blue2-causal-20261011-v1/protocol.json',
 'blue_coherent_report':arc/'blue2-causal-20261011-v2-coherent-goal/REPORT.md',
 'blue_coherent_protocol':arc/'blue2-causal-20261011-v2-coherent-goal/protocol.json',
 'input_audit':arc/'orientation-input-audit/REPORT.md',
 'visibility_protocol':Q/'ost-sweep-20261012/visibility/protocol.json',
 'visibility_summary':Q/'ost-sweep-20261012/visibility/summary.json',
 'prompt_report':arc/'release-prompts/prompt-controls-vanilla-v3-v4/REPORT.md',
 'prompt_protocol':arc/'release-prompts/prompt-controls-vanilla-v3-v4/protocol.json',
 'prompt_summary':arc/'release-prompts/prompt-controls-vanilla-v3-v4/summary.json',
 'replan_report':arc/'replanning-candidates/REPORT.md',
 'replan_protocol':arc/'replanning-candidates/protocol.json',
 'hierarchy_report':arc/'hierarchy-tranche4-final/REPORT-final.txt',
 'hierarchy_protocol':arc/'hierarchy-tranche4-final/paired-panel-v2/protocol.json',
 'video10_protocol':Q/'video10/data/protocol.json',
 'video10_analysis':Q/'video10/data/analysis.json'
}
for d in ['live06','live07','live08']:
 archive_specs[d+'_report']=Q/d/'REPORT.md';archive_specs[d+'_protocol']=Q/d/'protocol.json'
sources={}
for k,rel in repo_specs.items():
 p=W/rel;assert p.is_file(),p
 frozen=subprocess.check_output(['git','show',BASE+':'+rel],cwd=W)
 assert hashlib.sha256(frozen).hexdigest()==sha(p),rel
 sources[k]={'storage':'private_repository','path':rel,'sha256':sha(p),'pinned_commit':BASE,'url':'https://github.com/NatMota/typhon-robotics/blob/'+BASE+'/'+rel}
for k,p in archive_specs.items():
 assert p.is_file(),p
 sources[k]={'storage':'Mint_archive','path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'public_url':None}
l=load(W/repo_specs['eval_ledger']);media=load(W/repo_specs['del01_manifest']);d7=load(W/repo_specs['data07_summary'])
headers=l['model_lineage_and_data']['verified_saved_headers']
models={'original-v'+str(i):{'checkpoint_sha256':l['original775']['models_sha256']['v'+str(i)+'-m6'],'identity_evidence':'eval_ledger: original775.models_sha256'}for i in range(1,5)}
for k,tail in [('narrow-3k','/training/final.pt'),('broad-3k','/step3000.pt'),('broad-12k','/step12000.pt')]:
 row=next(h for h in headers if h['path'].endswith(tail));models[k]={'checkpoint_sha256':row['sha256'],'recorded_path':row['path'],'identity_evidence':'eval_ledger: verified_saved_headers'}
basecfg=headers[0]['header']['config']
models['vanilla']={'base_revision':basecfg['base_revision'],'base_weights_sha256':basecfg['base_weights_sha256'],'domain_id':3,'identity_evidence':'pinned base identity; original suite native contract'}
for t in l['later_training']:models[t['model']]={'checkpoint_sha256':t['checkpoint_sha256'],'parent_sha256':t['parent_sha256'],'updates':t['updates'],'fit_seed':t['seed'],'actual_unique_trajectories':len(t['actual_trajectory_draws']),'windows':t['windows'],'identity_evidence':'eval_ledger: later_training'}
vr=next(r for r in media['records']if r['slot']=='video10_seq')
models['video10-controller']={'kind':'supplied visual-servo controller, no learned checkpoint','identity':vr['method'],'assistance':vr['observation_assistance']}
claims=load(P/'claim_spec.json')
for c in claims:
 assert all(k in sources for k in c['sources']),c['id']
 assert all(k in models for k in c['models']),c['id']
 assert all(any(r['slot']==s for r in media['records'])for s in c['media']),c['id']
 c['media_manifest_source']='del01_manifest'
index={'schema':'typhon-private-claim-index-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ticket':11008,'synthesis':'experiment-history/2026-10-12-ost-sweep/REPORT.md','pinned_audits':{'DATA07':'7620c1a68409ab2d2ca85f6fc69b368611f074f3','EVAL02R':'dda6e135bde90e732dbf69845ac25003b629921e','DEL01':BASE},'models':models,'claims':claims,'sources':sources,'fresh_checkpoint_rehash_performed':False,'identity_scope':'Prior verified headers/weight hashes and episode provenance reused; DEL02 verifies these evidence files, not multi-GB weights.'}
(P/'claim_index.json').write_text(json.dumps(index,indent=2)+'\n')
lines=['# Claim-to-evidence index','','The synthesis is the existing [research report](../2026-10-12-ost-sweep/REPORT.md). Full checkpoint identities, source hashes and exact locators are in [claim_index.json](claim_index.json). Archive-only evidence has no invented public URL.','','| ID | Claim | Audited sources |','|---|---|---|']
for c in claims:
 refs=[]
 for k in c['sources']:
  s=sources[k]
  refs.append('['+k+']('+os.path.relpath(W/s['path'],P)+')'if s['storage']=='private_repository'else k+' (Mint path/hash in JSON)')
 lines.append('| '+c['id']+' | '+c['claim']+' | '+'; '.join(refs)+' |')
lines+=['','Numerical and mechanism scopes are kept in each claim’s limit field. A linked retrospective region analysis does not replace the original strict endpoint. Legacy 70 mm pickup and later 10 mm supported-lift definitions remain separate.','']
(P/'CLAIM_INDEX.md').write_text('\n'.join(lines))
numeric=[]
def check(name,actual,expected):
 assert actual==expected,(name,actual,expected)
 numeric.append({'claim':name,'actual':actual,'expected':expected,'passed':True})
groups=l['later_panels']['groups']
def group(method,cat):return next(r for r in groups if r['method']==method and r['group']==cat)
for m,cat,expected in [('broad-Pt-12000','near_personal',[40,31,22,17]),('vanilla-Pt-native','near_personal',[40,0,0,0]),('broad-Pt-12000','broader_starts',[20,0,0,0]),('vanilla-Pt-native','broader_starts',[20,0,0,0]),('varied','new_start_goal_combination',[12,12,10,10]),('fixed','new_start_goal_combination',[12,12,0,0])]:
 g=group(m,cat);check(m+'/'+cat,[g[k]for k in ['n','supported_grasp_10mm','operational_place','strict']],expected)
for m in ['base12k','fixed','varied','recovery']:
 check(m+'/loading',[group(m,'loading_transfer_no_B_labels')[k]for k in ['n','strict']],[12,0])
 check(m+'/repair',[group(m,'quarantined_failure_state')[k]for k in ['n','strict']],[4,0])
for m,expected in [('broad-Pt-12000',8),('vanilla-Pt-native',1)]:check(m+'/preservation',[group(m,'already_completed')[k]for k in ['n','strict']],[8,expected])
for r in l['original775']['results']:check(r['model']+'/no_new_strict',[r['n'],r['initially_incomplete_n'],r['new_strict_completions'],r['new_region_place_eligible_n']],[155,137,0,116])
for m,vals in [('v3-m6',[10,5,4]),('v4-m6',[1,1,0])]:
 r=next(x for x in l['original775']['results']if x['model']==m)
 check(m+'/partial',[r['pickup_legacy_70mm'],r['posthoc']['new_region_placement'],r['posthoc']['new_region_task_success']],vals)
check('catalog subset',[d7['original_catalog_episodes'],d7['broad_overlap_count'],d7['later_admitted_corrections'],d7['original_v4_arrays_currently_verified']],[18,6,4,0])
g=group('broad-Pt-12000','near_personal')
check('near pooled',g['pooled_success_fraction'],.425)
check('near family interval rounded',[round(100*x,2)for x in [g['strict_family_bootstrap']['family_balanced_mean'],*g['strict_family_bootstrap']['percentile95']]],[41.67,22.22,55.56])
for x in l['broad_development_and_source']:
 for g in x['groups']:check(x['panel']+'/'+g['method'],[g['n'],g['grasp_10mm_bilateral_3steps'],g['operational_place'],g['strict']],[6,6,5,4]if x['panel']=='source'else[6,0,0,0])
check('later updates',[t['updates']for t in l['later_training']],[3000,3000,3000])
check('varied/fixed data counts',[models['fixed']['actual_unique_trajectories'],models['varied']['actual_unique_trajectories']],[6,12])
check('media count',len(media['records']),8)
check('derived copies',sum('derived_video'in r for r in media['records']),2)
check('selected hosted matches',sum(len(r['hosted_copies']['verified_matching_entries'])for r in media['records']),0)
check('completed baselines',sum(g['n']for g in groups if g['panel']=='baselines'),136)
check('unrun intervention arithmetic',204-136,68)
(P/'numerical_checks.json').write_text(json.dumps(numeric,indent=2)+'\n')
print(json.dumps({'source_files':len(sources),'claims':len(claims),'model_identities':len(models),'numerical_checks':len(numeric)}))
