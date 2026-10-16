"""Read-only verification of the 16 October decision index; no simulator or model imports."""
from pathlib import Path
import argparse,json,hashlib,subprocess,re,urllib.parse,csv,io
ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
w=args.repo.resolve();d=w/'experiment-history/2026-10-16-session-recovery';p=w/'experiment-history/2026-10-13-del02-package'
def digest(b):return hashlib.sha256(b).hexdigest()
def gitblob(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=w)
base='d3da433054b435a9d1e99b8dc64fe0f516666e51'
before=json.loads(gitblob(base,'experiment-history/2026-10-13-del02-package/claim_index.json'))
now=json.loads((p/'claim_index.json').read_text())
assert now['claims'][:len(before['claims'])]==before['claims']
assert len(now['claims'])==21
assert {c['id']for c in now['claims'][14:]}=={'C15','C16','C17','C18','C19','C20','C21'}
report=w/'experiment-history/2026-10-12-ost-sweep/REPORT.md'
assert report.read_bytes().startswith(gitblob(base,str(report.relative_to(w))))
sources=[]
new_ids={k for c in now['claims'][14:]for k in c['sources']}
for k in sorted(new_ids):
 s=now['sources'][k]
 if s['storage']=='private_repository':
  b=gitblob(s['pinned_commit'],s['path']);assert len(s['pinned_commit'])==40
 elif s['storage']=='verified_archive':
  b=Path(s['path']).read_bytes();assert (w/s['package_copy']).read_bytes()==b
 else:b=(w/s['path']).read_bytes()
 assert digest(b)==s['sha256'],k
 sources.append({'id':k,'sha256':s['sha256'],'verified':True})
core=json.loads(gitblob(now['sources']['init07_table']['pinned_commit'],now['sources']['init07_table']['path']))
expected={'T':[5,0,2,3],'Q0':[8,0,2,2],'Q1':[4,8,2,3]}
numeric=[]
s=now['sources']['init02_table']; rows=list(csv.DictReader(io.StringIO(gitblob(s['pinned_commit'],s['path']).decode())))
assert len(rows)==69
for model,expected_counts in {'P':[0,8],'R':[0,8],'T':[5,0]}.items():
 for panel,value in zip(['initiation','preservation'],expected_counts):
  selected=[r for r in rows if r['model']==model and r['panel']==panel];assert len(selected)==8
  actual=sum(r['qualified']=='True' for r in selected);assert actual==value
  numeric.append({'experiment':'INIT02','model':model,'panel':panel,'expected':value,'actual':actual,'denominator':8})
for model,vals in expected.items():
 for panel,value in zip(['initiation','preservation','prior_feeding','prior_loading'],vals):
  actual=core['panels'][model][panel]['qualified'];assert actual==value
  numeric.append({'model':model,'panel':panel,'expected':value,'actual':actual})
assert not core['Q1_repair_gate_passed']
assert core['gates']['Q1']['lost_T_loading_ids']==['init-demo_0003-loading-912172','init-demo_0005-loading-912171']
for model in ['Q0','Q1']:
 assert core['training'][model]['presentations']==12000
 assert core['training'][model]['checkpoint_sha256']==now['models']['init07-'+model]['checkpoint_sha256']
bound=json.loads((d/'handback-boundaries.json').read_text())
assert bound['prepared_only'] and bound['executions']==0
for c in bound['cases']:
 assert c['native_history_matches_masked_target']
 assert c['native_next_history']==c['native_last_target'][:9]+[0.0]*11
 assert c['expected_d5_next_history'][9:]==[0.0]*11
 assert c['next_query_index']==4 and c['next_action']==c['last_segment_action']+1
 assert c['remaining_policy_actions']==1316 and c['original_passive_settle']==24
 assert c['remaining_policy_actions']+c['policy_actions_used_before_handback']==1436
 assert c['query4_noise_reference']
 assert Path(c['query4_input_audit']).is_file()
 assert digest(Path(c['query4_input_audit']).read_bytes())==c['query4_input_audit_sha256']
receipt=json.loads((d/'recovery_receipt.json').read_text())
assert receipt['new_executions']==receipt['new_training_updates']==receipt['new_inference_calls']==0
assert not receipt['compute']['extension_verified']
assert 'prepared_only' in receipt['init11_status']
for a in receipt['archives']:
 manifest=Path(a['archive'])/'raw-manifest.json'
 assert digest(manifest.read_bytes())==a['manifest_sha256']
 assert digest((Path(a['archive'])/'raw-verification.json').read_bytes())==a['verification_receipt_sha256']
for key,m in receipt['models'].items():
 assert len(m['checkpoint_sha256'])==64
 if 'path'in m:assert Path(m['path']).stat().st_size==m['present_bytes']
links=[]
for file in [report,p/'CLAIM_INDEX.md',d/'INIT11_PROTOCOL.md',d/'README.md']:
 for raw in re.findall(r'\]\(([^)]+)\)',file.read_text()):
  target=raw.split()[0]
  if target.startswith(('http://','https://','mailto:','#')):continue
  path=urllib.parse.unquote(target.split('#',1)[0]);resolved=(file.parent/path).resolve();assert resolved.exists(),(str(file),path)
  links.append({'from':str(file.relative_to(w)),'target':path})
result={'passed':True,'previous_14_claims_preserved':True,'previous_report_preserved_as_prefix':True,'new_claims':7,'new_source_checks':sources,'numeric_checks':numeric,'handback_cases_checked':len(bound['cases']),'relative_links_checked':len(links),'links':links,'large_weights_rehashed':False,'new_simulator_calls':0,'new_model_calls':0,'protocol_executions':0}
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in ['new_source_checks','numeric_checks','links']}))
