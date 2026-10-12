"""Audit the frozen offline visual critic without treating repeated orders as new states."""
import sys,json,hashlib,collections,statistics,csv,shutil,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import exp20261011_ghost_critic_v2 as critic
import exp20261011_ghost_visual_goal_compare_v3 as secondary
from live09_ghost_decisions import SETS,SOURCE,sha
OUT=Path('/var/tmp/typhon-live09-ghost-decisions-20261012')
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
p=read(OUT/'protocol.json');statuses=[read(OUT/(j['id']+'-status.json')) for j in p['jobs']]
assert all(x['completed'] for x in statuses)
assert sum(x['choice_calls'] for x in statuses)==32
inventory=read(OUT/'source-inventory.json');assert sha(OUT/'source-inventory.json')==p['source_inventory_sha256']
for item in inventory:assert sha(Path(item['source']))==item['sha256']
source_audit=OUT/'source-evidence';source_audit.mkdir(exist_ok=True)
for name,(source,candidates) in SETS.items():
 src=SOURCE/source;dst=OUT/name
 shutil.copy2(src/'validation_v2.json',source_audit/(name+'-original-validation.json'))
 shutil.copy2(src/'comparison_v2.json',source_audit/(name+'-original-comparison.json'))
 critic.compare(dst,SOURCE/candidates)
secondary.compare(OUT/'supplement')
summary={'frozen_protocol_sha256':sha(OUT/'protocol.json'),'original_source_files_unchanged':len(inventory),
 'choice_calls':sum(x['choice_calls'] for x in statuses),'reason_calls':sum(x['reason_calls'] for x in statuses),
 'all_jobs_complete':True,'sets':{},'no_robot_actions_or_training':True}
rows=[];requests=[];latencies={'choice':[],'reason':[]}
for name in SETS:
 root=OUT/name;comparison=read(root/'comparison_v2.json')
 for manifest in sorted((root/'slates').glob('*/*/*/manifest.json')):
  m=read(manifest);folder=manifest.parent/'luna';choice=read(folder/'choice.json')
  expected=critic.request_payload(m,root);sent=read(folder/'choice-request.json');assert sent==expected
  raw=read(folder/'choice-raw.json')
  parsed=critic.parse_choice(raw,critic.CHOICES,'best_candidate');assert parsed['choice']==choice['choice']
  assert raw['model']==p['model']
  poisoned=json.loads(json.dumps(m));poisoned['oracle']={'best':'A','truth':[1,2,3]}
  assert critic.request_payload(poisoned,root)==expected
  reason=read(folder/'reason.json')
  expected_reason=critic.request_payload(m,root,choice['choice']);assert read(folder/'reason-request.json')==expected_reason
  reason_raw=read(folder/'reason-raw.json')
  rp=critic.parse_choice(reason_raw,list(critic.REASONS),'visual_reason');assert rp['choice']==reason['choice']
  for kind,d in [('choice',choice),('reason',reason)]:latencies[kind].append(d['wall_seconds'])
  requests.append({'set':name,'manifest':str(manifest.relative_to(root)),'choice_payload_matches_frozen_builder':True,'reason_payload_matches_frozen_builder':True,'typed_identity_and_choice_verified':True})
  row=next(r for r in comparison['rows'] if r['state']==m['state'] and r['mode']==m['mode'] and r['repeat']==m['order_repeat'])
  rows.append({'set':name,'state':m['state'],'mode':m['mode'],'order':m['order_repeat'],'choice':choice['choice'],'confidence':choice['confidence'],
   'branch':row['branch'],'adequate':row['selection_quality']['adequate'] if row['selection_quality'] else None,
   'oracle_best':row['selection_quality']['oracle_best'] if row['selection_quality'] else None,
   'strict_success':row['selected_physical_outcome']['physical_outcome_after24step_settle']['success'] if row['selected_physical_outcome'] else None,
   'reason':reason['choice'],'choice_seconds':choice['wall_seconds'],'reason_seconds':reason['wall_seconds']})
 summary['sets'][name]={'states':len(comparison['states']),'choice_records':comparison['actual_choice_responses'],'by_mode':comparison['by_mode'],
  'selected_labels':dict(collections.Counter(r['choice'] for r in rows if r['set']==name)),
  'reason_codes':dict(collections.Counter(r['reason'] for r in rows if r['set']==name))}
sec=read(OUT/'supplement/secondary_visual_goal_v3/comparison.json')
summary['secondary_visual_goal']=sec
def quant(a,q):
 b=sorted(a);i=(len(b)-1)*q;lo=int(i);hi=min(lo+1,len(b)-1);return b[lo]+(b[hi]-b[lo])*(i-lo)
summary['latency_seconds']={k:{'n':len(v),'median':statistics.median(v),'p95':quant(v,.95)} for k,v in latencies.items()}
summary['limitations']=['Eight inspected archived states, not independent rollout trials; two orders per mode','Both modes use privileged physically simulated lookahead','Operational proxy cannot judge whether images establish enough evidence; insufficiency is not automatically wrong','No live control benefit or deployment conclusion','Initial held supplement cases never count as new pickups','Original strict metric and secondary regional visual goal remain distinct']
save(OUT/'validation_summary.json',summary);save(OUT/'request-audit.json',{'requests':requests,'source_files_unchanged':len(inventory),'all_typed_responses_verified':True})
with (OUT/'decisions.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
save(OUT/'complete.json',{'completed':True,'choice_calls':32,'reason_calls':32,'request_and_source_audit_passed':True})
print(json.dumps({k:summary[k] for k in ['sets','latency_seconds']},indent=2))
print('SECONDARY',json.dumps({'keys':list(sec),'by_mode':sec.get('by_mode')},indent=2))
