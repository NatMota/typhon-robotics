"""Summarize all frozen LIVE07 routes and verify executed candidate identities."""
import argparse,collections,csv,gzip,hashlib,json,pickle
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def quant(values):return dict(zip(['p50','p95'],[float(x) for x in np.quantile(values,[.5,.95])])) if values else None
def analyze(root):
 protocol=read(root/'protocol.json');assert read(root/'complete.json')['episodes']==9
 rows=[];checks=[];ranks=[];generation=[];preview=[];api=[];activation=[];unknown=[];statehashes={}
 for scene in protocol['scenes']:
  sid=scene['scenario_id'];statehashes[sid]=[]
  for mode in protocol['routes']:
   folder=root/mode/sid
   if mode=='native_reference':
    m=read(folder/'metrics.json');ep=read(folder/'episode.json')
    row={k:m[k] for k in ['scenario_id','pickup','placement_without_yaw','initial_yaw_gap_deg','remaining_yaw_error_deg','yaw_gap_closed_deg']}
    row.update(mode=mode,strict_success=m['success'],actions_executed=m['policy_actions'],guard_holds=0,api_selected_chunks=0,activated_chunks=len(ep['inference_measurements']),wall_s=m['wall_seconds'])
    statehashes[sid].append(hashlib.sha256(json.dumps(ep['initial_truth'],sort_keys=True,separators=(',',':')).encode()).hexdigest());assert not m['runtime_errors']
    row.update(square_footprint_inside=ep['evaluation']['criteria']['footprint_inside_target'],position_tolerance_met=ep['evaluation']['criteria']['position'])
   else:
    m=read(folder/'summary.json');statehashes[sid].append(hashlib.sha256(json.dumps(m['initial_truth'],sort_keys=True,separators=(',',':')).encode()).hexdigest())
    row={k:m[k] for k in ['mode','scenario_id','strict_success','pickup','placement_without_yaw','initial_yaw_gap_deg','remaining_yaw_error_deg','yaw_gap_closed_deg','guard_holds','api_selected_chunks','activated_chunks','wall_s']}
    row.update(square_footprint_inside=m['evaluation']['criteria']['footprint_inside_target'],position_tolerance_met=m['evaluation']['criteria']['position'])
    e=lines(folder/'events.jsonl');re=lines(folder/'ranking/events.jsonl')
    with gzip.open(folder/'execution.jsonl.gz','rt') as f:ex=[json.loads(s) for s in f]
    assert len(ex)==protocol['steps_after_reset']
    by_id=collections.defaultdict(list)
    for x in ex:
     if x['origin']=='proposal':
      ident=x['identity'];by_id[(ident['proposal'],ident['candidate'])].append(x)
    for (index,label),executed in by_id.items():
     with gzip.open(folder/f'proposal-{index:04d}.pkl.gz','rb') as f:packet=pickle.load(f)
     expected=packet['response']['actions'][ord(label)-65]
     assert [x['identity']['next_index'] for x in executed]==list(range(len(executed)))
     np.testing.assert_array_equal(np.asarray([x['target'] for x in executed]),expected[:len(executed)])
    activations=[x for x in e if x['event']=='activated']
    assert all(x['start_index']==0 and x['remaining_steps']==30 for x in activations)
    requested={x['index']:x for x in e if x['event']=='requested'}
    for x in activations:
     activation.append(x['wall_s']-requested[x['proposal']]['wall_s'])
     assert 1.9<x['wall_s']-requested[x['proposal']]['wall_s']
    starts=sum(len(v)>0 for v in by_id.values());complete=sum(len(v)==30 for v in by_id.values())
    row.update(actions_executed=sum(len(v) for v in by_id.values()),boundary_hold_steps=sum(x['origin']=='boundary_hold' for x in ex),fully_executed_chunks=complete,
     started_chunks=starts,handoff_rejections=dict(collections.Counter(x['reason'] for x in e if x['event']=='handoff_rejected')),
     api_requests=sum(x['type']=='submitted' for x in re),rank_deadlines=sum(x['type']=='discarded' and x.get('reason')=='deadline' for x in re))
    accepted_indices={x['index'] for x in re if x['type']=='selected'}
    for x in activations:
     if x['reason']=='decisions_api':
      # Selection log identifies the API request; local scheduler also links the
      # proposal via its rank_reply event. Both identities must be retained.
      reply=next(t for t in e if t['event']=='rank_reply' and t['index']==x['proposal'])
      assert reply['candidate']==x['candidate']
    for x in e:
     if x['event']=='proposals':generation.append(x['generation_s'])
    for p in (folder/'ranking').glob('*/result.json'):
     r=read(p);ranks.append(r)
     if 'api_seconds' in r:api.append(r['api_seconds'])
     if 'preview_seconds' in r:preview.append(r['preview_seconds'])
    checks.append({'mode':mode,'scenario':sid,'every_executed_target_equals_saved_candidate':True,'no_skipped_or_duplicate_prefix':True,'started_chunks':starts,'full_chunks':complete})
   rows.append(row)
 assert all(len(set(v))==1 for v in statehashes.values()),statehashes
 grouped={}
 for mode in protocol['routes']:
  rr=[r for r in rows if r['mode']==mode]
  grouped[mode]={k:sum(bool(r[k]) for r in rr) for k in ['strict_success','pickup','placement_without_yaw']}
  grouped[mode].update(episodes=len(rr),manipulation_episodes=sum(r['scenario_id']!='film-demo_0018' for r in rr),
   manipulation_strict_success=sum(r['strict_success'] for r in rr if r['scenario_id']!='film-demo_0018'),
   completed_preserved=sum(r['strict_success'] for r in rr if r['scenario_id']=='film-demo_0018'),
   api_selected_chunks=sum(r['api_selected_chunks'] for r in rr),actions_executed=sum(r['actions_executed'] for r in rr))
 summary={'all_frozen_episodes_evaluated':len(rows)==9,'rows':rows,'by_route':grouped,'latency_seconds':{'candidate_generation':quant(generation),'preview':quant(preview),'api_transport':quant(api),'observation_to_activation':quant(activation)},
  'api_results':dict(collections.Counter((r['status']+':'+r['choice']) for r in ranks)),'same_initial_truth_each_route':True,
  'execution_identity_checks':checks,'not_confirmatory':True,'no_new_training':True}
 (root/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 cols=['mode','scenario_id','strict_success','pickup','placement_without_yaw','square_footprint_inside','position_tolerance_met','initial_yaw_gap_deg','remaining_yaw_error_deg','yaw_gap_closed_deg','actions_executed','api_selected_chunks','guard_holds','wall_s']
 with (root/'comparisons.csv').open('w') as f:
  w=csv.DictWriter(f,cols,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
 extras=[];identities=[]
 for mode in ['first_admissible','decisions']:
  for scene in protocol['scenes']:
   folder=root/mode/scene['scenario_id']
   e=lines(folder/'events.jsonl');ranking=lines(folder/'ranking/events.jsonl')
   with gzip.open(folder/'execution.jsonl.gz','rt') as f:trace=[json.loads(line) for line in f]
   changes=[x for x in trace if x['raw_guard_reason']=='proposed_target_jump' and x['execution_guard_reason'] is None]
   rejected=[x for x in trace if x['execution_guard_reason']]
   extras.append({'mode':mode,'scene':scene['scenario_id'],'old_jump_veto_avoided':len(changes),
    'proposed_jump_veto_avoided':sum(x['requested_origin']=='proposal' for x in changes),
    'execution_reasons':dict(collections.Counter(x['execution_guard_reason'] for x in rejected)),
    'handoff_reasons':dict(collections.Counter(x['reason'] for x in e if x['event']=='handoff_rejected')),
    'rank_result_status':dict(collections.Counter(x['type'] for x in ranking)),
    'late_results':sum(x['type']=='discarded' and x.get('reason')=='deadline' for x in ranking)})
   # API selection, activation, and execution are distinct outcomes.
   submissions={x['index']:x for x in ranking if x['type']=='submitted'}
   for selected in [x for x in ranking if x['type']=='selected']:
    submitted=submissions[selected['index']];index=int(submitted['context'].rsplit(':',1)[1]);label=selected['candidate']
    assert selected['context']==submitted['context'] and selected['elapsed_steps']==0
    with gzip.open(folder/f'proposal-{index:04d}.pkl.gz','rb') as f:packet=pickle.load(f)
    for k,h in submitted['candidate_sha256'].items():
     a=np.asarray(packet['response']['actions'][ord(k)-65],float).copy()
     assert hashlib.sha256(a.tobytes()).hexdigest()==h
    source=next(x for x in e if x['event']=='requested' and x['index']==index)
    for j,name in enumerate(['primary','secondary']):
     assert hashlib.sha256(packet['payload']['images'][j].tobytes()).hexdigest()==source['frame_sha256'][name]
    raw=read(folder/'ranking'/f"{selected['index']:05d}"/'response.json')
    assert raw['answers'][0]['choice']==label
    activations=[x for x in e if x['event']=='activated' and x['reason']=='decisions_api' and x['proposal']==index]
    assert len(activations)<=1
    executed=[x for x in trace if x['origin']=='proposal' and x['identity']['proposal']==index and x['identity']['candidate']==label]
    if activations:
     assert activations[0]['candidate']==label and activations[0]['start_index']==0
     np.testing.assert_array_equal(np.asarray([x['target'] for x in executed]).reshape(-1,20),packet['response']['actions'][ord(label)-65][:len(executed)])
    else:assert not executed
    identities.append({'scene':scene['scenario_id'],'api_request':selected['index'],'proposal':index,'candidate':label,
     'source_images_and_candidate_hashes_match':True,'api_choice_matches':True,'activated':bool(activations),
     'executed_targets':len(executed),'fully_executed':len(executed)==30,
     'handoff_rejections':[x for x in e if x['event']=='handoff_rejected' and x['proposal']==index and x['candidate']==label]})
 for row in rows:
  if row['mode']=='decisions':
   choices=[x for x in identities if x['scene']==row['scenario_id']]
   row['api_actually_executed_chunks']=sum(x['executed_targets']>0 for x in choices)
   row['api_selected_but_zero_actions']=sum(x['activated'] and not x['executed_targets'] for x in choices)
  else:row['api_actually_executed_chunks']=0;row['api_selected_but_zero_actions']=0
 for mode,group in grouped.items():
  group['api_actually_executed_chunks']=sum(r['api_actually_executed_chunks'] for r in rows if r['mode']==mode)
  group['api_selected_but_zero_actions']=sum(r['api_selected_but_zero_actions'] for r in rows if r['mode']==mode)
 summary['activation_count_caveat']='Legacy api_selected_chunks counts activated queues, including queues cleared during the final stability hold. api_actually_executed_chunks requires at least one executed proposal target.'
 summary['execution_contract_audit']=extras;summary['api_identity_audit']=identities
 summary['guard_integration_status']='Qualified simulated execution helper; production wrappers updated, complete perception/completion deployment not qualified.'
 summary['limitations']=protocol['limits']+[protocol['preview_limit'],protocol['execution_environment']]
 (root/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 (root/'api-identity-audit.json').write_text(json.dumps({'passed':True,'choices':identities},indent=2)+'\n')
 (root/'guard-integration-audit.json').write_text(json.dumps(extras,indent=2)+'\n')
 print(json.dumps({k:summary[k] for k in ['by_route','latency_seconds','api_results','same_initial_truth_each_route']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);analyze(p.parse_args().root)
