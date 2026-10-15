"""Execute all sixteen qualified cases; task-only dispatcher unchanged."""
import os,sys,time,json,copy,random,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
import numpy as np
import hier01_task as h
from load04_common import array_sha
O=Path('/var/tmp/typhon-hier01-dev-20261015');h.O=O
read,dump,sha=h.read,h.dump,h.sha
def run():
 p=read(O/'protocol.json');assert read(O/'qualification.json')['passed']
 for f,digest in p['source_sha256'].items():assert sha(R/f)==digest,f
 rows=[];noise_refs={}
 for item in p['execution_order']:
  context,route=item['context'],item['route'];case=copy.deepcopy(p['cases'][context]);seed=case['policy_seed']
  h.limit();model='B' if route=='single' else h.select_checkpoint(case['requested_task'])
  tag=context+'-'+route;modelout=O/'workers'/tag;modelout.mkdir(parents=True);h.base.OUT=modelout
  w=h.base.Worker({'model':p['models'][model],'expected_initial_nontext':case['expected_initial_nontext']})
  dest=modelout/'episode';policy=None
  try:
   random.seed(case['simulation_seed']);policy=h.Policy(w,case,dest,seed,context)
   scenario={'scenario_id':'hier01-dev-'+tag,'source_demo_id':case['source_demo_id'],'group':'inspected-loading-combination-development','task':case['physical_scene_task'],'seed':case['simulation_seed'],'config':copy.deepcopy(case['config'])}
   ep=h.run_episode(scenario,'pretrained',dest,policy,{'experiment':'HIER01-DEV','route':route,'supplied_task':case['requested_task'],'selected_model':model,'model_sha256':p['models'][model]['sha256'],'policy_seed':seed,'dispatcher_uses_only_supplied_task':True,'mid_episode_switch':False,'harness':False,'TTC':False,'new_training':False,'holdout':False,'protocol_sha256':sha(O/'protocol.json')},episode_setup=policy.instrument)
   policy.telemetry.close();assert not any(ep[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors'])
   assert len(policy.chunks)==48 and len(policy.records)==1460
   noise=h.base.noise_gate(dest,seed,noise_refs.get(seed));noise_refs.setdefault(seed,dest)
   with np.load(dest/'input-audit/initial-prepared.npz') as prep:
    assert {k:array_sha(prep[k]) for k in prep.files}==case['input_hashes']
    with np.load(case['source_prepared_archive']) as archive:
     assert prep.files==archive.files
     for k in prep.files:np.testing.assert_array_equal(prep[k],archive[k])
   both=h.base.scores(ep,case['config']);s=both[case['requested_task']];opposite='loading' if case['requested_task']=='feeding' else 'feeding'
   row={'id':tag,'context':context,'family':case['source_demo_id'],'route':route,'model':model,'model_sha256':p['models'][model]['sha256'],'supplied_task':case['requested_task'],'seed':seed,'directory':str(dest),'requested_success':s['success'],'opposite_success':both[opposite]['success'],'outcome':'requested' if s['success'] else 'opposite' if both[opposite]['success'] else 'neither','position_error_mm':s['position_error_m']*1000,'yaw_error_deg':s['orientation_error_deg'],'distances_mm':{k:v['position_error_m']*1000 for k,v in both.items()},'criteria':s['criteria'],'scores':both,'final_xy_m':ep['final_truth']['object_position_m'][:2],'wall_seconds':ep['wall_duration_s'],'worker_load_seconds':w.load_info['load_time_s'],'query_count':len(policy.chunks),'inference_total_seconds':sum(x['inference_s'] for x in policy.inferences),'noise_qualification':noise,**h.supplemental(ep,dest)}
   dump(dest/'dispatch-summary.json',row);rows.append(row);dump(O/'main-summary.json',rows)
   print(json.dumps({k:row[k] for k in ['id','model','requested_success','outcome','position_error_mm','supported_lift_10mm_3steps']}),flush=True)
  finally:
   if policy is not None and hasattr(policy,'telemetry') and not policy.telemetry.closed:policy.telemetry.close()
   w.close()
 assert len(rows)==16
 pairs=[]
 for context in p['cases']:
  a=next(r for r in rows if r['context']==context and r['route']=='single');b=next(r for r in rows if r['context']==context and r['route']=='router')
  pairs.append({'context':context,'B':a['requested_success'],'router':b['requested_success'],'gain':not a['requested_success'] and b['requested_success'],'loss':a['requested_success'] and not b['requested_success']})
 qualifies=all(x['router'] and not x['loss'] for x in pairs)
 failures=sorted((x for x in rows if x['route']=='router' and not x['requested_success']),key=lambda r:(r['family'],r['supplied_task'],r['seed']))
 dump(O/'next-branch.json',{'branch':'HIER01-CHAIN' if qualifies else 'read-only-localization','max_minutes':45 if qualifies else 20,'all_eight_router_pass':qualifies,'gains':sum(x['gain'] for x in pairs),'losses':sum(x['loss'] for x in pairs),'first_router_failure':failures[0]['id'] if failures else None})
 dump(O/'main-complete.json',{'completed':True,'episodes':len(rows),'epoch':time.time(),'new_training':False,'holdouts':False})
 print('ALL_16_COMPLETE',flush=True)
if __name__=='__main__':
 try:run()
 except BaseException:dump(O/'STOPPED-run.json',{'epoch':time.time(),'error':traceback.format_exc()});raise
