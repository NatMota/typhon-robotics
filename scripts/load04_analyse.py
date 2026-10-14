"""Dual-destination scoring and paired input/noise verification; no new rollouts."""
from load04_common import *
import numpy as np,itertools,csv
def run():
 assert read(OUT/'evaluation-status.json')['state']=='complete'
 p=read(OUT/'protocol.json');rows=[];roots={}
 for method in ['C','B']:
  for s in p['cases']:
   vis=s['visibility'];scene=s['source_demo_id'];d=OUT/'evaluation'/method/vis/scene/s['scenario_id']
   r=read(d/'episode.json');row=read(d/'dual-scores.json')
   assert row['scores']==scores(r,s['config'])
   assert not any(r[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors'])
   assert read(d/'reset-isolation.json')['passed'];assert read(d/'visibility.json')['hidden']==(vis=='hidden')
   prepared=np.load(d/'input-audit/initial-prepared.npz');expected=read(OUT/'fixtures'/s['fixture_id']/'isolation.json')['hashes'][s['requested_task']]
   assert {k:array_sha(prepared[k]) for k in prepared.files}==expected
   rows.append(row);roots[(method,vis,scene,s['seed'],s['requested_task'])]=d
 pairs=[]
 for vis,scene,seed,task in itertools.product(['visible','hidden'],['demo_0012','demo_0013'],[912171,912172],['feeding','loading']):
  a=roots[('C',vis,scene,seed,task)];b=roots[('B',vis,scene,seed,task)]
  af=sorted((a/'input-audit').glob('noise-*.npz'));bf=sorted((b/'input-audit').glob('noise-*.npz'))
  assert len(af)==len(bf)==48
  for x,y in zip(af,bf):
   u=np.load(x);v=np.load(y);assert u.files==v.files
   for k in u.files:np.testing.assert_array_equal(u[k],v[k])
  pairs.append({'visibility':vis,'scene':scene,'seed':seed,'instruction':task,'paired_generation_noise_and_rng_exact':True,'chunks':48})
 summaries=[];switches=[]
 for method,vis,task in itertools.product(['C','B'],['visible','hidden'],['feeding','loading']):
  a=[r for r in rows if r['model']==method and r['visibility']==vis and r['requested_task']==task]
  summaries.append({'model':method,'visibility':vis,'instruction':task,'n':len(a),'requested_success':sum(x['requested_success'] for x in a),'wrong_destination_success':sum(x['other_success'] for x in a),'neither':sum(x['neither'] for x in a),'pickups':sum(x['supported_lift_10mm_3steps'] for x in a),'mean_distance_feeding_mm':float(np.mean([x['distances_mm']['feeding'] for x in a])),'mean_distance_loading_mm':float(np.mean([x['distances_mm']['loading'] for x in a]))})
 for method,vis,scene,seed in itertools.product(['C','B'],['visible','hidden'],['demo_0012','demo_0013'],[912171,912172]):
  a=read(roots[(method,vis,scene,seed,'feeding')]/'dual-scores.json');b=read(roots[(method,vis,scene,seed,'loading')]/'dual-scores.json')
  switches.append({'model':method,'visibility':vis,'scene':scene,'seed':seed,'both_requested_success':a['requested_success'] and b['requested_success']})
 totals=[{'model':m,'visibility':v,'both_requested_success':sum(x['both_requested_success'] for x in switches if x['model']==m and x['visibility']==v),'pairs':4} for m,v in itertools.product(['C','B'],['visible','hidden'])]
 for name,value in [('episodes',rows),('summary',summaries),('matched-noise',pairs),('instruction-switch-pairs',switches),('switch-summary',totals)]:dump(OUT/'analysis'/(name+'.json'),value)
 for name,value in [('summary',summaries),('instruction-switch-pairs',switches),('switch-summary',totals)]:
  with (OUT/'analysis'/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(value[0]));w.writeheader();w.writerows(value)
 dump(OUT/'analysis/validation.json',{'passed':True,'episodes':32,'dual_scores_reproduced':True,'initial_inputs_match_frozen_fixtures':True,'actual_paired_noise_and_rng_exact':True,'runtime_errors':0,'time':time.time()})
 print(json.dumps({'summary':summaries,'switch_summary':totals},indent=2))
if __name__=='__main__':run()
