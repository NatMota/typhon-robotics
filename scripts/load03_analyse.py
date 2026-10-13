"""LOAD03 paired visual intervention analysis; no simulation or inference."""
from load03_common import *
import numpy as np,itertools,csv
def run():
 assert read(OUT/'driver-status.json')['state']=='complete'
 p=read(OUT/'protocol.json');rows=[];roots={}
 for s in p['cases']:
  vis=s['visibility'];scene=s['source_demo_id'];d=OUT/'evaluation'/vis/scene/s['scenario_id']
  r=read(d/'episode.json');row=read(d/'dual-scores.json')
  assert row['scores']==scores(r,s['config'])
  assert not any(r[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors'])
  assert read(d/'reset-isolation.json')['passed'];assert read(d/'visibility.json')['hidden']==(vis=='hidden')
  prepared=np.load(d/'input-audit/initial-prepared.npz')
  expected=read(OUT/'fixtures'/s['fixture_id']/'isolation.json')['hashes'][s['requested_task']]
  assert {k:array_sha(prepared[k]) for k in prepared.files}==expected
  rows.append(row);roots[(vis,scene,s['seed'],s['requested_task'])]=d
 pairs=[]
 for scene,seed in itertools.product(['demo_0012','demo_0013'],[912171,912172]):
  for task in ['feeding','loading']:
   a=roots[('visible',scene,seed,task)];b=roots[('hidden',scene,seed,task)]
   af=sorted((a/'input-audit').glob('noise-*.npz'));bf=sorted((b/'input-audit').glob('noise-*.npz'))
   assert len(af)==len(bf)==48
   for x,y in zip(af,bf):
    u=np.load(x);v=np.load(y);assert u.files==v.files
    for k in u.files:np.testing.assert_array_equal(u[k],v[k])
   ca=np.load(a/'raw_predictions.npz')['chunks'][0];cb=np.load(b/'raw_predictions.npz')['chunks'][0]
   ra=read(a/'dual-scores.json');rb=read(b/'dual-scores.json')
   pairs.append({'scene':scene,'seed':seed,'instruction':task,'visible_requested_success':ra['requested_success'],'hidden_requested_success':rb['requested_success'],'visible_feeding_success':ra['scores']['feeding']['success'],'hidden_feeding_success':rb['scores']['feeding']['success'],'visible_loading_success':ra['scores']['loading']['success'],'hidden_loading_success':rb['scores']['loading']['success'],'first_xyz_rms_difference_mm':float(np.sqrt(np.mean((ca[:,:3]-cb[:,:3])**2))*1000),'first_xyz_max_difference_mm':float(np.linalg.norm(ca[:,:3]-cb[:,:3],axis=1).max()*1000),'paired_actual_noise_exact':True})
 summaries=[];switches=[]
 for vis,task in itertools.product(['visible','hidden'],['feeding','loading']):
  a=[r for r in rows if r['visibility']==vis and r['requested_task']==task]
  summaries.append({'visibility':vis,'instruction':task,'n':len(a),'requested_success':sum(x['requested_success'] for x in a),'wrong_destination_success':sum(x['other_success'] for x in a),'neither':sum(x['neither'] for x in a),'feeding_success':sum(x['scores']['feeding']['success'] for x in a),'loading_success':sum(x['scores']['loading']['success'] for x in a),'pickups':sum(x['supported_lift_10mm_3steps'] for x in a),'mean_distance_feeding_mm':float(np.mean([x['distances_mm']['feeding'] for x in a])),'mean_distance_loading_mm':float(np.mean([x['distances_mm']['loading'] for x in a]))})
 for vis,scene,seed in itertools.product(['visible','hidden'],['demo_0012','demo_0013'],[912171,912172]):
  a=read(roots[(vis,scene,seed,'feeding')]/'dual-scores.json');b=read(roots[(vis,scene,seed,'loading')]/'dual-scores.json')
  switches.append({'visibility':vis,'scene':scene,'seed':seed,'both_requested_success':a['requested_success'] and b['requested_success']})
 parity=[]
 for scene,seed,task in itertools.product(['demo_0012','demo_0013'],[912171,912172],['feeding','loading']):
  new=roots[('visible',scene,seed,task)];old=Path('/var/tmp/typhon-load02-20261013/evaluation/FL')/scene/('load02-'+scene+'-seed'+str(seed)+'-'+task)
  a=np.load(new/'raw_predictions.npz')['chunks'];b=np.load(old/'raw_predictions.npz')['chunks']
  rn=read(new/'episode.json');ro=read(old/'episode.json')
  parity.append({'scene':scene,'seed':seed,'instruction':task,'chunks_exact':bool(np.array_equal(a,b)),'initial_truth_exact':rn['initial_truth']==ro['initial_truth'],'final_truth_exact':rn['final_truth']==ro['final_truth'],'same_loading_outcome':rn['success']==ro['success'],'max_action_difference':float(np.abs(a-b).max())})
 out=OUT/'analysis'
 for name,value in [('episodes',rows),('summary',summaries),('visual-pairs',pairs),('instruction-switch-pairs',switches),('native-repeat-parity',parity)]:dump(out/(name+'.json'),value)
 for name,value in [('summary',summaries),('visual-pairs',pairs),('instruction-switch-pairs',switches)]:
  with (out/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(value[0]));w.writeheader();w.writerows(value)
 dump(out/'validation.json',{'passed':True,'episodes':len(rows),'all_scores_reproduced':True,'actual_initial_inputs_exactly_match_qualified_fixtures':True,'actual_noise_and_rng_match_all_visual_pairs':True,'noise_chunks_per_pair':48,'runtime_errors':0,'render_physics_and_crop_qualification':read(OUT/'render-qualification.json')['passed'],'native_exact_repeats':sum(x['chunks_exact'] and x['final_truth_exact'] for x in parity),'native_repeat_count':len(parity),'time':time.time()})
 print(json.dumps({'summary':summaries,'switches':switches,'native_parity':parity},indent=2))
if __name__=='__main__':run()
