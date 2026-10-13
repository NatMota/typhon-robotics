"""Offline dual-endpoint and paired-input/noise analysis. No policy or simulator reruns."""
from load02_common import *
import numpy as np,csv,itertools
def run():
 assert read(OUT/'driver-status.json')['state']=='complete'
 protocol=read(OUT/'protocol.json');rows=[];roots={}
 for m in ['F','FL']:
  for s in protocol['cases']:
   d=OUT/'evaluation'/m/s['source_demo_id']/s['scenario_id'];r=read(d/'episode.json');row=read(d/'dual-scores.json')
   assert not any(r[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors'])
   assert row['scores']==scores(r,s['config']);assert read(d/'reset-isolation.json')['passed']
   a=np.load(d/'input-audit/initial-prepared.npz')
   target=read(OUT/'fixtures'/s['fixture_id']/'isolation.json')['hashes'][s['requested_task']]
   assert {k:array_sha(a[k]) for k in a.files}==target
   rows.append(row);roots[(m,s['source_demo_id'],s['seed'],s['requested_task'])]=d
 pairs=[]
 for m,scene,seed in itertools.product(['F','FL'],['demo_0012','demo_0013'],[912171,912172]):
  f=roots[(m,scene,seed,'feeding')];l=roots[(m,scene,seed,'loading')]
  rf=read(f/'dual-scores.json');rl=read(l/'dual-scores.json')
  nf=sorted((f/'input-audit').glob('noise-*.npz'));nl=sorted((l/'input-audit').glob('noise-*.npz'))
  assert len(nf)==len(nl)==48,(m,scene,seed,len(nf),len(nl))
  for af,al in zip(nf,nl):
   zf=np.load(af);zl=np.load(al)
   assert zf.files==zl.files
   for k in zf.files:np.testing.assert_array_equal(zf[k],zl[k])
  cf=np.load(f/'raw_predictions.npz')['chunks'][0];cl=np.load(l/'raw_predictions.npz')['chunks'][0]
  d=cf[:,:3]-cl[:,:3]
  pair={'model':m,'scene':scene,'seed':seed,'feeding_success':rf['requested_success'],'loading_success':rl['requested_success'],'both_requested_success':rf['requested_success'] and rl['requested_success'],'feeding_requested_but_loaded':rf['other_success'],'loading_requested_but_fed':rl['other_success'],'paired_noise_draws_identical':len(nf),'first_proposal_xyz_rms_difference_mm':float(np.sqrt(np.mean(d*d))*1000),'first_proposal_xyz_max_difference_mm':float(np.linalg.norm(d,axis=1).max()*1000),'first_gripper_mean_abs_difference':float(np.abs(cf[:,9]-cl[:,9]).mean()),'first_proposal_exact_match':bool(np.array_equal(cf,cl))}
  pairs.append(pair)
 summaries=[]
 for m,task in itertools.product(['F','FL'],['feeding','loading']):
  group=[r for r in rows if r['model']==m and r['requested_task']==task]
  summaries.append({'model':m,'instruction':task,'n':len(group),'requested_success':sum(r['requested_success'] for r in group),'wrong_destination_success':sum(r['other_success'] for r in group),'neither':sum(r['neither'] for r in group),'pickups':sum(r['supported_lift_10mm_3steps'] for r in group),'mean_distance_feeding_mm':float(np.mean([r['distances_mm']['feeding'] for r in group])),'mean_distance_loading_mm':float(np.mean([r['distances_mm']['loading'] for r in group]))})
 parity=[]
 for m,scene in itertools.product(['F','FL'],['demo_0012','demo_0013']):
  d=roots[(m,scene,912171,'loading')];old=OLD/'evaluation'/m/'remainder'/('load01-development-'+scene)
  x=np.load(d/'raw_predictions.npz')['chunks'];y=np.load(old/'raw_predictions.npz')['chunks']
  rx=read(d/'episode.json');ry=read(old/'episode.json')
  parity.append({'model':m,'scene':scene,'chunks_byte_identical':bool(np.array_equal(x,y)),'maximum_action_difference':float(np.abs(x-y).max()),'initial_truth_identical':rx['initial_truth']==ry['initial_truth'],'final_truth_identical':rx['final_truth']==ry['final_truth'],'same_strict_loading_outcome':rx['success']==ry['success']})
 validation={'passed':True,'episodes':len(rows),'scoring_reproduced':True,'actual_initial_prepared_inputs_match_fixtures':True,'image_mask_proprio_domain_identical_between_instructions':True,'paired_noise_matches_all_chunks':True,'noise_chunks_per_pair':48,'all_resets_match':True,'runtime_errors':0,'same_loading_condition_load01_parity':parity,'both_endpoints_feasible_from_old_evidence':read(OUT/'feasibility.json')['passed'],'completed':time.time()}
 out=OUT/'analysis';dump(out/'episodes.json',rows);dump(out/'summary.json',summaries);dump(out/'pairs.json',pairs);dump(out/'validation.json',validation)
 for name,items in [('summary',summaries),('pairs',pairs)]:
  with (out/(name+'.csv')).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
 print(json.dumps({'summary':summaries,'pairs':pairs,'parity':parity},indent=2))
if __name__=='__main__':run()
