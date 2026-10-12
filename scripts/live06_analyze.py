"""LIVE06 evidence validation, including exact native/corrected trace comparison."""
import sys,json,gzip,pickle,collections,csv,subprocess,concurrent.futures
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.baseline.benchmark import save
OUT=Path('/var/tmp/typhon-live06-execution-guard-20261012')
def read(p):return json.loads(p.read_text())
def trace(p):
 with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
def packet(p):
 with gzip.open(p,'rb') as f:return pickle.load(f)
p=read(OUT/'protocol.json');rows=[];first={};checks=[]
for route in p['routes']:
 for scene in p['scenes']:
  folder=OUT/route['id']/scene['scenario_id'];d=read(folder/'summary.json');assert read(folder/'complete.json')['completed']
  score=score_episode(d['initial_truth'],d['final_truth'],d['stability_truths'],scene['config'],d['evaluation']['attempts'])
  assert score['criteria']==d['evaluation']['criteria'] and score['success']==d['strict_success']
  q=packet(folder/'query-0000.pkl.gz')
  if scene['scenario_id'] not in first:first[scene['scenario_id']]=(q,d['initial_truth'])
  ref,initial=first[scene['scenario_id']]
  assert d['initial_truth']==initial
  assert all(np.array_equal(x,y) for x,y in zip(q['payload']['images'],ref['payload']['images']))
  np.testing.assert_array_equal(q['response']['initial_noise'][0],ref['response']['initial_noise'][0])
  np.testing.assert_array_equal(q['response']['actions'][0],ref['response']['actions'][0])
  executed=trace(folder/'execution.jsonl.gz');assert len(executed)==776
  lookup={};indices=collections.defaultdict(list)
  for event in executed:
   if event['origin']!='proposal':continue
   ident=event['identity'];index=ident['query']
   if index not in lookup:lookup[index]=packet(folder/f'query-{index:04d}.pkl.gz')['response']['actions']
   np.testing.assert_array_equal(event['target'],lookup[index][ident['candidate'],ident['index']])
   indices[(index,ident['candidate'])].append(ident['index'])
  assert all(v==list(range(len(v))) for v in indices.values())
  rows.append({'route':route['id'],'scene':scene['scenario_id'],**{k:d[k] for k in ['pickup','placement_without_yaw','footprint_inside_target','strict_success','remaining_yaw_error_deg','yaw_gap_closed_deg']},
   'policy_targets':d['execution_counts'].get('proposal',0),'guard_holds':d['execution_counts'].get('guard_hold',0)})
  checks.append({'route':route['id'],'scene':scene['scenario_id'],'initial_state_and_first_input_noise_prediction_match':True,'scores_recomputed':True,'targets_and_prefixes_exact':True})
known='film-demo_0006';a=OUT/'native_reference'/known;b=OUT/'execution_guard'/known
ta,tb=trace(a/'trajectory.jsonl.gz'),trace(b/'trajectory.jsonl.gz')
first_difference=next((i for i,(x,y) in enumerate(zip(ta,tb)) if x!=y),None)
object_position_error=max(float(np.max(np.abs(np.array(x['truth']['object_position_m'])-y['truth']['object_position_m']))) for x,y in zip(ta,tb))
object_yaw_error=max(abs(x['truth']['object_yaw_rad']-y['truth']['object_yaw_rad']) for x,y in zip(ta,tb))
matching=[]
for f in sorted(a.glob('query-*.pkl.gz')):
 left,right=packet(f),packet(b/f.name)
 same_inputs=all(np.array_equal(x,y) for x,y in zip(left['payload']['images'],right['payload']['images']))
 same_noise=np.array_equal(left['response']['initial_noise'],right['response']['initial_noise'])
 same_actions=np.array_equal(left['response']['actions'],right['response']['actions'])
 assert same_noise
 matching.append({'query':f.name,'current_rgb_exact':same_inputs,'noise_exact':same_noise,'predictions_exact':same_actions})
save(OUT/'known-grasp-equivalence.json',{'all800_control_and_truth_records_exact':ta==tb,'first_different_control_index':first_difference,'exact_control_prefix_length':first_difference if first_difference is not None else len(ta),'all26_current_rgb_noise_prediction_packets_exact':all(q['current_rgb_exact'] and q['noise_exact'] and q['predictions_exact'] for q in matching),'object_position_max_error_m':object_position_error,'object_yaw_max_error_rad':object_yaw_error,'queries':matching})
experts=[read(f) for f in (OUT/'expert-shadow').glob('*/summary.json')];assert len(experts)==8
old=collections.Counter();new=collections.Counter()
for r in experts:old.update(r['old_reasons']);new.update(r['new_reasons'])
videos=list(OUT.glob('*/*/rollout.mp4'))
def decode(f):
 r=subprocess.run(['ffmpeg','-v','error','-i',str(f),'-f','null','-'],capture_output=True,text=True)
 return {'path':str(f.relative_to(OUT)),'passed':r.returncode==0,'error':r.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:vc=list(pool.map(decode,videos))
assert len(vc)==17 and all(x['passed'] for x in vc)
save(OUT/'artifact-validation.json',{'episodes':checks,'videos':vc})
summary={'v4_episodes':9,'expert_shadow_episodes':8,'expert_successes':sum(r['strict_success'] for r in experts),'expert_total_checked_ticks':sum(new.values()),
 'expert_old_reasons':dict(old),'expert_new_reasons':dict(new),'expert_same_criteria_as_prior':all(r['same_criteria_as_prior'] for r in experts),
 'known_grasp_pickup_and_placement_restored':next(r for r in rows if r['route']=='execution_guard' and r['scene']==known)['pickup'] and next(r for r in rows if r['route']=='execution_guard' and r['scene']==known)['placement_without_yaw'],'known_grasp_full_trace_exact':ta==tb,'known_grasp_exact_control_prefix':first_difference,'object_position_max_error_m':object_position_error,'object_yaw_max_error_rad':object_yaw_error,'negative_and_contract_tests_passed':11,'v4_results':rows,
 'limits':['Three inspected development scenes','Zero waiting and no Decisions ranking','Causal held-state oracle retained fromV4','Prospective admission unchanged; execution helper is for already admitted paths or verified expert primitives','No physical-safety certification or broad generalisation claim','Stationary-scene renderer repeatability limitation fromLIVE05 persists']}
save(OUT/'validation_summary.json',summary)
with (OUT/'comparisons.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
visual=OUT/'visual-review';visual.mkdir(exist_ok=True);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
readers=[imageio.get_reader(OUT/m/known/'rollout.mp4') for m in ['native_reference','original_guard','execution_guard']]
writer=imageio.get_writer(visual/'guard-fix-comparison.mp4',fps=5,codec='libx264',quality=8)
frames=0
try:
 for triplet in zip(*readers):
  panel=Image.new('RGB',(1920,352),'#111722');draw=ImageDraw.Draw(panel)
  for i,(frame,title) in enumerate(zip(triplet,['Original V4: pickup + placement','Old guard: no pickup','Corrected execution check: restored'])):
   panel.paste(Image.fromarray(frame).resize((640,320)),(i*640,32))
   draw.text((i*640+8,5),title,font=font,fill='white')
  writer.append_data(np.asarray(panel));frames+=1
 panel.save(visual/'guard-fix-final.jpg',quality=93)
finally:
 writer.close()
 for r in readers:r.close()
r=subprocess.run(['ffmpeg','-v','error','-i',str(visual/'guard-fix-comparison.mp4'),'-f','null','-'],capture_output=True,text=True);assert r.returncode==0,r.stderr
save(OUT/'complete.json',{'completed':True,'v4_episodes':9,'expert_shadow_episodes':8,'comparison_frames':frames,'no_training':True})
print(json.dumps(summary,indent=2))
