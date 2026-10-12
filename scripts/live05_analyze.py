"""Verify immutable LIVE05 records and produce comparison tables."""
import sys,json,gzip,pickle,csv,math,collections,concurrent.futures,subprocess,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.baseline.benchmark import save
OUT=Path('/var/tmp/typhon-live05-isolation-20261012-v2')
OLD=Path('/var/tmp/typhon-live04-boundary-20261012-v3')
def read(p):return json.loads(p.read_text())
def load(p):
 with gzip.open(p,'rb') as f:return pickle.load(f)
def lines(p):
 with gzip.open(p,'rt') as f:return [json.loads(x) for x in f]
protocol=read(OUT/'protocol.json');rows=[];validation=[];first={};full=[]
for route in protocol['routes']:
 for scene in protocol['scenes']:
  path=OUT/route['id']/scene['scenario_id'];d=read(path/'summary.json');assert read(path/'complete.json')['completed']
  score=score_episode(d['initial_truth'],d['final_truth'],d['stability_truths'],scene['config'],d['evaluation']['attempts'])
  assert score['criteria']==d['evaluation']['criteria'] and score['success']==d['strict_success']
  initial=read(OLD/'native_reference'/scene['scenario_id']/'episode.json')['initial_truth']
  assert d['initial_truth']==initial
  trace=lines(path/'trajectory.jsonl.gz');executed=lines(path/'execution.jsonl.gz');assert len(trace)==800 and len(executed)==776
  lookup={}
  for e in executed:
   if e['origin']!='proposal':continue
   ident=e['identity'];n=ident['query']
   if n not in lookup:lookup[n]=load(path/f'query-{n:04d}.pkl.gz')['response']['actions']
   np.testing.assert_array_equal(e['target'],lookup[n][ident['candidate'],ident['index']])
  indices=collections.defaultdict(list)
  for e in executed:
   if e['origin']=='proposal':indices[(e['identity']['query'],e['identity']['candidate'])].append(e['identity']['index'])
  assert all(v==list(range(len(v))) for v in indices.values())
  q=load(path/'query-0000.pkl.gz')
  if scene['scenario_id'] not in first:first[scene['scenario_id']]=q
  ref=first[scene['scenario_id']]
  np.testing.assert_array_equal(q['response']['initial_noise'][0],ref['response']['initial_noise'][0])
  np.testing.assert_array_equal(q['response']['actions'][0],ref['response']['actions'][0])
  assert all(np.array_equal(a,b) for a,b in zip(q['payload']['images'],ref['payload']['images']))
  valid={'route':route['id'],'scenario':scene['scenario_id'],'score_recomputed':True,'initial_state_matched':True,'steps':len(trace),
   'executed_targets_exact':True,'no_skipped_prefix':True,'first_input_noise_prediction_matched':True,'reference_reproduction':d.get('reference_reproduction')}
  validation.append(valid)
  row={'route':route['id'],'scene':scene['scenario_id'],'noise':route['noise'],'wait_s':route['wait_ticks']/20,'guard':route['guard'],'candidates':route['candidates'],
   **{k:d[k] for k in ['pickup','placement_without_yaw','footprint_inside_target','strict_success','initial_yaw_gap_deg','remaining_yaw_error_deg','yaw_gap_closed_deg','queries','wall_s']},
   'policy_actions':d['execution_counts'].get('proposal',0),'boundary_hold_ticks':d['execution_counts'].get('boundary_hold',0),
   'guard_hold_ticks':d['execution_counts'].get('guard_hold',0),'evidence_directory':str(path)}
  rows.append(row);full.append(d)
with (OUT/'comparisons.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
groups=[]
for route in protocol['routes']:
 rr=[r for r in rows if r['route']==route['id'] and r['scene']!='film-demo_0018']
 completed=next(r for r in rows if r['route']==route['id'] and r['scene']=='film-demo_0018')
 groups.append({'route':route['id'],'manipulation_trials':len(rr),'pickups':sum(r['pickup'] for r in rr),'placements':sum(r['placement_without_yaw'] for r in rr),'strict':sum(r['strict_success'] for r in rr),'completed_preserved':completed['strict_success']})
videos=list(OUT.glob('*/*/rollout.mp4'))
def check(p):
 r=subprocess.run(['ffmpeg','-v','error','-i',str(p),'-f','null','-'],capture_output=True,text=True)
 return {'path':str(p.relative_to(OUT)),'passed':r.returncode==0,'stderr':r.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:video_checks=list(pool.map(check,videos))
assert len(videos)==30 and all(v['passed'] for v in video_checks)
save(OUT/'artifact-validation.json',{'episodes':validation,'videos':video_checks})
summary={'episodes':len(rows),'planned':protocol['episodes'],'model':'unchanged original V4-M6','all_completed':len(rows)==protocol['episodes'],
 'groups':groups,'known_grasp_scene':[r for r in rows if r['scene']=='film-demo_0006'],
 'validation':{'initial_states_and_first_inputs_noise_predictions_matched':True,'scores_recomputed':True,'execution_targets_verified':True,'videos_decode':True,
 'known_grasp_exact_reference':next(r['reference_reproduction'] for r in validation if r['route']=='c0_shadow' and r['scenario']=='film-demo_0006')['passed'],
 'full_state_reference_not_exact_on_stationary_scene':next(r['reference_reproduction'] for r in validation if r['route']=='c0_shadow' and r['scenario']=='film-demo_0002')['passed'] is False},
 'limits':protocol['limits'],'qualification_amendment':read(OUT/'qualification-amendment.json')}
save(OUT/'validation_summary.json',summary)
# Contact sheet from saved final camera frames, one tile per arm on the known-grasp scene.
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
tiles=[]
for route in protocol['routes']:
 row=next(r for r in rows if r['route']==route['id'] and r['scene']=='film-demo_0006')
 p=OUT/route['id']/'film-demo_0006';im=Image.open(p/'final.png').convert('RGB');im.thumbnail((640,320))
 tile=Image.new('RGB',(640,372),'#121821');tile.paste(im,(0,52));d=ImageDraw.Draw(tile)
 d.text((8,4),route['id'],font=font,fill='white')
 d.text((8,26),f"pickup={row['pickup']} place={row['placement_without_yaw']} yaw gap={row['remaining_yaw_error_deg']:.1f} deg",font=font,fill='#c6d2e0')
 tiles.append(tile)
panel=Image.new('RGB',(1280,372*5),'#121821')
for i,im in enumerate(tiles):panel.paste(im,((i%2)*640,(i//2)*372))
panel.save(OUT/'known-grasp-contact-sheet.jpg',quality=92)
save(OUT/'complete.json',{'completed':True,'episodes':30,'no_training':True,'preview_audit_requests':read(OUT/'preview-audit/summary.json')['requests']})
print(json.dumps(summary,indent=2))
