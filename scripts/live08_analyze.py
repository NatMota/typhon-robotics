"""Score every frozen LIVE08 cell, audit identities, and retain visual evidence."""
import os,sys,json,gzip,pickle,math,subprocess,collections,concurrent.futures
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from supervisor.feeding_loading.evaluation import score_episode
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/var/tmp/typhon-live08-choice-timing-20261012')
def read(p):return json.loads(p.read_text())
def packed(p):
 with gzip.open(p,'rb') as f:return pickle.load(f)
def trace(p):
 with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
p=read(OUT/'protocol.json');assert all(read(OUT/f'worker-{i}/complete.json')['completed'] for i in [0,1])
sid=p['scenes'][0]['scenario_id'];rows=[];checks=[];initial=None;first=None;trajectories={}
for route in p['routes']:
 f=OUT/route['id']/sid;d=read(f/'summary.json');assert read(f/'complete.json')['completed']
 score=score_episode(d['initial_truth'],d['final_truth'],d['stability_truths'],p['scenes'][0]['config'],d['evaluation']['attempts'])
 assert score['criteria']==d['evaluation']['criteria'] and score['success']==d['strict_success']
 q=packed(f/'query-0000.pkl.gz')
 if first is None:first=q;initial=d['initial_truth']
 assert d['initial_truth']==initial
 np.testing.assert_array_equal(q['payload']['proprio'],first['payload']['proprio'])
 assert all(np.array_equal(a,b) for a,b in zip(q['payload']['images'],first['payload']['images']))
 np.testing.assert_array_equal(q['response']['initial_noise'],first['response']['initial_noise'])
 np.testing.assert_array_equal(q['response']['actions'],first['response']['actions'])
 events=[json.loads(line) for line in (f/'events.jsonl').read_text().splitlines()]
 admitted=[x for x in events if x['type']=='admitted'];assert admitted[0]['query']==0 and admitted[0]['candidate']==route['initial_candidate']
 executed=trace(f/'execution.jsonl.gz');assert len(executed)==776
 indices=collections.defaultdict(list);queries={}
 for row in executed:
  if row['origin']!='proposal':continue
  ident=row['identity'];index=ident['query']
  if index not in queries:queries[index]=packed(f/f'query-{index:04d}.pkl.gz')['response']['actions']
  np.testing.assert_array_equal(row['target'],queries[index][ident['candidate'],ident['index']])
  indices[(index,ident['candidate'])].append(ident['index'])
 assert all(v==list(range(len(v))) for v in indices.values())
 t=trace(f/'trajectory.jsonl.gz');assert len(t)==800;trajectories[route['id']]=t
 row={'route':route['id'],'first_preference':chr(65+route['initial_candidate']),'boundary_hold_ticks':route['wait_ticks'],
  **{k:d[k] for k in ['pickup','placement_without_yaw','footprint_inside_target','strict_success','remaining_yaw_error_deg','yaw_gap_closed_deg']},
  'centre_error_mm':1000*score['position_error_m'],'grasp_attempts':len(score['attempts']),
  'verified_lifts':sum(a['verified_lift'] for a in score['attempts']),
  'maximum_recorded_lift_mm':1000*max([a['maximum_lift_m'] for a in score['attempts']] or [0]),
  'policy_actions':d['execution_counts'].get('proposal',0),'queries':d['queries'],'wall_s':d['wall_s'],'execution_counts':d['execution_counts'],
  'admitted_candidates':[{'query':x['query'],'candidate':chr(65+x['candidate'])} for x in admitted],
  'boundary_rejections':d['boundary_rejections']}
 rows.append(row);checks.append({'route':route['id'],'same_initial_state_and_all_four_candidates':True,'intended_initial_choice_executed':True,'scores_recomputed':True,'every_target_matches_saved_candidate_and_contiguous_prefix':True})
contrasts=[]
for wait in [40,41]:
 a,b=[trajectories[f'{label}_wait{wait}'] for label in ['A','C']]
 first_different=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
 contrasts.append({'wait_ticks':wait,'exact_control_and_truth_prefix':first_different,'first_difference_at_choice':first_different==wait})
 assert first_different==wait
live=packed(Path('/var/tmp/typhon-live07-live-integration-20261012/first_admissible')/sid/'proposal-0000.pkl.gz')
live_match=all(np.array_equal(a,b) for a,b in zip(first['payload']['images'],live['payload']['images'])) and np.array_equal(first['response']['initial_noise'],live['response']['initial_noise']) and np.array_equal(first['response']['actions'],live['response']['actions'])
videos=list(OUT.glob('*/*/rollout.mp4'))
def decode(f):
 r=subprocess.run(['ffmpeg','-v','error','-i',str(f),'-f','null','-'],capture_output=True,text=True)
 return {'path':str(f.relative_to(OUT)),'passed':r.returncode==0,'error':r.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:qa=list(pool.map(decode,videos))
assert len(qa)==4 and all(q['passed'] for q in qa)
summary={'all_four_frozen_cells_complete':True,'results':rows,'within_wait_contrasts':contrasts,'initial_RGB_noise_and_four_proposals_match_LIVE07':bool(live_match),
 'api_calls':0,'no_training':True,'limits':p['limits']+['Fixed40/41 tick experiment differs from variable wall scheduler; no claim of exact full LIVE07 reproduction']}
save(OUT/'validation_summary.json',summary)
import csv
with (OUT/'comparisons.csv').open('w') as f:
 keys=['route','first_preference','boundary_hold_ticks','pickup','placement_without_yaw','footprint_inside_target','strict_success','centre_error_mm','remaining_yaw_error_deg','maximum_recorded_lift_mm','policy_actions']
 w=csv.DictWriter(f,keys,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
v=OUT/'visual-review';v.mkdir(exist_ok=True);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
readers=[imageio.get_reader(OUT/r['id']/sid/'rollout.mp4') for r in p['routes']]
writer=imageio.get_writer(v/'choice-timing-comparison.mp4',fps=5,codec='libx264',quality=8);count=0
try:
 for frames in zip(*readers):
  panel=Image.new('RGB',(1280,768),'#111722');draw=ImageDraw.Draw(panel)
  for i,(frame,row) in enumerate(zip(frames,rows)):
   x=(i%2)*640;y=(i//2)*384
   draw.text((x+8,y+4),f"First {row['first_preference']}, fixed {row['boundary_hold_ticks']} hold ticks | V4 unchanged",font=font,fill='white')
   draw.text((x+8,y+26),f"Pickup {row['pickup']} | place {row['placement_without_yaw']} | yaw {row['remaining_yaw_error_deg']:.1f} deg",font=font,fill='#acd3e2')
   panel.paste(Image.fromarray(frame).resize((640,320)),(x,y+52))
  writer.append_data(np.asarray(panel));count+=1
 panel.save(v/'choice-timing-final.jpg',quality=93)
finally:
 writer.close()
 for r in readers:r.close()
qa.append(decode(v/'choice-timing-comparison.mp4'));assert qa[-1]['passed']
save(OUT/'artifact-validation.json',{'episodes':checks,'videos':qa,'comparison_frames':count})
save(OUT/'complete.json',{'completed':True,'episodes':4,'artifact_validation_passed':True})
print(json.dumps(summary,indent=2))
