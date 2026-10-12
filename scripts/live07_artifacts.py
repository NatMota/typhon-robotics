"""Validate all nine LIVE07 artifacts, then export a descriptive comparison."""
import json,gzip,pickle,hashlib,subprocess,concurrent.futures,sys
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from supervisor.feeding_loading.evaluation import score_episode
from live07_analyze import analyze
from supervisor.adaptation.m7_timing import reconstruct_20hz
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/var/tmp/typhon-live07-live-integration-20261012')
def read(p):return json.loads(p.read_text())
def packed(p):
 with gzip.open(p,'rb') as f:return pickle.load(f)
def trace(p):
 with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
protocol=read(OUT/'protocol.json')
assert all(read(OUT/f'worker-{i}/complete.json')['completed'] for i in [0,1])
checks=[];first={}
for scene in protocol['scenes']:
 sid=scene['scenario_id']
 for mode in protocol['routes']:
  folder=OUT/mode/sid
  d=read(folder/('episode.json' if mode=='native_reference' else 'summary.json'))
  truth=trace(folder/'trajectory.jsonl.gz');assert len(truth)==800
  q=packed(folder/('query-0000.pkl.gz' if mode=='native_reference' else 'proposal-0000.pkl.gz'))
  if sid not in first:first[sid]=q
  ref=first[sid]
  np.testing.assert_array_equal(q['payload']['proprio'],ref['payload']['proprio'])
  assert all(np.array_equal(a,b) for a,b in zip(q['payload']['images'],ref['payload']['images']))
  np.testing.assert_array_equal(q['response']['initial_noise'][0],ref['response']['initial_noise'][0])
  # Native packet contains raw reconstructed candidate actions, as does live.
  def targets(packet):
   return np.asarray(reconstruct_20hz(packet['response']['timed_candidates'][0],oracle_replay=False)['targets'],np.float32)
  np.testing.assert_array_equal(targets(q),targets(ref))
  final=d['final_truth'];initial=d['initial_truth']
  score=score_episode(initial,final,[t['truth'] for t in truth[-24:]],scene['config'],d['evaluation']['attempts'])
  assert score['criteria']==d['evaluation']['criteria'] and score['success']==d['evaluation']['success']
  checks.append({'mode':mode,'scene':sid,'control_steps':len(truth),'score_recomputed':True,'initial_rgb_proprio_noise_candidate0_match':True})
save(OUT/'complete.json',{'completed':True,'episodes':9,'no_training':True,'execution_only':True})
analyze(OUT)
videos=list(OUT.glob('*/*/rollout.mp4'))
def decode(f):
 r=subprocess.run(['ffmpeg','-v','error','-i',str(f),'-f','null','-'],capture_output=True,text=True)
 return {'path':str(f.relative_to(OUT)),'passed':r.returncode==0,'error':r.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:qa=list(pool.map(decode,videos))
assert len(qa)==9 and all(q['passed'] for q in qa)
summary=read(OUT/'validation_summary.json')
visual=OUT/'visual-review';visual.mkdir(exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
known='film-demo_0006';modes=protocol['routes']
readers=[imageio.get_reader(OUT/m/known/'rollout.mp4') for m in modes]
writer=imageio.get_writer(visual/'live-integration-comparison.mp4',fps=5,codec='libx264',quality=8)
count=0
try:
 for triplet in zip(*readers):
  panel=Image.new('RGB',(1920,400),'#111722');draw=ImageDraw.Draw(panel)
  for i,(frame,mode) in enumerate(zip(triplet,modes)):
   row=next(r for r in summary['rows'] if r['mode']==mode and r['scenario_id']==known)
   draw.text((640*i+8,4),{'native_reference':'Unchanged V4 (inference pauses physics)','first_admissible':'Live: first admissible, corrected guard','decisions':'Live: Decisions / fallback, corrected guard'}[mode],font=font,fill='white')
   draw.text((640*i+8,26),f"Pickup {row['pickup']} | place {row['placement_without_yaw']} | yaw gap {row['remaining_yaw_error_deg']:.1f} deg",font=font,fill='#aacfdf')
   panel.paste(Image.fromarray(frame).resize((640,320)),(640*i,52))
  draw.text((8,379),'Matched simulation time (live capture within 0.2 s); final outcomes labelled. Three inspected scenes; no generalisation claim.',font=font,fill='white')
  writer.append_data(np.asarray(panel));count+=1
 panel.save(visual/'live-integration-final.jpg',quality=93)
finally:
 writer.close()
 for r in readers:r.close()
extra=decode(visual/'live-integration-comparison.mp4');assert extra['passed']
save(OUT/'artifact-validation.json',{'episodes':checks,'videos':qa+[extra],'comparison_frames':count,'same_initial_inputs_and_first_candidate':True})
save(OUT/'complete.json',{'completed':True,'episodes':9,'no_training':True,'artifact_validation_passed':True})
print(json.dumps({'episodes':9,'videos_decoded':10,'comparison_frames':count,'all_inputs_and_scores_verified':True}))
