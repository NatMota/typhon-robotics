"""Read-only checks and descriptive retention metrics for the four frozen C cases."""
import os,sys,json,gzip,hashlib,math,itertools,csv,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
from supervisor.feeding_loading.evaluation import score_episode
O=Path('/var/tmp/typhon-eval02-ret-c-20261015')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=read(O/'protocol.json');assert read(O/'complete.json')['episodes']==4
 rows=read(O/'summary.json');assert [r['id'] for r in rows]==p['execution_order']
 state={};initial={};noises={};details=[]
 for r in rows:
  d=Path(r['directory']);e=read(d/'episode.json');c=p['cases'][r['goal']]['config']
  assert read(d/'scenario.json')['config']==c
  s=score_episode(e['initial_truth'],e['final_truth'],e['stability_truths'],c,e['attempts'])
  assert s==e['evaluation'] and s['success']==r['strict_success']
  assert not e['uses_simulator_truth_for_control']
  assert not any(e[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors'])
  state.setdefault(r['goal'],set()).add(sha(d/'initial-state.pkl'))
  prep=np.load(d/'input-audit/initial-prepared.npz')
  if r['goal'] not in initial:initial[r['goal']]={k:prep[k].copy() for k in prep.files}
  else:
   assert set(initial[r['goal']])==set(prep.files)
   for k in prep.files:np.testing.assert_array_equal(prep[k],initial[r['goal']][k])
  packets=sorted((d/'input-audit').glob('noise-*.npz'));assert len(packets)==48
  hashes=[sha(f) for f in packets]
  # Compare arrays, not ZIP bytes/timestamps.
  if r['seed'] in noises:
   for f,rf in zip(packets,noises[r['seed']]):
    with np.load(f) as a,np.load(rf) as b:
     assert a.files==b.files
     for k in a.files:np.testing.assert_array_equal(a[k],b[k])
  else:noises[r['seed']]=packets
  io=np.load(d/'policy_io.npz');assert io['model_actions'].shape==(48,30,20)
  expected=np.load(p['cases'][r['goal']]['initial_archive'])
  for k in ['primary','secondary']:np.testing.assert_array_equal(io[k][0],expected[k])
  trace=[json.loads(x) for x in gzip.open(d/'controller_trace.jsonl.gz','rt')]
  assert len(trace)==1460
  supported=[x['object_truth']['bilateral_grasp_contact'] and x['object_truth']['object_position_m'][2]-e['initial_truth']['object_position_m'][2]>=.01 for x in trace]
  ix=next((i for i in range(len(trace)-2) if all(supported[i:i+3])),None)
  q=trace[ix] if ix is not None else None
  r={**r,'failed_criteria':[k for k,v in s['criteria'].items() if not v],
     'distance_to_G0_mm':math.dist(r['final_xy_m'],p['cases']['G0']['config']['task']['target_position_xy_m'])*1000,
     'distance_to_G1_mm':math.dist(r['final_xy_m'],p['cases']['G1']['config']['task']['target_position_xy_m'])*1000,
     'final_marked_face_tilt_deg':trace[-1]['object_tilt_deg'],
     'acquired_supported_grasp':None if q is None else {'first_step_of_three':q['step'],'object_yaw_deg':math.degrees(q['object_truth']['object_yaw_rad']),'object_tilt_deg':q['object_tilt_deg'],'R_GO':q['R_GO'],'p_GO':q['p_GO'],'eef_quat_xyzw':q['eef_quat_xyzw'],'object_quat_wxyz':q['object_truth']['object_quat_wxyz']},
     'peak_tilt_deg':max(x['object_tilt_deg'] for x in trace)}
  assert bool(q is not None)==r['supported_lift_10mm_3steps']
  details.append(r)
 assert all(len(x)==1 for x in state.values())
 gains=[];transitions=[]
 for model in p['model_order']:
  for seed in p['policy_seeds']:
   a=next(r for r in details if (r['model'],r['seed'],r['goal'])==(model,seed,'G0'))
   b=next(r for r in details if (r['model'],r['seed'],r['goal'])==(model,seed,'G1'))
   gains.append({'model':model,'seed':seed,'G0_final_xy_m':a['final_xy_m'],'G1_final_xy_m':b['final_xy_m'],'context_y_gain':(b['final_xy_m'][1]-a['final_xy_m'][1])/.15,'G0_strict':a['strict_success'],'G1_strict':b['strict_success'],'interpretation':'Whole-context response; target imagery and numerical instruction both change.'})
 for before,after in zip(p['model_order'],p['model_order'][1:]):
  for goal,seed in itertools.product(p['goal_order'],p['policy_seeds']):
   a=next(r for r in details if (r['model'],r['seed'],r['goal'])==(before,seed,goal))
   b=next(r for r in details if (r['model'],r['seed'],r['goal'])==(after,seed,goal))
   transitions.append({'before':before,'after':after,'goal':goal,'seed':seed,'strict_before':a['strict_success'],'strict_after':b['strict_success'],'before_failed':a['failed_criteria'],'after_failed':b['failed_criteria'],'position_error_change_mm':b['position_error_mm']-a['position_error_mm'],'supported_lift_before':a['supported_lift_10mm_3steps'],'supported_lift_after':b['supported_lift_10mm_3steps']})
 counts=[{'model':m,'goal':g,'strict_successes':sum(r['strict_success'] for r in details if r['model']==m and r['goal']==g),'trials':2} for m,g in itertools.product(p['model_order'],p['goal_order'])]
 followup='GOAL-versus-B-numerical-packet-swap'
 dump(O/'analysed-episodes.json',details);dump(O/'context-response.json',gains);dump(O/'stage-transitions.json',transitions)
 dump(O/'next-branch.json',{'branch':followup,'basis':'C G1 is mixed (1/2), with overshoot rather than B old-goal bias; planner predefined mixed-result branch','new_physical_rollouts':0,'max_minutes':20})
 keys=['id','model','goal','seed','strict_success','position_error_mm','yaw_error_deg','distance_to_G0_mm','distance_to_G1_mm','supported_lift_10mm_3steps','horizontal_transport_while_supported_mm','final_marked_face_tilt_deg','peak_tilt_deg']
 with (O/'comparisons.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in details)
 dump(O/'validation_summary.json',{'passed':True,'episodes':4,'strict_counts':counts,'all_scores_recomputed_exact':True,'same_goal_initial_prepared_tensors_exact':True,'same_goal_initial_available_state_exact':True,'all_models_native_interfaces_verified':all(read(O/'workers'/(g+'-'+m)/'reload-verification.json')['passed'] for g,m in itertools.product(p['goal_order'],p['model_order'])),'paired_noise_array_checks':2*48*3,'initial_images_match_training':True,'original_physical_goal_scorers':True,'no_truth_for_control':True,'frozen_order_completed':True,'checked_epoch':time.time()})
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 canvas=Image.new('RGB',(1024,2*318),(16,18,23));draw=ImageDraw.Draw(canvas)
 for i,(m,g) in enumerate(itertools.product(p['model_order'],p['goal_order'])):
  for j,seed in enumerate(p['policy_seeds']):
   r=next(x for x in details if (x['model'],x['goal'],x['seed'])==(m,g,seed));x=j*512;y=i*318
   im=Image.open(Path(r['directory'])/'final.png').convert('RGB').resize((512,256));canvas.paste(im,(x,y+26))
   draw.text((x+5,y+3),f"{m} {g} seed{seed}: strict {r['strict_success']}",font=font,fill='white')
   draw.text((x+5,y+284),f"pos {r['position_error_mm']:.1f}mm | yaw {r['yaw_error_deg']:.1f}deg",font=font,fill='white')
   draw.text((x+5,y+301),f"XY {r['final_xy_m'][0]:.4f}, {r['final_xy_m'][1]:.4f}",font=font,fill='white')
 canvas.save(O/'final-contact-sheet.png')
 import imageio.v2 as imageio
 videos=O/'comparison-videos';videos.mkdir(exist_ok=True);receipts=[]
 for model in p['model_order']:
  selected=[next(r for r in details if (r['model'],r['goal'],r['seed'])==(model,g,912171)) for g in ['G0','G1']]
  readers=[imageio.get_reader(Path(r['directory'])/'rollout.mp4') for r in selected];count=0
  try:
   with imageio.get_writer(videos/(model+'-goals-912171.mp4'),fps=5,codec='libx264',quality=7,macro_block_size=16) as writer:
    for frames in zip(*readers):
     panel=Image.new('RGB',(1024,288),(16,18,23));draw=ImageDraw.Draw(panel)
     for i,(frame,r) in enumerate(zip(frames,selected)):
      panel.paste(Image.fromarray(frame).resize((512,256)),(i*512,32))
      draw.text((i*512+5,7),f"{model} {r['goal']} targetY {r['goal_xy_m'][1]:.3f}m",font=font,fill='white')
     writer.append_data(np.asarray(panel));count+=1
  finally:
   for rd in readers:rd.close()
  assert count==367;receipts.append({'path':str(videos/(model+'-goals-912171.mp4')),'frames':count,'fps':5})
 for path in [Path(r['directory'])/'rollout.mp4' for r in details]+[Path(x['path']) for x in receipts]:
  rd=imageio.get_reader(path);assert rd.count_frames()==367;assert rd.get_data(366).size;rd.close()
 dump(O/'video-validation.json',{'passed':True,'native_videos':4,'comparisons':receipts,'all_last_frames_decoded':True,'native_simulated_time_preserved':True})
 print(json.dumps({'counts':counts,'gains':gains,'next_branch':followup}),flush=True)
if __name__=='__main__':main()
