"""Re-score continuous segments and audit preserved world boundaries."""
import os,sys,json,pickle,gzip,hashlib,time,csv
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
from load04_common import scores
from supervisor.feeding_loading.evaluation import score_episode
O=Path('/var/tmp/typhon-hier01-chain-amend1-20261015');ORIGINAL=Path('/var/tmp/typhon-hier01-chain-20261015')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def analyse():
 p=read(O/'protocol.json');assert read(O/'main-complete.json')['segments']==2
 assert read(ORIGINAL/'partial-validation.json')['executed_segments']==3
 rs=read(ORIGINAL/'main-summary.json')+read(O/'main-summary.json');assert len(rs)==4
 p['workflows']=read(ORIGINAL/'protocol.json')['workflows']
 assert [r['context'] for r in rs]==[w['id']+'-stage'+str(i+1)+'-'+task for w in p['workflows'] for i,task in enumerate(w['tasks'])]
 proofs=[]
 for r in rs:
  d=Path(r['directory']);e=read(d/'episode.json');source=p['sources'][r['requested_task']]
  both=scores(e,source['config']);assert both==r['scores'];assert both[r['requested_task']]['success']==r['requested_success']
  assert e['policy_actions_executed']==r['policy_actions']==1436 and r['queries']==48 and r['final_step']-r['initial_step']==1460
  assert r['initial_step']==(40 if r['stage']==1 else 1500)
  assert r['final_step']==(1500 if r['stage']==1 else 2960)
  before=(d/'boundary-before.pkl').read_bytes();after=(d/'boundary-after-request.pkl').read_bytes();assert before==after
  state=pickle.loads(before);assert state['steps']==r['initial_step']
  with np.load(d/'initial-live.npz') as z:
   for k in ['qpos','qvel','mocap_pos','mocap_quat']:np.testing.assert_array_equal(z[k],state[k])
   with np.load(d/'predictions.npz') as pred:np.testing.assert_array_equal(pred['histories'][0],z['proprio'])
  assert not e['uses_simulator_truth_for_control']
  if r['stage']==1:assert read(d/'native-first-stage-equivalence.json')['passed']
  else:
   previous=next(q for q in rs if q['workflow']==r['workflow'] and q['stage']==1);pd=Path(previous['directory']);pe=read(pd/'episode.json')
   assert pe['final_truth']==e['initial_truth']
   np.testing.assert_array_equal(np.asarray(Image.open(pd/'final.png')),np.asarray(Image.open(d/'initial.png')))
   for task in ['feeding','loading']:
    # Initial assessment reuses genuine preceding stable samples only for reporting,
    # never as second-stage final success credit.
    if task==r['requested_task']:
     from load04_common import endpoint
     checked=score_episode(e['initial_truth'],e['initial_truth'],pe['stability_truths'],endpoint(source['config'],task),[])
     assert checked==e['initial_endpoint_assessment']
  trace=[json.loads(x) for x in gzip.open(d/'controller_trace.jsonl.gz','rt')];assert len(trace)==1460
  supp=[x['object_truth']['bilateral_grasp_contact'] and x['object_truth']['object_position_m'][2]-e['initial_truth']['object_position_m'][2]>=.01 for x in trace]
  lift=any(all(supp[i:i+3]) for i in range(len(supp)-2));assert lift==r['supported_lift_10mm_3steps']
  proofs.append({'context':r['context'],'initial_step':r['initial_step'],'final_step':r['final_step'],'boundary_physics_sha256':hashlib.sha256(before).hexdigest(),'fresh_measured_history_exact':True,'prior_terminal_equals_current_initial':r['stage']==2,'native_first_stage_exact':r['stage']==1,'strict_success':r['requested_success']})
 keys=['context','workflow','stage','requested_task','model','requested_success','opposite_success','initial_pose_satisfied','initial_stability_evidence','interpretation','supported_lift_10mm_3steps','position_error_mm','yaw_error_deg','tilt_deg','failed_criteria','wall_seconds']
 with (O/'comparisons.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rs)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
 sheet=Image.new('RGB',(1024,318*4),(16,18,23));draw=ImageDraw.Draw(sheet)
 for i,r in enumerate(rs):
  for j,(fn,label) in enumerate([('initial.png','START'),('final.png','END')]):
   x,y=512*j,318*i;sheet.paste(Image.open(Path(r['directory'])/fn).convert('RGB').resize((512,256)),(x,y+27))
   draw.text((x+4,y+4),f"{r['workflow']} stage{r['stage']} {r['requested_task']} {label}",font=font,fill='white')
   text=f"start already satisfied: {r['initial_pose_satisfied']}" if j==0 else f"strict {r['requested_success']} | pos {r['position_error_mm']:.1f}mm | tilt {r['tilt_deg']:.1f}"
   draw.text((x+4,y+286),text,font=font,fill='white')
   draw.text((x+4,y+303),f"supported lift {r['supported_lift_10mm_3steps']} | yaw {r['yaw_error_deg']:.1f}",font=font,fill='white')
 sheet.save(O/'initial-final-contact-sheet.png')
 videos=O/'videos';videos.mkdir(exist_ok=True);receipts=[]
 for flow in ['F-L','L-F']:
  path=videos/(flow+'-continuous.mp4');n=0
  with imageio.get_writer(path,fps=5,codec='libx264',quality=7,macro_block_size=16) as writer:
   for r in [q for q in rs if q['workflow']==flow]:
    rd=imageio.get_reader(Path(r['directory'])/'rollout.mp4')
    try:
     assert rd.count_frames()==367
     for frame in rd:
      panel=Image.new('RGB',(1024,544),(16,18,23));panel.paste(Image.fromarray(frame),(0,32));dr=ImageDraw.Draw(panel)
      dr.text((8,6),f"{flow} | request {r['stage']}: {r['requested_task']} | continuous world; fixed budget",font=font,fill='white');writer.append_data(np.asarray(panel));n+=1
    finally:rd.close()
  assert n==734;rd=imageio.get_reader(path);assert rd.count_frames()==734 and rd.get_data(733).size;rd.close();receipts.append({'path':str(path),'frames':n,'fps':5})
 workflows=[{'workflow':flow,'success':all(r['requested_success'] for r in rs if r['workflow']==flow),'segments':[r['context'] for r in rs if r['workflow']==flow],'same_simulator_retained':True} for flow in ['F-L','L-F']]
 dump(O/'combined-workflows.json',workflows);dump(O/'combined-summary.json',rs)
 for w in workflows:assert w['success']==all(r['requested_success'] for r in rs if r['workflow']==w['workflow'])
 dump(O/'validation_summary.json',{'passed':True,'canonical_segments':4,'total_physical_segments_including_interrupted_duplicate':5,'original_interrupted_second_segment_unexecuted':True,'workflows':2,'all_strict_scores_recomputed':True,'first_stage_native_trajectories_exact':2,'both_world_handover_states_unchanged':True,'no_reset_home_object_injection_or_truth_switch':True,'same_low_level_controller_retained':True,'bookkeeping_only_lifetime_budget':2960,'canonical_native_videos':4,'additional_interrupted_first_stage_video_preserved':str(ORIGINAL/'workers/L-F-stage1-loading/episode/rollout.mp4'),'continuous_videos':receipts,'proofs':proofs,'workflow_results':workflows,'epoch':time.time()})
 print(json.dumps({'workflows':workflows,'segments':[{k:r[k] for k in keys} for r in rs]}),flush=True)
if __name__=='__main__':analyse()
