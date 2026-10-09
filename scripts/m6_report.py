"""Audit isolated M6 episodes, full-weight training and separate raw-base control."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,csv,hashlib,gzip,math,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from supervisor.feeding_loading.evaluation import score_episode
ROOT=Path('results/personal-m6-joint-20261009')

def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def main():
 rows=[];audit=[];groups=['seen_source','development_validation','already_completed'];labels=['custom-init','FT3000','vanilla-domain3']
 for model in labels:
  folder=ROOT/'evaluation'/model
  if not (folder/'summary.json').exists():
   raise AssertionError('Missing complete M6 evaluation '+model)
  for r in json.loads((folder/'summary.json').read_text()):
   d=folder/r['id'];e=json.loads((d/'episode.json').read_text());s=json.loads((d/'scenario.json').read_text())
   result=score_episode(e['initial_truth'],e['final_truth'],e['stability_truths'],s['config'],e['attempts'])
   assert result['success']==e['success']==r['success']
   trace=[json.loads(x) for x in gzip.open(d/'trajectory.jsonl.gz','rt')]
   assert all(np.isfinite(x['applied_action']).all() and np.max(np.abs(x['applied_action']))<=1+1e-7 for x in trace)
   assert len(trace)<=1460
   tol=s['config']['evaluation'];fin=e['final_truth'];st=e['stability_truths'];criteria=result['criteria']
   physical_stable=(result['stability_sample_count']>=tol['minimum_stability_samples'] and result['stability_duration_s']+1e-9>=tol['minimum_stability_duration_s'] and result['object_movement_after_release_m']<=tol['stability_position_tolerance_m'] and result['object_yaw_movement_after_release_rad']<=tol['stability_yaw_tolerance_rad'] and result['maximum_linear_velocity_m_s']<=tol['linear_velocity_tolerance_m_s'] and result['maximum_angular_velocity_rad_s']<=tol['angular_velocity_tolerance_rad_s'])
   preserved=all(criteria[k] for k in ['position','orientation','marked_face_up','footprint_inside_target','placement_height']) and physical_stable and not fin['grasp_contact'] and all(not x['grasp_contact'] for x in st)
   rows.append({'model':model,**r,'object_goal_preserved_without_empty_gripper_open_requirement':preserved});audit.append({'model':model,'scenario':r['id'],'steps':len(trace),'score_recomputed':True,'actions_finite_bounded':True,'episode_sha256':digest(d/'episode.json')})
 assert len([r for r in rows if r['model'] in labels])==24
 tables=[]
 for model in labels:
  for g in groups:
   xs=[r for r in rows if r['model']==model and r['group']==g]
   tables.append({'model':model,'group':g,'successes':sum(bool(x['success']) for x in xs),'episodes':len(xs),'episodes_bilateral_contact':sum(x['bilateral_contact_steps']>0 for x in xs),'supported_lift_episodes':sum(x['supported_lift_10mm_3steps'] for x in xs),'mean_position_mm':float(np.mean([x['position_error_mm'] for x in xs])),'mean_yaw_deg':float(np.mean([x['yaw_error_deg'] for x in xs])),'mean_eef_path_m':float(np.mean([x['eef_path_m'] for x in xs])),'max_block_displacement_mm':max(x['max_object_displacement_mm'] for x in xs),'object_goal_preserved':sum(x['object_goal_preserved_without_empty_gripper_open_requirement'] for x in xs)})
 for model in labels:
  xs=[r for r in rows if r['model']==model+'-loading']
  if xs:tables.append({'model':model,'group':'exploratory_loading','successes':sum(bool(x['success']) for x in xs),'episodes':len(xs),'episodes_bilateral_contact':sum(x['bilateral_contact_steps']>0 for x in xs),'supported_lift_episodes':sum(x['supported_lift_10mm_3steps'] for x in xs),'mean_position_mm':float(np.mean([x['position_error_mm'] for x in xs])),'mean_yaw_deg':float(np.mean([x['yaw_error_deg'] for x in xs])),'mean_eef_path_m':float(np.mean([x['eef_path_m'] for x in xs])),'max_block_displacement_mm':max(x['max_object_displacement_mm'] for x in xs),'object_goal_preserved':sum(x['object_goal_preserved_without_empty_gripper_open_requirement'] for x in xs)})
 with (ROOT/'comparisons.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
 dump(ROOT/'episode_diagnostics.json',rows);dump(ROOT/'independent_score_audit.json',{'passed':True,'episodes':audit})
 training=json.loads((ROOT/'training/training.json').read_text());assert training['passed'] and training['updates']==3000
 assert json.loads((ROOT/'training/final.reload.json').read_text())['passed']
 # Fixed representative cases; no outcome-selected video replacement.
 import cv2
 preview=ROOT/'visual_review';preview.mkdir(exist_ok=True)
 for sid in ['m5-source-01','recovery-already_completed-01']:
  sheet=Image.new('RGB',(1536,3*300),'white');draw=ImageDraw.Draw(sheet)
  for rowidx,model in enumerate(labels):
   d=ROOT/'evaluation'/model/sid;e=json.loads((d/'episode.json').read_text());states=[json.loads(x) for x in gzip.open(d/'trajectory.jsonl.gz','rt')]
   z0=e['initial_truth']['object_position_m'][2];peak=max(states,key=lambda x:x['truth']['object_position_m'][2]-z0)
   if peak['truth']['object_position_m'][2]-z0<.01:
    peak=min(states,key=lambda x:np.linalg.norm(np.array(x['truth']['eef_position_m'])-x['truth']['object_position_m']))
    why='closest approach'
   else:why='maximum object height'
   cap=cv2.VideoCapture(str(d/'rollout.mp4'));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));decoded=0;selected={};idx=max(0,min(count-1,(peak['step']-e['initial_truth']['control_steps'])//4))
   while True:
    ok,frame=cap.read()
    if not ok:break
    if decoded in [0,idx,count-1]:selected[decoded]=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    decoded+=1
   cap.release();assert decoded==count and count>0
   for col,index in enumerate([0,idx,count-1]):
    im=selected[index].resize((512,256));sheet.paste(im,(col*512,rowidx*300+38))
    draw.text((col*512+5,rowidx*300+5),f'{model} | {sid} | '+(['initial',why,'final'][col]),fill='black')
    if col==0:draw.text((5,rowidx*300+20),f"strict success={e['success']}",fill='black')
  sheet.save(preview/(sid+'.jpg'))
 # Decode every video independently; reports never rewrite scientific episodes.
 videos=[]
 for p in sorted((ROOT/'evaluation').glob('*/*/rollout.mp4')):
  cap=cv2.VideoCapture(str(p));expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));count=0
  while True:
   ok,frame=cap.read()
   if not ok:break
   assert frame.size>0,p
   count+=1
  cap.release();assert count==expected and count>0,(p,count,expected)
  videos.append({'path':str(p.relative_to(ROOT)),'frames':count,'decoded':True})
 assert len(videos)==24
 dump(ROOT/'video_validation.json',{'passed':True,'videos':videos,'not_exhaustive_human_review':True})
 losses=[json.loads(x) for x in (ROOT/'training/losses.jsonl').read_text().splitlines()]
 assert len(losses)==3000 and all(not r['joint'] for r in losses[:1000]) and all(r['joint'] for r in losses[1000:])
 assert all(r['grads']['vlm']==0 and r['grads']['core']==0 for r in losses[:1000])
 assert all(r['grads']['vlm']>0 and r['grads']['core']>0 for r in losses[1000:])
 phases={name:{'updates':len(xs),'sum_update_seconds':sum(r['seconds'] for r in xs),'mean_position_loss_last100':float(np.mean([r['position_loss'] for r in xs[-100:]])),'mean_rotation_loss_last100':float(np.mean([r['rotate6D_loss'] for r in xs[-100:]])),'mean_gripper_loss_last100':float(np.mean([r['gripper_loss'] for r in xs[-100:]]))} for name,xs in [('warmup',losses[:1000]),('joint_ramp',losses[1000:2000]),('joint_hold',losses[2000:])]}
 dump(ROOT/'training_phase_audit.json',{'passed':True,'phases':phases,'warmup':json.loads((ROOT/'training/warmup_gate.json').read_text())})
 lines=['M6: separate foundation full-weight joint fine-tuning','', 'Fixed primary comparison: Pt-custom-init versus Pt-FT3000, 16 episodes. Additional unchanged released-Pt/domain3 diagnostic:8 episodes, outside primary denominator.','', 'MODEL / GROUP / STRICT SUCCESS / BILATERAL CONTACT / SUPPORTED LIFT']
 for t in tables:lines.append(f"{t['model']} / {t['group']} / {t['successes']}/{t['episodes']} / {t['episodes_bilateral_contact']}/{t['episodes']} / {t['supported_lift_episodes']}/{t['episodes']}")
 lines+=['','Training uses2toINF/X-VLA-Pt revisionc1c4a64a7e03ac5b95c468bf1578f3d03651b53b, separate from every earlier checkpoint. The pinned native forward and loss are used; this is full-weight adaptation of original VLM and action backbone, not LoRA.','Domain29 prompt/action heads initialized independently with pinned initializer distributions; inactive domain rows hash-unchanged. First1000 updates train prompt/heads; next1000 ramp all four parameter groups; final1000 hold peak learning rates. AdamW(0.9,0.95),eps1e-8,clip1,wd0.01(active domain row only for tables),VLM/prompt peak1e-5,action core/heads1e-4. FP32 weights/gradients/moments with BF16 autocast; microbatch1 x accumulation4.','Data:one personal-derived source simulation configuration, personal_demo_0001; 48 complete30-step windows, seed17, 3000updates,12000window presentations. Robot labels came from a geometric expert, not human robot-action measurements. No held-out data used for fitting. Development-validation configurations were previously inspected and are not untouched confirmatory tests.','M6 converts labels, input proprio and native outputs consistently to generic interleaved6D rotation. Physical XYZ remains world metres, 20-dimensional single-arm-plus-dummy contract. The existing concatenated-column OSC decoder is unchanged; the worker permutes back at that boundary. Existing M0-M5 arrays and results are unchanged.','O2:two full RGBviews and RGB-only detail crop, masked where unreliable. H30,10denoising steps,20Hz. Initial measured robot pose then previous-command history. Existing controller/scorer, no completion guard, wrist projection or expert rescue. Simulator pauses for inference.','The untouched released-Pt/domain3 diagnostic keeps released weights and uses the LIBERO-style concatenated-column adapter. It is a custom-scene diagnostic, not a native LIBERO benchmark. Pt-custom-init loads the saved pretraining domain tables and is not labelled untouched vanilla.','The encoder embedding missing-key warning is resolved by restoring shared Parameter identity to the serialized shared embedding, with exact safetensors equality checked. No randomly initialized missing encoder tensor is used.','',f"Measured training wall time, including final save/probe/hash: {training['wall_seconds']:.2f}s. Mean optimizer step: {training['mean_step_seconds']:.4f}s. Peak allocated GPU memory: {training['peak_cuda_allocated_bytes']/1e9:.3f}GB.",'Full model and optimizer saved and reloaded in a fresh process with zero denoised-action discrepancy. Shared weights were hash-unchanged and had zero Adam state during warmup. VLM and core have nonzero gradients throughout joint stages; changed tensors and inactive-row hashes are recorded.','Both feeding and loading geometric-expert decoder replays succeeded; maximum command roundtrip error2.22e-16. These qualify the interface, not the learned policy.','', 'Interpretation:lower training loss does not establish manipulation skill. Source fitting may memorize command history. Completed-state preservation does not establish completion recognition. The arm may move while the block remains in the target; traces and displacement/path metrics expose that distinction. Closing commands are not grasps; supported lift requires10mm rise with simultaneous bilateral contact for3consecutive20Hzsteps. Small denominators and one training seed do not support broad transfer or reliability claims.','M6 differs from M5 in base checkpoint, custom-domain initialization, rotation ordering and XYZ loss weighting as well as optimizer scope. This is not an isolated causal comparison of full-weight versus LoRA. This truncated3000-update run does not reproduce the paper benchmark training budget.','', 'Evidence:comparisons.csv,episode_diagnostics.json,independent_score_audit.json,training_phase_audit.json,training/final.reload.json,video_validation.json,visual_review/*.jpg. All24video files fully decoded; representative visual inspection recorded separately.','Reproduction commands:handoff/M6_RUN_PLAN.txt; scripts/m6_after_training.sh; source hashes in execution_source_sha256.json. Fresh result directories required. This dated experiment has an explicit training cutoff; future reproductions must record a new budget.','Final backup and visual findings are recorded separately after verification.']
 (ROOT/'M6_report.txt').write_text('\n'.join(lines)+'\n')
 dump(ROOT/'validation_summary.json',{'passed':True,'primary_episodes':16,'additional_vanilla_diagnostic_episodes':8,'all_scores_recomputed':True,'actions_finite_bounded':True,'all_video_frames_decoded':True,'checkpoint_reload_exact':True,'tables':tables,'visual_review_status':'contact sheets generated; inspection pending','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 print(json.dumps(tables,indent=2))
if __name__=='__main__':main()
