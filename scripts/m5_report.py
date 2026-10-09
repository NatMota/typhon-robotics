"""Audit every saved M5 episode and make compact scientific closeout artifacts."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,csv,hashlib,gzip,math,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from supervisor.feeding_loading.evaluation import score_episode
ROOT=Path('results/personal-m5-20261009')

def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def main():
 rows=[];audit=[];groups=['seen_source','development_validation','already_completed'];labels=['F','C3000','A3000']
 for model in labels+['W1000']+[m+'-loading' for m in labels]:
  folder=ROOT/'evaluation'/model
  if not (folder/'summary.json').exists():
   assert model.endswith('-loading'),model
   continue
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
 training={m:json.loads((ROOT/'training'/m/'training.json').read_text()) for m in labels[1:]}
 sampling=json.loads((ROOT/'sampling_validation.json').read_text());assert sampling['passed']
 for m in labels[1:]:assert json.loads((ROOT/'training'/m/(m+'.reload.json')).read_text())['passed']
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
 # Loss components and checkpoints are diagnostics, not selection criteria.
 # Raw and per-update losses are retained in each training folder; no plotting dependency added.
 lines=['M5 — Bounded source-task adaptation','', 'The fixed comparison completed. Counts below are independent closed-loop outcomes, not inferred from training loss.','', 'METHOD / GROUP / STRICT SUCCESS / BILATERAL CONTACT / SUPPORTED LIFT']
 for t in tables:lines.append(f"{t['model']} / {t['group']} / {t['successes']}/{t['episodes']} / {t['episodes_bilateral_contact']}/{t['episodes']} / {t['supported_lift_episodes']}/{t['episodes']}")
 lines+=['','Design: one personal-derived source configuration (personal_demo_0001); robot action labels generated by the geometric expert. Three source seeds, three historically exposed development-validation configurations, two already-completed states per model. W1000 is an additional one-case diagnostic, outside the 24 core outcomes.','Both trained conditions start the pinned original X-VLA, seed 17, FP32 effective batch 4, 3000 updates, identical 48-window sampling order. C changes prompts only. A trains prompts/heads for 1000 steps then adds LoRA for 2000 steps with a 1000-step ramp. Complete checkpoint save/reload and base/inactive-domain audits pass.','LoRA: rank 8, alpha 16, 266 compatible linear modules; 3,178,496 policy and 4,849,664 VLM adapter parameters. Domain 3 prompt/head entries 128,020; inactive domain rows remain unchanged. Optimizer details and actual changed tensors are in training manifests.','Native LIBERO was unavailable in the existing environments; no rebuild was attempted. Native training dropout mode differs from earlier M3 eval-mode adaptation, but is matched across both M5 conditions.','O2 is two full views plus current-RGB detail crop; occluded/missing crops are masked. H30, domain 3, 10 denoising steps, 20 Hz, existing action adapter/controller/scorer unchanged. No completion guard, wrist projection or expert rescue. Inference pauses simulator time.','', 'Measured training-loop wall times (include preprocessing/checkpoint probes; exclude initial loading and final independent reload):']
 for m,t in training.items():lines.append(f"{m}: {t['wall_s']:.2f}s; mean update {t['mean_update_s']:.4f}s; peak allocated CUDA {t['peak_cuda_bytes']/1e9:.3f}GB; frozen original parameters unchanged={t['frozen_parameters_unchanged']}.")
 lines+=['','Interpretation boundaries: source fitting can memorize command history and does not prove visual grounding. Development cases are not untouched confirmatory tests. Already-completed preservation alone is not completion recognition. Closing edges are not grasps. Supplemental supported lift requires >=10mm rise with bilateral contact for >=3 consecutive 20 Hz steps and does not replace the strict score or visual review. Small denominators and one training seed do not support broad reliability/transfer claims.','', 'Evidence: comparisons.csv; episode_diagnostics.json; independent_score_audit.json; evaluation/*/*/rollout.mp4 and controller/trajectory traces; visual_review/*.jpg; training/*/losses.jsonl; sampling_validation.json; training/*/*.reload.json.','Reproduction: prepare the pinned base and canonical M3 data/environment symlinks; install peft==0.15.2 --no-deps --target .m5-deps; run python -m supervisor.adaptation.m5 protocol, six-update profile+verify, then bash scripts/m5_run.sh in a fresh output root. Runners refuse to overwrite prior experiment directories. Code-enforced deadlines apply to this dated run and must be explicitly re-declared for a future reproduction.','LoRA parameters had requires_grad=False, grad=None and no AdamW history through update 1000; prompt/head optimizer state continues into stage 2. Profile and scientific training were separate processes with fresh original-base loads. Parameter isolation does not establish unchanged behavior in other domains because shared LoRA changed.', 'Optional loading runs are reported only if final A had strict source success. Their predeclared cases are loading-standard-01 seed 221000 and loading-standard-02 seed 221001. No passing gate means no loading rollouts; this is not a policy failure score.', 'Final backup hashes and visual interpretation are saved separately after review.']
 (ROOT/'M5_report.txt').write_text('\n'.join(lines)+'\n')
 dump(ROOT/'validation_summary.json',{'core_episodes':24,'warmup_diagnostic_episodes':1,'all_scores_recomputed':True,'same_sampling':True,'exact_checkpoint_reloads':True,'tables':tables,'visual_review_status':'contact sheets generated; agent inspection pending','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 print(json.dumps(tables,indent=2))
def closeout():
 """Decode every rollout and document the completed-state scoring edge case."""
 import cv2
 videos=[]
 for p in sorted((ROOT/'evaluation').glob('*/*/rollout.mp4')):
  cap=cv2.VideoCapture(str(p));expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));count=0
  fps=cap.get(cv2.CAP_PROP_FPS)
  while True:
   ok,frame=cap.read()
   if not ok:break
   assert frame.size>0,p
   count+=1
  cap.release();assert count==expected and count>0,(p,expected,count)
  videos.append({'path':str(p.relative_to(ROOT)),'frames':count,'fps':fps,'all_frames_decoded':True})
 assert len(videos)==25,len(videos)
 dump(ROOT/'video_validation.json',{'passed':True,'videos':videos,'scope':'All frames decoded; this is not exhaustive human visual review.'})
 e=json.loads((ROOT/'evaluation/C3000/recovery-already_completed-03/episode.json').read_text())
 assert not e['success'] and e['evaluation']['criteria']['released'] and not e['evaluation']['criteria']['stable']
 assert e['final_truth']['gripper_open'] and not e['stability_truths'][0]['gripper_open']
 assert all(x['gripper_open'] and not x['grasp_contact'] for x in e['stability_truths'][1:])
 assert not e['stability_truths'][0]['grasp_contact']
 assert e['initial_truth']['object_position_m']==e['final_truth']['object_position_m']
 explanation='C3000 completed03 preserved the block at its goal. Final gripper was open and released criterion passed. The empty gripper was not fully open at the first stability sample (step 1478), then was open at steps 1480 through 1500; the original stable criterion therefore failed. The brick was not disturbed. Original score remains failure.'
 dump(ROOT/'completed_state_score_detail.json',{'case':'C3000/recovery-already_completed-03','explanation':explanation,'criteria':e['evaluation']['criteria'],'stability_gripper_open':[{'step':x['control_steps'],'open':x['gripper_open'],'contact':x['grasp_contact']} for x in e['stability_truths']]})
 review=ROOT/'visual_review.json'
 if review.exists():
  r=json.loads(review.read_text());r['observations']=[x for x in r['observations'] if not x.startswith('C completed03') and not x.startswith('C3000 completed03')]+[explanation]
  r['corrected_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();dump(review,r)
 v=json.loads((ROOT/'validation_summary.json').read_text());v.update({'visual_review_status':'Two representative contact sheets inspected by Codex; all 25 videos fully decoded; exhaustive human video review not claimed.','all_videos_decoded':True,'completed_case_score_explanation':explanation,'loading_gate':'Not run: A3000 achieved no strict source success.','closed_out_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});dump(ROOT/'validation_summary.json',v)
 with (ROOT/'M5_report.txt').open('a') as f:
  f.write('\nRESULT AND INTERPRETATION\n')
  f.write('The broader adaptation did not establish source-task competence: 0/3 source and 0/3 development-validation successes, with no object contact or supported lift. Prompt-only adaptation also completed none of the unfinished cases. The frozen baseline contacted objects in two validation episodes but achieved no supported lift or completion. The W1000 diagnostic failed without contact or lift. Optional loading was not run because its source-success gate was not met.\n')
  f.write('Already-completed block arrangements were preserved in both cases by all three models. Original strict scores are F 2/2, C3000 1/2, A3000 2/2. '+explanation+'\n')
  f.write('The broader model reduced mean end-effector travel from 5.49 to 2.31 m on source cases and from 5.13 to 1.26 m on already-completed cases. It still moved the arm, and neither shorter travel nor leaving the block alone establishes completion recognition or a useful manipulation policy. Shared-head and LoRA changes are evaluated together, so their individual effects are not isolated.\n')
  f.write('Representative source01 and completed01 contact sheets were inspected: the source block remains at its initial location, and the completed block remains in the target while the arm moves. All 25 rollout videos decode completely. These checks do not claim every frame was human reviewed. The invalid first explanation of the C3000 stability failure was rejected by an assertion and preserved in visual_review_initial_assertion_failure.json.\n')
  f.write('Closeout reproduction: .venv-sim/bin/python scripts/m5_report.py. This regenerates derived reports and contact sheets, independently recomputes scores, validates all saved actions and videos, and checks the completed-state scoring detail without rerunning training or simulation. Scientific episode files are read-only inputs.\n')
 print(json.dumps({'closeout':'passed','videos':len(videos),'frames':sum(x['frames'] for x in videos)}))

if __name__=='__main__':
 main()
 closeout()
