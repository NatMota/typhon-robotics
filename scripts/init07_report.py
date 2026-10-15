"""Compact INIT07 results and reproduction notes from independently checked records."""
import sys,csv,json,gzip,math
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path[:0]=[str(W),str(W/'scripts')]
import numpy as np
from init07_common import OUT,read,dump,sha
from supervisor.feeding_loading.evaluation import score_episode
from load04_common import endpoint
def yes(v):return v is True or v=='True'
def main():
 s=read(OUT/'core-comparison.json');assert s['complete']and s['core_new_completed']==46 and s['historical_T_reused']==23
 assert read(OUT/'reproduction-validation.json')['passed']
 rows=list(csv.DictReader((OUT/'comparisons.csv').open()));by={(r['model'],r['case']):r for r in rows}
 p=s['panels'];g=s['gates'];opt=read(OUT/'optional-execution.json');optional=[]
 if opt['executed']:
  assert s['Q1_repair_gate_passed']
  for m in ['Q0','Q1']:
   for c in read(OUT/'optional-evaluation-cases.json')['cases']:
    d=OUT/'episodes'/m/c['id'];e=read(d/'episode.json');r=read(d/'summary.json')
    result=score_episode(e['initial_truth'],e['final_truth'],e['stability_truths'],endpoint(c['config'],c['requested_task']),e['attempts'])
    assert result==e['evaluation']and result['success']==r['requested_success']
    with gzip.open(d/'trajectory.jsonl.gz','rt')as f:states=[json.loads(x)['truth']for x in f]
    with gzip.open(d/'controller_trace.jsonl.gz','rt')as f:tr=[json.loads(x)for x in f]
    assert len(states)==len(tr)==1460
    displacement=float(np.linalg.norm(np.asarray([z['object_position_m']for z in states])-e['initial_truth']['object_position_m'],axis=1).max()*1000)
    contact=any(z['grasp_contact']for z in states)or any(any('cube'in str(c)and'gripper'in str(c)for c in z['contacts'])for z in tr)
    qualified=bool(result['success']and not contact and displacement<=1)if c['requested_task']=='feeding'else result['success']
    assert abs(displacement-r['maximum_object_displacement_mm'])<1e-9 and contact==r['any_gripper_object_contact']
    optional.append({'model':m,'case':c['id'],'task':c['requested_task'],'qualified':qualified,'strict_endpoint':result['success'],'any_contact':contact,'maximum_object_displacement_mm':displacement,'position_error_mm':r['position_error_mm'],'yaw_error_deg':r['yaw_error_deg'],'source_summary_sha256':sha(d/'summary.json')})
 else:assert not list((OUT/'episodes').glob('*/dev-*/summary.json'))
 dump(OUT/'optional-comparison.json',{'executed':opt['executed'],'episodes':optional,'new_episodes':len(optional),'quarantined_previously_inspected_handover':True,'not_new_human_or_session_holdout':True})
 if g['Q1']['passed']and not g['Q0']['passed']:
  conclusion='Learner-state repair passed the frozen local gate; equal-budget rehearsal of the old states did not. This supports the added state-conditioned supervision within this panel.'
 elif g['Q1']['passed']and g['Q0']['passed']:
  conclusion='Both matched continuations passed the frozen local gate. Extra old-state rehearsal was sufficient here; this comparison does not establish that learner-state examples were necessary.'
 elif not g['Q1']['passed']and g['Q0']['passed']:
  conclusion='Old-state rehearsal passed the frozen local gate; learner-state repair did not. The proposed repair is not qualified by this comparison.'
 else:
  conclusion='Neither matched continuation passed the frozen local gate. The paired learner-state examples did not resolve the full preservation-versus-loading requirement under this training budget.'
 text=['# INIT07 — paired learner-state repair','',conclusion,'',
 'T is the frozen INIT02 model. Q0 and Q1 each receive 3,000 additional full-joint updates starting independently from T. Q0 rehearses original q0 states; Q1 substitutes qualified paired hold/loading references at two q1 states actually reached by T after its first 30 controls.','',
 '| Test | T: historical parent | Q0: old-state rehearsal | Q1: learner-state repair |','|---|---:|---:|---:|']
 titles={'initiation':'Load from a completed feeding state','preservation':'Leave completed feeding intact','prior_feeding':'Earlier feeding cases','prior_loading':'Earlier loading cases'}
 for panel,title in titles.items():text.append('| '+title+' | '+' | '.join('%d/%d'%(p[m][panel]['qualified'],p[m][panel]['planned'])for m in ['T','Q0','Q1'])+' |')
 text+=['','There are 46 new full native episodes and 23 reused historical T reference episodes. The eight-case panels are two policy seeds at four nearby starting states, not eight independent scene families. No new human/session holdout claim or population-level reliability claim follows from these counts.','',
 '## Pickup and completed-task disturbance','',
 '| Model | Loading pickup | Earlier feeding pickup | Earlier loading pickup | Contact in completed-task cases | Maximum completed-task brick movement |','|---|---:|---:|---:|---:|---:|']
 for m in ['T','Q0','Q1']:
  rr=[r for r in rows if r['model']==m and r['panel']=='preservation']
  text.append('| %s | %d/8 | %d/3 | %d/4 | %d/8 | %.3f mm |'%(m,p[m]['initiation']['supported_lift'],p[m]['prior_feeding']['supported_lift'],p[m]['prior_loading']['supported_lift'],sum(yes(r['any_contact'])for r in rr),max(float(r['maximum_object_displacement_mm'])for r in rr)))
 text+=['','## Separate task subgoals','','| Panel | Model | Supported pickup | Stable goal placement, ignoring yaw | Mean final yaw error |','|---|---|---:|---:|---:|']
 for panel,title in titles.items():
  for m in ['T','Q0','Q1']:
   z=p[m][panel];text.append('| %s | %s | %d/%d | %d/%d | %.2f deg |'%(title,m,z['supported_lift'],z['planned'],z['goal_placement_without_yaw'],z['planned'],z['mean_final_yaw_error_deg']))
 text+=['','Placement here requires every endpoint criterion except directed yaw. In completed-feeding cases this measures preserved goal placement, not a newly performed move. comparisons.csv also records initial angular error and its signed reduction; no unstable ratio is used when the starting angle is already nearly correct.']
 text+=['','Pickup requires bilateral-contact-supported lift of at least 10 mm for three consecutive control steps. Strict endpoint success also requires the goal position (15 mm tolerance), directed yaw (10 degrees), uprightness, height, footprint, release and stability; loading must be seated in the fixture. Preservation additionally requires zero gripper-object contact and no more than 1 mm brick movement throughout all 1,460 control ticks. Wrist movement is recorded separately and does not itself prove completion recognition.','',
 '## Frozen gate and paired changes','',
 '| Gate | Q0 | Q1 |','|---|---|---|']
 for name in g['Q0']['criteria']:text.append('| '+name.replace('_',' ')+' | '+str(g['Q0']['criteria'][name])+' | '+str(g['Q1']['criteria'][name])+' |')
 for m in ['Q0','Q1']:
  text+=['',m+' lost original successful loading cases: '+(', '.join(g[m]['lost_T_loading_ids'])or'none')+'.',m+' lost original successful earlier-task cases: '+(', '.join(g[m]['lost_T_prior_ids'])or'none')+'.']
 text+=['','The gate retains the same five T loading successes and the same five T earlier-task successes, rather than accepting an equal total with different lost cases. paired-changes.json lists every paired gain and loss; comparisons.csv gives pickup, final position/yaw, contact, wrist travel and failure class for every core case.','',
 '## Training and data','',
 '| Model | Additional joint updates | Presentations | Training/save/probe wall time | Final weight SHA-256 |','|---|---:|---:|---:|---|']
 for m in ['Q0','Q1']:
  c=read(OUT/('train-'+m)/'complete.json');text.append('| %s | 3000 | 12000 | %.1f s | %s |'%(m,c['wall_s'],c['checkpoint_sha256']))
 text+=['',
 'Both start from immutable T SHA-256 '+read(OUT/'protocol.json')['initializer']['sha256']+'. They share seed 151902, initial parameter and RNG hashes, fresh AdamW moments, effective batch four with microbatch one, BF16 autocast and the executed INIT02 optimizer groups. Shared vision-language/core weights and active domain-29 prompt/head parameters train jointly; inactive domain rows remain unchanged. No new prompt-only phase, warm-up, scheduler, LoRA or RL was added. Optimizer moments were not saved, so an interrupted optimizer state cannot be resumed exactly. Only update 3,000 is eligible.','',
 'Each arm receives 3,000 varied-goal feeding samples, 3,000 earlier loading samples, 1,000 common completed-feeding samples, 1,000 common transition-loading samples and 4,000 designated q0-or-q1 samples. This is 6,000 feeding plus 6,000 loading presentations. The 8,000 common sample identities and positions are identical across arms. Each designated parent/task supplies 1,000 draws: 250 of the first full H30 window plus 750 balanced draws from other complete windows. Each three-update block preserves the fixed mixture while shuffling positions. No partial-window padding, new loss weights, phase labels or outcome-selected checkpoint are introduced.','',
 'The qualified bank has 28 trajectories and 1,537 complete 30-action windows. Four new q1 trajectories contribute 124 windows (16 hold and 108 loading); earlier source trajectories are unchanged. q1 first-step proprioception preserves the native previous-prediction raw-6D convention and the paired reference inputs qualified in INIT06. Model observation O2 remains two wide camera views plus the original object crop. Hidden audit arrays are not supplied as model inputs.','',
 'Personal recordings supply documented scenario origins and task goals. Expert Panda controls supply action labels. These are not recovered human motor commands or frame-aligned phone-video action labels. The teacher uses simulator truth; the evaluated policy receives the native RGB/proprioception/instruction inputs.','',
 '## Native execution and optional development panel','',
 'Every core episode executes 1,436 policy targets plus 24 fixed settling ticks, with H30 reobservation and ten flow refinements. There is no harness, candidate ranking, TTC, teacher intervention, runtime phase gate, scene reset after request, or outcome retry. All initial processed observations and all 48 random-noise packets match the original T cases. Fresh-process checkpoint reloads must reproduce their saved native probe exactly.']
 if optional:
  text+=['','The frozen repair gate passed, so the eight optional episodes were executed on the previously inspected demo0012 F-to-L handover (two requests, two seeds, both models). This is a quarantined development handover, not an untouched human/session holdout. Original demo0012 feeding examples already occur in prior rehearsal.','',
  '| Model | Preserve completed feeding | Load from the handover |','|---|---:|---:|']
  for m in ['Q0','Q1']:text.append('| %s | %d/2 | %d/2 |'%(m,sum(r['qualified']for r in optional if r['model']==m and r['task']=='feeding'),sum(r['qualified']for r in optional if r['model']==m and r['task']=='loading')))
 else:text+=['','The repair gate failed. The optional handover was not executed. No extra updates, additional examples or outcome-selected retries were added to INIT07.']
 text+=['','## Evidence and reproduction','',
 'media/ contains all-case final contact sheets and four fixed-case T/Q0/Q1 comparison videos derived from saved episode frames. Video time follows simulation capture time; model latency is measured separately in the episode records. The current block geometry is the inherited provisional canonical proxy with the v2 blue-2 visual asset; this run does not establish physical calibration or real-robot reliability.','',
 'Read-only scoring and media regeneration from this worktree:','',
 '    .venv-sim/bin/python scripts/init07_analyse.py',
 '    TYPHON_INIT07_ANALYSIS_OUTPUT=/var/tmp/init07-independent-check .venv-sim/bin/python scripts/init07_analyse.py',
 '    .venv-sim/bin/python scripts/init07_media.py',
 '    .venv-sim/bin/python scripts/init07_report.py','',
 'Original execution: scripts/init07_prepare.py, scripts/init07_train.py gate, then scripts/init07_train.py train --variant Q0, followed by scripts/init07_driver.py. The driver starts two evaluation lanes, Q1 training, its two evaluation lanes and only the preregistered optional branch if eligible. These commands are recorded for provenance; completed outputs are protected against accidental overwrite. A new training reproduction requires a fresh namespace and explicit remapping of the frozen inputs and deadlines.','',
 'Runtime: /var/tmp/typhon-init07-20261015. Worktree: /var/tmp/typhon-worktrees-20261015/init07. Private branch: research/init07-paired-state-repair. Mint archive: /media/nm/Fast Drive/Typhon/research-20261015/init07-paired-state-repair. Large weights, camera/tensor packets and traces are kept outside Git. Dataset manifests retain exact original file hashes and paths; INIT01, INIT02 and INIT06 source archives are required for full reproduction.','',
 'The experiment-level outcome does not end the wider OST programme. Subsequent work must use a separately frozen protocol and preserve this comparison.','']
 report='\n'.join(text)
 (OUT/'INIT07_report.md').write_text(report)
 q=W/'experiment-history/2026-10-15-init07-paired-state-repair';q.mkdir(exist_ok=True);(q/'INIT07_report.md').write_text(report)
 dump(OUT/'report-validation.json',{'passed':True,'core_new_episodes':46,'historical_T_reused':23,'optional_new_episodes':len(optional),'report_sha256':sha(OUT/'INIT07_report.md')})
 print(conclusion)
if __name__=='__main__':main()
