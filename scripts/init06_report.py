"""Recompute INIT06 admission from saved commands, inputs and traces. No physics."""
import os,sys,json,csv,hashlib,time,io
from pathlib import Path
W=Path(__file__).resolve().parents[1];os.chdir(W);sys.path[:0]=[str(W),str(W/'scripts')]
import numpy as np
from scipy.spatial.transform import Rotation
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.adaptation.actions import absolute_target
O=Path('/var/tmp/typhon-init06-20261015')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def contact(r):return bool(r['object_truth']['grasp_contact'])or any('gripper'in str(c)and'cube'in str(c)for c in r['contacts'])
def intervals(a):
 out=[];start=None
 for i,v in enumerate(list(a)+[False]):
  if v and start is None:start=i
  elif not v and start is not None:out.append([start,i-1]);start=None
 return out
def main():
 complete=read(O/'complete.json');protocol=read(O/'protocol.json');rows=[];banks=[];contrasts=[]
 for s in protocol['parents']:
  parent=s['parent'];parts={}
  for branch in ['F-preserve','L-teacher']:
   base=O/'continuations'/parent/branch;d=base/'episode'
   if not d.exists():continue
   ep=read(d/'episode.json');reported=read(d/'qualification-final.json');re=read(base/'reconstruction/reconstruction.json')
   acts=[json.loads(x)for x in(d/'robot_actions.jsonl').read_text().splitlines()]
   rich=read(d/'controller-rich.json');states=[x['post_action_truth']for x in acts];assert len(rich)==len(acts)
   initial=np.asarray(ep['initial_truth']['object_position_m'])
   rgb=np.load(d/'candidate.npz');targets=rgb['targets'];idx=rgb['frame_action_indices'];prop=rgb['proprio']
   assert np.array_equal(targets,np.asarray([x['encoded_target']for x in acts],np.float32))
   assert np.array_equal(idx,np.arange(0,len(acts),4))
   with np.load(Path(s['folder'])/'predictions.npz')as z:np.testing.assert_array_equal(prop[0],z['histories'][1])
   with np.load(Path(s['folder'])/'policy_io.npz')as q:initial_quat=q['eef_quat_xyzw'][1].copy()
   for j,a in enumerate(acts):
    before=ep['initial_truth']if j==0 else states[j-1]
    prepos=before['eef_position_m']
    prequat=initial_quat if j==0 else rich[j-1]['eef_quat_xyzw']
    assert a['action_index']==j and abs(a['post_time_s']-a['pre_time_s']-.05)<1e-9
    expected=absolute_target(a['normalized_command'],{'proprio':{'eef_position_m':prepos,'eef_quat_xyzw':prequat}},ep['config']).astype(np.float32)
    np.testing.assert_array_equal(targets[j],expected)
   for frame,j in enumerate(idx):
    if j>0:np.testing.assert_array_equal(prop[frame,:9],targets[j-1,:9])
   full=[int(x)for x in idx if x+30<=len(acts)];assert full==reported['complete_30_action_windows']
   maxdisp=float(np.linalg.norm(np.array([x['object_position_m']for x in states])-initial,axis=1).max())
   touches=any(contact(x)for x in rich)
   lifts=[x['object_truth']['bilateral_grasp_contact']and x['object_truth']['object_position_m'][2]-initial[2]>=.07 and x['object_tilt_deg']<=np.degrees(ep['config']['evaluation']['upright_tolerance_rad'])for x in rich]
   ir=intervals(lifts);lift=any(b-a+1>=3 for a,b in ir)
   scored=score_episode(ep['initial_truth'],ep['final_truth'],states[-24:],ep['config'],ep['attempts'])
   assert scored==ep['evaluation']
   minmargin=min(x['nearest_joint_limit_rad']for x in rich)
   nonconv=sum(not x['converged']for a in ep['attempts']for x in a['phases'])
   preserve=scored['success']and not touches and maxdisp<=.001 and len(acts)==60
   clean=scored['success']and lift and nonconv==0 and ep['parking']['converged']and minmargin>=0 if branch=='L-teacher'else preserve
   admitted=bool(clean and not ep['controller_exception']and read(d/'input-qualification.json')['passed'])
   assert admitted==reported['qualified']
   assert ep['instruction']==read(d/'input-request.json')['instruction']
   assert re['historical_first30_rich_trace_max_abs_difference']==0
   budget=60 if branch=='F-preserve'else 1430;assert len(acts)<=budget
   row={'parent':parent,'branch':branch,'admitted':admitted,'strict_endpoint':scored['success'],'actions':len(acts),'complete_H30_windows':len(full),'position_error_mm':scored['position_error_m']*1000,'yaw_error_deg':scored['orientation_error_deg'],'supported_upright_70mm_3samples':lift,'lift_samples':sum(lifts),'longest_lift_samples':max([b-a+1 for a,b in ir]or[0]),'maximum_block_displacement_mm':maxdisp*1000,'any_gripper_object_contact':touches,'minimum_joint_margin_rad':minmargin,'phase_nonconvergence':nonconv,'directory':str(d)}
   rows.append(row);parts[branch]={'row':row,'targets':targets.copy(),'indices':idx.copy(),'states':states}
   banks.append({'id':parent+'-q1-'+branch,'parent':parent,'policy_seed':912171,'query':1,'source_episode':s['folder'],'source_sha256':s['source_sha256'],'path':str(d/'candidate.npz'),'sha256':sha(d/'candidate.npz'),'instruction':ep['instruction'],'config':ep['config'],'task':'feeding'if branch=='F-preserve'else'loading','views':{'primary':'primary_visible','secondary':'secondary_visible'},'observations':'physical q1 followed by actual pre-action renders; inherited previous-target proprio, measured pose separate','labels':'privileged expert or measured-pose hold robot commands; not human motion labels','admitted':admitted,'complete_H30_windows':full,'no_terminal_padding':True})
  if len(parts)==2:
   a,b=[O/'continuations'/parent/k/'reconstruction'for k in ['F-preserve','L-teacher']]
   assert(a/'q1-state.pkl').read_bytes()==(b/'q1-state.pkl').read_bytes()
   with np.load(a/'actual-prepared.npz')as x,np.load(b/'actual-prepared.npz')as y:
    for k in x.files:
     if k!='input_ids':np.testing.assert_array_equal(x[k],y[k])
   f,l=parts['F-preserve']['targets'],parts['L-teacher']['targets'];n=min(len(f),len(l));dist=np.linalg.norm(l[:n,:3]-f[:n,:3],axis=1)*1000
   first=np.flatnonzero(dist>1.)
   changing=np.flatnonzero(np.linalg.norm(np.array([t['object_position_m']for t in parts['L-teacher']['states']])-np.asarray(parts['L-teacher']['states'][0]['object_position_m']),axis=1)>.001)
   contrasts.append({'parent':parent,'paired_complete_state_exact':True,'paired_nontext_inputs_exact':True,'compared_prefix_actions':n,'first_absolute_position_label_difference_over1mm':int(first[0])if len(first)else None,'prefix_max_label_difference_mm':float(dist.max()),'first_loading_object_displacement_over1mm_action':int(changing[0])if len(changing)else None,'loading_full_windows_containing_label_difference':[(int(i),int(i+29))for i in parts['L-teacher']['indices']if i+30<=len(l)and any(i<=int(j)<=i+29 for j in first)],'label_difference_threshold_descriptive_not_admission':True})
 sources={p:d for s in protocol['parents']for p,d in s['source_sha256'].items()}
 changed=[p for p,d in sources.items()if sha(p)!=d];assert not changed
 pairs=sum(all(x['admitted']for x in rows if x['parent']==p['parent'])and len([x for x in rows if x['parent']==p['parent']])==2 for p in protocol['parents'])
 assert len(rows)==complete['continuations']and pairs==complete['qualified_pairs']
 dump(O/'candidate-manifest.json',{'experiment':'INIT06','candidate_only':True,'active_training_manifest_modified':False,'training':False,'new_policy_queries':0,'episodes':banks,'qualified_pairs':pairs})
 dump(O/'behavioral-contrasts.json',contrasts)
 with(O/'comparisons.csv').open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 v={'passed':True,'continuations':len(rows),'qualified_pairs':pairs,'source_files_checked':len(sources),'original_source_files_changed':changed,'q1_pair_state_exact':True,'actual_prepared_nontext_exact':True,'new_policy_queries':0,'new_optimizer_updates':0,'same_unchanged_expert':True,'T0':protocol['clock']['t0'],'execution_completed_epoch':complete['epoch'],'validated_epoch':time.time(),'technical_preparation_failure_preserved':'float64 diagnostic calculation differed8.503e-8degrees; restored archived float32 dtype, no tolerance/physics change'}
 dump(O/'validation_summary.json',v)
 table='| Parent | Branch | Qualified | Endpoint | H30 windows | Position mm | Yaw deg | 70mm lift | Min joint margin rad |\n|---|---|---|---|---|---|---|---|---|\n'
 for r in rows:table+='| '+ ' | '.join([r['parent'],r['branch'],str(r['admitted']),str(r['strict_endpoint']),str(r['complete_H30_windows']),f"{r['position_error_mm']:.3f}",f"{r['yaw_error_deg']:.3f}",str(r['supported_upright_70mm_3samples']),f"{r['minimum_joint_margin_rad']:.6f}"])+' |\n'
 report=f"""# INIT06 — two instructions at the actual learner-reached state

{pairs} of 2 prescribed parent pairs qualified; {len(rows)} continuations were executed. This qualifies candidate supervision, not a trained repair.

{table}
## What was compared
Each branch independently replayed the original qualified feeding prefix and the same archived T first30 commands, stopping before query1. No poses were injected, controller reset, new policy query made, or online image/pose clamp applied. Both available archived rich traces and raw RGB pixels reconstructed exactly. Actual prepared O2 inputs matched original hashes. First recorded proprioception retained the actual previous predicted target; later records use the existing command-history convention.

The q0 complete semantic state was archived and matched exactly. Full internal state at historical q1 was not separately saved: historical qualification uses deterministic replay plus all available rich fields and actual model inputs. Complete newly recorded q1 states match exactly across paired branches, including velocity, controller, warm-start and contacts under the existing comparator.

Feeding uses the measured full pose captured once and open gripper for60ticks. Loading uses unchanged nominal expert from that physical state, at most1430remainingticks including24settle. Original instructions are retained. Hold success is a3second teacher reference, not full-horizon learned preservation.

## Data contract
Each trajectory contains real rendered robot observations, exact executed normalized commands, qualified absolute EEF targets, previous-target proprioception, source identity and full diagnostic traces. Complete H30 windows only; no terminal padding or shifted labels. Simulator object truth is used for the declared loading expert and independent scorer, never added to learner images, prompt or proprioception. Hidden-camera arrays inherited from the recorder are retained for audit but the candidate manifest selects visible views only.

Both parents belong to the eligible training families. Their geometry is tightly concentrated and they are not independent broad scene diversity. Personal footage supplied scenario provenance, not measured Franka action labels.

## Failures and limitations
One preparation attempt stopped on a rotation diagnostic difference8.503e-8degrees caused by reconstructing its target asfloat64. A saved offline calculation reproduced that discrepancy and showed the archived target wasfloat32. Restoring that dtype left commands and thresholds unchanged. Failed attempt remains archived. No continuation outcome was retried.

Any missing parent after a failed admission gate remains unexecuted, not an unsuccessful rollout. Candidate admission requires strict endpoint, clean phase convergence, no observed joint-limit excursion and three consecutive upright bilateral70mm lift samples for loading; preservation requires no contact and <=1mm maximum block movement.

## Reproduction
Frozen executable commit: {(O/'frozen-commit.txt').read_text().strip()}.
Environment: existing .venv-sim for physics and scoring; .venv-xvla only for CPU native preprocessing. No weights loaded.
Commands used:
```sh
.venv-sim/bin/python scripts/init06.py prepare
.venv-sim/bin/python scripts/init06.py run
.venv-sim/bin/python scripts/init06_report.py
```
Physics commands deliberately refuse existing episode directories. Preserve the frozen outputs and use a separately declared fresh root for an actual rerun. The report command rechecks saved outputs without physics or model calls.
"""
 (O/'REPORT.md').write_text(report)
 if pairs==2:
  (O/'REPAIR_PROPOSAL.md').write_text("""# Candidate proposal; training not launched
Initialize both control and repair from the same frozen T checkpoint. Keep identical model, optimizer, step count, random seeds, common rehearsal sample IDs and observation/action conventions. Preserve source F/L skills with the same existing rehearsal bank. Reserve an equal fixed exposure budget: control reuses the original q0 paired preservation/loading references; repair replaces only those selected exposures with the qualified q1 paired references, balancing both task instructions and both parent families. Predeclare which timestamps/windows are sampled; do not oversample a long loading suffix just because it has more windows.

Suggested budget for planner review:3000full-weight updates per branch, same actual effective batch4, matched ordering/noise; checkpoint at3000 only. This is continued adaptation of T, not foundation pretraining. No automatic launch is authorized by this data protocol.

Compare frozen T, control, repair on the same INIT02 full-horizon preservation8 and new-loading8 cells plus original feeding3/loading4 checks. Preserve both policy seeds, no harness/TTC, original endpoint and contact/lift metrics, and record every trial. Reuse verified frozen-T outcomes where initial contract/source hashes match; do not call that a newly rerun baseline. Preparation should also determine whether the paired bank has enough temporal coverage;60ticks alone does not establish72second inactivity. Success must retain loading while preserving completed feeding across the full horizon. No quarantine imports. No language wording change in the same experiment.

Storage: one new checkpoint fits current scratch; archive and verify each branch before considering offload. Keep active T and all frozen original data. Planner must settle allocation and timeline before training.
""")
 print(json.dumps(v),flush=True)
if __name__=='__main__':main()
