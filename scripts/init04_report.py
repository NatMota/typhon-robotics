"""Recompute INIT04 output measurements and render its bounded evidence report."""
import os,sys,json,time,gzip,hashlib,csv,shutil
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path[:0]=[str(W),str(W/'scripts'),str(W/'.m5-deps'),str(W/'.m6-deps')];os.chdir(W)
import numpy as np,torch
from init04_prepare import O,read,dump,sha
from init04_infer import metrics
def main(dest):
 dest.mkdir(parents=True,exist_ok=True)
 protocol=read(O/'protocol.json');onset=read(O/'onset-audit.json');complete=read(O/'complete.json');qualification=read(O/'qualification.json');clock=read(O/'clock.json')
 assert onset['complete'] and complete['passed'] and qualification['passed']and qualification['max_abs_action_error']==0
 assert complete['distinct_outputs']==18 and complete['fresh_calls']==16 and complete['archived_reuses']==2
 packets=torch.load(O/'packets.pt',weights_only=False,map_location='cpu');measurements=read(O/'measurements.json')
 with np.load(O/'outputs.npz')as outputs:
  for r in measurements:
   a=outputs[r['id']];p=packets[r['query']];recomputed=metrics(a,p)
   assert all(r[k]==v for k,v in recomputed.items())
   assert r['close_count']==int((a[:,9]>.5).sum())
   assert np.isclose(r['xyz_departure_supplied_max_mm'],max(np.linalg.norm(x[:3]-p['native_history_executor'][:3])*1000 for x in a),atol=2e-5)
 for file,h in protocol['source_sha256'].items():assert sha(file)==h
 checks={'passed':True,'output_chunks_recomputed':len(measurements),'unchanged_source_files':len(protocol['source_sha256']),'source_manifest_sha256':sha(O/'protocol.json'),'clock_gates':{'prepare':protocol['freeze_epoch']<clock['preparation_deadline'],'qualification':qualification['epoch']<clock['qualification_deadline'],'matrix':complete['epoch']<clock['matrix_deadline']},'original_reproduction_max_error':0.0,'model_state_unchanged':complete['model_state_sha256_before']==complete['model_state_sha256_after'],'new_physics':0,'new_optimizer_updates':0}
 dump(dest/'validation_summary.json',checks)
 with (dest/'measurements.csv').open('w')as f:
  wr=csv.DictWriter(f,fieldnames=list(measurements[0]));wr.writeheader();wr.writerows(measurements)
 lines=['# INIT04 — unwanted initiation and same-state prompt contrast','',
'The feeding request did not change before the unwanted grasp in any of the eight saved failures. In the selected later state, explicit instructions to hold position and keep the gripper open still produced approach and closing commands. This local prompt change did not repair the proposal; it does not prove that the model ignores all language.','',
'## Execution and qualification','',
'The unchanged INIT02 T checkpoint was used: '+protocol['weight_sha256']+'. Current O2 views and native command-history proprioception were retained. There is no temporal image stack or elapsed-time input. No physics rollout, optimizer update or R model load was performed.','',
'Three packets (queries 0, 3 and 4) × three instructions × two saved noises produced 18 distinct outputs: 16 fresh calls, including one exact original-condition replay, plus two qualified archived reuses. The fresh replay had zero maximum absolute action error. Full parameter/buffer hashes matched before and after inference, and the checkpoint file hash was unchanged. Both gripper input channels still undergo native zeroing; output probabilities retain the native sigmoid and >0.5 closing threshold.','',
'The deterministic post-hoc mechanistic selection chose demo_0001, seed 912171: the lowest parent and seed with verified supported lift. This is a diagnosis of an inspected failure, not an independent held-out success estimate.','',
'## Onset in all eight preservation runs','',
'Times below are relative to the feeding request and taken from post-step samples at 20 Hz. Query and waypoint indices are zero based. Lift time is the start of the first ≥3-sample run with bilateral contact and block centre ≥10 mm above its initial resting height; the third sample confirms it.','',
'| Parent | Seed | First executed close (s; query) | First contact (s; query) | Supported-lift run begins (s) |',
'|---|---:|---:|---:|---:|']
 for r in onset['rows']:
  def ev(name):
   x=r[name];return 'absent'if x is None else f"{x['seconds_since_request']:.2f}; q{x['query']}"
  l=r['first_supported_lift_run_start'];lines.append(f"| {r['parent']} | {r['seed']} | {ev('first_close')} | {ev('first_contact')} | {'absent' if l is None else format(l['seconds_since_request'],'.2f')} |")
 lines+=['','All eight retain the same full serialized feeding instruction and token hash through initiation. Input reconstruction exactly matches the saved hash of every field at every inspected query. The only task-related scalar is the unchanged domain ID 29; the task goal is serialized in the text. No default loading request or truncated relevant suffix was found.','',
'The selected run makes a small initial departure, then the proposed maximum departure from supplied pose grows from 11.6 mm at q0 to 127.4 mm at q1, 310.9 mm at q2 and 164.4 mm at q3. At q3 it proposes getting within 25.1 mm of the block; q4 first commands closing at waypoint 8 (6.45 s). Contact follows at 7.85 s and the first supported-lift run begins at 9.00 s.','',
'The object crop becomes invalid at q3, after the larger q1–q2 approach proposals; this timing does not establish the crop loss as the cause of initiation. The supplied-versus-measured wrist gap is 0.64 mm at q1, 20.90 mm at q2, 8.12 mm at q3 and 10.41 mm at q4. The gap may matter, but no history or measured-pose substitution was tested.','',
'## Same-state instruction comparison','',
'F-original and L-original are the exact archived feeding and loading requests. H-explicit is:',
'> '+protocol['prompts']['H-explicit'],'',
'H-explicit supplies a no-motion primitive; it does not supply a completion label or demonstrate autonomous recognition of task completion. The original protocol has no duration field to preserve. All nontext inputs, including command history, image masks and current camera views, are identical across the three conditions at each query. The two noises are the original query noise and the corresponding query-index noise from the other original seed.','',
'Ranges below span the two paired noises. Distances are proposed targets, not newly executed motions. The block location is used only by the offline scorer.','',
'| Query / scene time | Instruction | Max XYZ departure from supplied pose | Max departure from measured pose | Min proposed block distance | First close waypoint |',
'|---|---|---:|---:|---:|---:|']
 def bounds(rs,k):
  v=[x[k]for x in rs];lo=min(v);hi=max(v);return f'{lo:.1f}'if abs(lo-hi)<.05 else f'{lo:.1f}–{hi:.1f}'
 for q in protocol['selected_queries']:
  for c in ['F-original','L-original','H-explicit']:
   rs=[x for x in measurements if x['query']==q and x['condition']==c]
   close=sorted(set(x['first_close_waypoint']for x in rs),key=lambda v:-1 if v is None else v);close=', '.join('none'if x is None else str(x)for x in close)
   lines.append(f"| q{q} / {q*1.5:.1f} s | {c} | {bounds(rs,'xyz_departure_supplied_max_mm')} mm | {bounds(rs,'xyz_departure_measured_max_mm')} mm | {bounds(rs,'min_proposed_object_distance_mm')} mm | {close} |")
 lines+=['',
'At q3, all three requests still propose about 164 mm of movement toward the block with no closing yet. At q4, F closes for 22 of 30 waypoints, while L and H close for 21 of 30, beginning one waypoint later. H has peak closing probability approximately 0.998 under both noises. Stronger wording did not suppress acquisition in this local contrast.','',
'The instructions produce measurable numerical differences above the zero replay-error floor. For H versus F at q4, mean XYZ difference is 7.18–7.60 mm and mean absolute gripper-probability difference is approximately 0.0387. Those differences do not amount to the requested hold behaviour. At q0 every condition has roughly 11–12 mm maximum departure and no close: closer to a hold label does not mean stationary. Full wrist-rotation measurements and first/last/max values are retained in measurements.csv.','',
'## Coverage and interpretation','',
'Each preservation demonstration contains 60 control ticks (3 s), 15 saved observations, and zero supplied-XYZ range. Query0 has an exact processed-image and pose match in that reference. In the selected run, later effective inputs differ: nearest recorded hold-pose distance is 9.80 mm at q1, 117.0 mm at q2, 397.3 mm at q3 and 471.3 mm at q4. Processed-image RMSE to the nearest hold frame is respectively 0.173, 0.425, 1.373 and 1.081 in the processor’s normalized tensor units. These are descriptive input differences, not physical calibration or an OOD threshold.','',
'Elapsed time alone is not the mismatch: time is not an actor input. The actual images and supplied poses evolve away from the stationary hold data. This supports state-dependent initiation while leaving the separate causal contributions of imagery, command-history feedback and learned conditioning unresolved.','',
'No request-interface defect was found and the explicit hold prompt did not repair the tested late response. Native conditional fitting remains the next unresolved diagnostic; INIT03 Part B is still unexecuted. This work makes no new physical-success, release-repair, deployment, blue-2 recognition or general language-capability claim.','',
'## Reproduction and retained evidence','',
'The frozen execution code is in commit 6eb9897c7a4b1d3ebb33fe0082d7ab8d0a976296. Raw packets, source/noise hashes, exact prompts, all outputs, per-query onset records and replay arrays are archived together. Original INIT01/INIT02 outputs remain unchanged.','',
'Recompute this report and its checks into a new directory without any model or physics execution:',
'~~~sh',
'.venv-xvla/bin/python scripts/init04_report.py /var/tmp/typhon-init04-report-reproduced',
'~~~',
'The original bounded preparation and native generation commands were:',
'~~~sh',
'.venv-xvla/bin/python scripts/init04_prepare.py',
'.venv-xvla/bin/python scripts/init04_infer.py',
'~~~',
'Those commands enforce the recorded original time gates; they are not a request to rerun or overwrite the frozen experiment. A new execution requires a fresh output root and explicit recorded clock.','',
'The numerical contact sheet is generated from saved outputs only, using a separate plot dependency directory. Its pinned dependencies are recorded in plot-requirements.txt; inference environments were not modified. Recreate it with:\n~~~sh\n.venv-xvla/bin/python -m pip install --no-cache-dir --target /var/tmp/typhon-plotdeps-20261015 -r experiment-history/2026-10-15-init04-initiation-onset/plot-requirements.txt\n.venv-xvla/bin/python scripts/init04_plot.py /var/tmp/typhon-init04-20261015\n~~~\n\n![Onset and instruction contrasts](diagnostic.png)']
 (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(checks),flush=True)
if __name__=='__main__':main(Path(sys.argv[1])if len(sys.argv)>1 else O)
