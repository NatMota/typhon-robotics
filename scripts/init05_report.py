"""Recompute INIT05 diagnostics from saved arrays only."""
import os,sys,json,csv,time
from pathlib import Path
W=Path(__file__).resolve().parents[1];os.chdir(W);sys.path[:0]=[str(W),str(W/'scripts'),str(W/'.m5-deps'),str(W/'.m6-deps')]
import numpy as np,torch
from init05_prepare import O
from init04_prepare import read,dump,sha
from init05_infer import metrics
from load04_common import array_sha
def main(dest):
 dest.mkdir(parents=True,exist_ok=True)
 protocol=read(O/'protocol.json');clock=read(O/'clock.json');qual=read(O/'qualification.json');done=read(O/'complete.json')
 assert qual['passed']and qual['max_abs_action_error']==0 and done['passed']and done['outputs']==len(done['calls'])==16 and done['model_state_before']==done['model_state_after']
 p=torch.load(O/'packets.pt',weights_only=False,map_location='cpu');seen=torch.load(O/'actual-effective-inputs.pt',weights_only=False,map_location='cpu')
 records=read(O/'measurements.json');effects=read(O/'factor-effects.json');byid={r['id']:r for r in records};callbyid={r['id']:r for r in done['calls']}
 with np.load(O/'outputs.npz')as outputs,np.load(O/'whole-sequence-effects.npz')as ve:
  assert len(outputs.files)==len(records)==len(seen)==16
  for r in records:
   a=outputs[r['id']];m=metrics(a,p,r['cell']);assert all(r[k]==v for k,v in m.items())
   assert int((a[:,9]>.5).sum())==r['close_count']
   independent=np.linalg.norm(a[:,:3].astype(float)-p['fixed_measured_q1_xyz'],axis=1).max()*1000
   assert np.isclose(independent,r['xyz_departure_measured_q1_max_mm'],atol=1e-6)
   assert {k:array_sha(v.numpy())for k,v in seen[r['id']].items()}==callbyid[r['id']]['actual_effective_input_hashes']
  for e in effects:
   c=e['instruction'];ni=e['noise'];rows={cell:byid[cell+'-'+c+'-n'+str(ni)]for cell in p['cells']}
   arrays={cell:outputs[cell+'-'+c+'-n'+str(ni)]for cell in p['cells']}
   for m,v in e['scalar_contrasts'].items():
    expected={'image_at_P0':rows['10'][m]-rows['00'][m],'proprio_at_I0':rows['01'][m]-rows['00'][m],'interaction':rows['11'][m]-rows['10'][m]-rows['01'][m]+rows['00'][m]}
    assert v==expected
   vectors={'image_at_P0':arrays['10']-arrays['00'],'proprio_at_I0':arrays['01']-arrays['00'],'interaction':arrays['11']-arrays['10']-arrays['01']+arrays['00']}
   for n,v in vectors.items():assert np.array_equal(v,ve[c+'-n'+str(ni)+'-'+n])
 for name,h in protocol['source_sha256'].items():assert sha(name)==h
 a=p['original_inputs'][0]['F-original']['image_input'][0].numpy();b=p['original_inputs'][1]['F-original']['image_input'][0].numpy()
 changes=[{'view':name,'tensor_rmse':float(np.sqrt(np.mean((a[i]-b[i])**2))),'unequal_scalar_fraction':float(np.mean(a[i]!=b[i])),'exact_same':bool(np.array_equal(a[i],b[i]))}for i,name in enumerate(['primary','secondary','crop'])]
 dump(dest/'image-differences.json',changes)
 checks={'passed':True,'output_measurements_recomputed':16,'effective_input_records_verified':16,'source_files_unchanged':len(protocol['source_sha256']),'original_reproduction_error':0.,'clock_gates':{'prepare':protocol['freeze_epoch']<clock['preparation_deadline'],'qualification':qual['epoch']<clock['qualification_deadline'],'matrix':done['epoch']<clock['matrix_deadline']},'model_state_unchanged':True,'new_physics':0,'new_training':0}
 assert all(checks['clock_gates'].values());dump(dest/'validation_summary.json',checks)
 keys=list(records[0]);withfile=dest/'measurements.csv'
 with withfile.open('w',newline='')as f:
  wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(records)
 lines=['# INIT05 — first-step image/proprioception factorial','',
 'Changing the q0 visual block to q1 is sufficient locally to reproduce the large approach proposal. Changing only the effective proprioception leaves the output close to its initial response. Both trained instruction forms still initiate approach. This identifies visual-block dependence at one inspected failure, not the specific visual cue or a validated repair.','',
 '## Contract and qualification','',
 'The unchanged INIT02-T checkpoint '+protocol['weight_sha256']+' was used. All16 cells were generated: four I/P combinations × feeding/loading × the same two saved q1 noise tensors. The first 11/F/noise0 call reproduced the original q1 output exactly (maximum error0). Actual effective tensors were captured at the native transformer boundary; native preprocessing ran ten times per output and only zeroed the existing gripper inputs. Model parameters/buffers and source records remain unchanged. No new physics or training was performed.','',
 'The selected episode is demo_0001/seed912171, carried forward from the frozen INIT04 post-hoc selection. q0 is task start and q1 the next observation after30 control actions. The two mixed-time cells10 and01 are synthetic input interventions, not physically validated states or deployable policies.','',
 'q0→q1 supplied position changes9.801mm and decoded rotation2.042°. q1 differs from measured wrist by0.643mm and0.114°. Raw6D column norms change from approximately1/1 to1.004892/1.003903; their condition number changes1.000000→1.001020. These differences were retained without normalization or repair. Both image masks are [true,true,true], and the crop bounding boxes agree; image-block effects do not include a validity-mask change.','',
 '## Absolute proposals and common physical reference','',
 'Distances below use the same measured q1 wrist and q1 block for every cell. Ranges are the two paired saved noises, not uncertainty intervals. Absolute first/final targets, full XYZ/rotation sequences and gripper probabilities are retained in outputs.npz and measurements.csv.','',
 '| Image / pose cell | Instruction | Max departure from measured q1 wrist | Minimum proposed block distance | Close waypoints |','|---|---|---:|---:|---:|']
 def rng(rows,key):
  a=min(r[key]for r in rows);b=max(r[key]for r in rows);return f'{a:.2f}'if abs(a-b)<.005 else f'{a:.2f}–{b:.2f}'
 for cell in ['00','10','01','11']:
  for c in ['F-original','L-original']:
   rr=[r for r in records if r['cell']==cell and r['instruction']==c]
   lines.append(f'| {cell} | {c} | {rng(rr,"xyz_departure_measured_q1_max_mm")}mm | {rng(rr,"min_proposed_object_distance_mm")}mm | {rng(rr,"close_count")} /30 |')
 lines+=['',
 '00 and01 remain about7.7–8.0mm from the common wrist reference, and roughly485–486mm from the block. With later images, 10 and11 propose approximately125–143mm of motion and reach roughly362–378mm from the block. These are actual changed absolute XYZ predictions, not merely a moving relative-error origin. No cell proposes closing yet; this is the onset of approach, before the later unwanted grasp documented in INIT04.','',
 '## Prespecified finite contrasts','',
 'The primary metric here is maximum proposed departure from fixed measured q1 wrist. Image effect=m10−m00; pose effect=m01−m00; interaction=m11−m10−m01+m00. The complete prespecified scalar contrasts and full sequence vector contrasts are also saved.','',
 '| Instruction | Noise | Image effect at P0 | Pose effect at I0 | Interaction | Mean absolute XYZ image-effect norm | Mean absolute XYZ pose-effect norm |','|---|---:|---:|---:|---:|---:|---:|']
 for e in effects:
  v=e['scalar_contrasts']['xyz_departure_measured_q1_max_mm'];a=e['absolute_output_contrasts']
  lines.append(f'| {e["instruction"]} | {e["noise"]} | {v["image_at_P0"]:.3f}mm | {v["proprio_at_I0"]:.3f}mm | {v["interaction"]:.3f}mm | {a["image_at_P0"]["xyz_mean_delta_norm_mm"]:.3f}mm | {a["proprio_at_I0"]["xyz_mean_delta_norm_mm"]:.3f}mm |')
 lines+=['',
 'The small interaction and pose-only effects do not establish that proprioception never matters. They concern this q0/q1 pair, these prompts and two fixed noises. Feeding and loading proposals differ numerically under later images, but neither selects preservation. No percentage of causal responsibility or population success rate is inferred.','',
 '## Which visual inputs changed?','',
 '| Processed view | RMSE in normalized tensor units | Fraction of unequal scalars | Exact equality |','|---|---:|---:|---|']
 for r in changes:lines.append(f'| {r["view"]} | {r["tensor_rmse"]:.6f} | {r["unequal_scalar_fraction"]:.6f} | {r["exact_same"]} |')
 lines+=['',
 'These are descriptive differences, not a per-camera causal ablation, pixel localization, recognition result or out-of-distribution threshold. The test swaps the complete visual block; it does not establish whether the arm appearance, a particular camera or another image feature drives the response. It also does not test blue-2 recognition.','',
 '## Next qualified repair direction','',
 'The result meets the planner’s visual-block sufficiency pattern. The next step is to qualify paired preservation and loading labels at a physically valid learner-reached precontact state: holding measured pose with open gripper must preserve the arrangement, while a teacher must still load from that same state. Both records must share the actual aligned initial observation and eligible training-family provenance. This experiment supplies no such new labels itself.','',
 'Do not freeze images online, clamp proprioception, force the gripper open or call this a repaired policy. Native conditional fitting (INIT03 PartB) remains unexecuted; this result does not settle capacity, teacher-forcing or training adequacy.','',
 '## Reproduction','',
 'Frozen execution commit: '+(O/'frozen-commit.txt').read_text().strip()+'. Recompute all numerical checks/report without model or physics calls:',
 '~~~sh','.venv-xvla/bin/python scripts/init05_report.py /var/tmp/typhon-init05-report-reproduced','~~~',
 'Original bounded execution commands (original clocks enforced; do not overwrite frozen outputs):',
 '~~~sh','.venv-xvla/bin/python scripts/init05_prepare.py','.venv-xvla/bin/python scripts/init05_infer.py','~~~',
 'The figure uses the separate pinned plotting environment already recorded by INIT04:',
 '~~~sh','.venv-xvla/bin/python scripts/init05_plot.py /var/tmp/typhon-init05-20261015','~~~','',
 '![Input-factor comparison](diagnostic.png)','']
 (dest/'REPORT.md').write_text('\n'.join(lines))
 print(json.dumps(checks),flush=True)
if __name__=='__main__':main(Path(sys.argv[1])if len(sys.argv)>1 else O)
