"""Rebuild summaries from saved outcomes; never run policies or modify their evidence."""
from pathlib import Path
import csv,json,math,shutil,hashlib
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/var/tmp/typhon-research-20261012')
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 rows=[];groups=[]
 for experiment in ['factorial','v4-success-anchor']:
  folder=OUT/'loading'/experiment
  protocol=json.loads((folder/'evaluation-protocol.json').read_text());scenarios={s['scenario_id']:s for s in protocol['scenarios']}
  for methoddir in (folder/'evaluation').iterdir():
   if not methoddir.is_dir():continue
   complete=json.loads((methoddir/'complete.json').read_text());assert complete['completed']
   for r in json.loads((methoddir/'summary.json').read_text()):
    episode=json.loads((methoddir/r['id']/'episode.json').read_text());sc=scenarios[r['id']]
    f=sc.get('factors',{});xyz=episode['final_truth']['object_position_m'];oldgoal=[.125,-.12]
    rows.append({'experiment':experiment,'method':methoddir.name,'case':r['id'],'source_demo':sc['source_demo_id'],'start_pose':f.get('start_pose','known_v4_grasp_demo0006'),'task':sc['task'],'seed':sc['seed'],'supported_grasp':r['supported_lift_10mm_3steps'],'stable_released_inside':r['operational_placement'],'strict_success':r['success'],'position_error_mm':r['position_error_mm'],'yaw_error_deg':r['yaw_error_deg'],'final_x_m':xyz[0],'final_y_m':xyz[1],'final_z_m':xyz[2],'distance_to_old_feeding_goal_mm':math.hypot(xyz[0]-oldgoal[0],xyz[1]-oldgoal[1])*1000,'runtime_error':bool(r['errors']),'evidence':str(methoddir/r['id'])})
 by=defaultdict(list)
 for r in rows:by[(r['experiment'],r['method'],r['start_pose'],r['task'])].append(r)
 for keys,rr in by.items():
  groups.append(dict(zip(['experiment','method','start_pose','task'],keys),n=len(rr),grasps=sum(r['supported_grasp'] for r in rr),stable_inside=sum(r['stable_released_inside'] for r in rr),strict=sum(r['strict_success'] for r in rr)))
 with (OUT/'loading/comparisons.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 probe=json.loads((OUT/'loading/goal-probe/summary.json').read_text());tok=json.loads((OUT/'loading/goal-probe/token-retention.json').read_text())
 token_ok=all(r['retains_requested_coordinates'] for r in tok['rows']);assert token_ok
 summary={'complete':True,'physical_rollouts':len(rows),'runtime_errors':sum(r['runtime_error'] for r in rows),'groups':groups,'offline_goal_only_queries':len(probe)*2,'max_horizontal_predicted_endpoint_shift_mm':max(math.hypot(*r['endpoint_shift_xyz_mm'][:2]) for r in probe),'requested_horizontal_target_shift_mm':math.hypot(5,270),'target_coordinates_retained_after_tokenization':token_ok,'interpretation':'Broad12k shows source-start grasp sensitivity and failure to retarget from feeding to loading. This is bounded development evidence, not an estimate of general task B performance. V4 additionally sensitive to loading task package.','v4_training_audit_limit':'18 shared V3 base episodes verified feeding only; exact three extra V4 recovery configs still unavailable. No invented zero-loading claim for unaudited rows.'}
 save(OUT/'loading/validation_summary.json',summary)
 text=['Task B/loading diagnosis — 12 October 2026','',
 'Task A moves the brick into the flat feeding square at (0.125,-0.120)m. Task B moves it into the loading fixture at (0.120,0.150)m, including its raised floor and walls. This is a 270mm destination change, not merely a different name for A.','',
 'Training exposure: V1–V3 declare the same exact source-snapshot hash (ee4f3959456afc2c1a3f5e025be5abbe3fb657fda8881496d6856dae207c39f9). All 18 source configurations are feeding. Broad12k is a separate fresh Pt adaptation on six P_train feeding takes / 439 complete windows, all with the same goal. V4 starts from V3 and adds three recovery takes; their exact configs are not yet recovered. The shared18 are verified; do not overstate the remaining3.','',
 'Matched physical diagnostic: same geometry/cameras/controller/limits; two start poses × two task packages × two seeds. A task package changes language, numeric goal, visible destination geometry, support height and scoring together. This is deliberately not a pure prompt ablation.','']
 for g in groups:text.append(f"{g['experiment']} | {g['method']} | {g['start_pose']} -> {g['task']} | grasp {g['grasps']}/{g['n']} | stable inside {g['stable_inside']}/{g['n']} | strict {g['strict']}/{g['n']}")
 text+=['','What the tests establish:',
 '1. The privileged geometric expert completes all8 factorial executions (four unique physical conditions, repeated seeds). The robot and fixture are feasible under this controller.',
 '2. Broad12k grasps from its familiar start, but on B leaves the brick within6.9–7.8mm of the old feeding goal and about276mm from the requested loading goal. From the historical loading start it fails to grasp for both A and B. Start-pose and destination failures are separately present.',
 '3. Holding images, native proprioception and noise fixed, changing only the supplied task/goal text moves the final predicted horizontal waypoint by at most %.3fmm across9 pairs, despite a270mm requested shift. Actual tokenized inputs retain both sets of coordinates. This establishes weak goal response on those sampled states, not a universal claim about language or recognition.'%summary['max_horizontal_predicted_endpoint_shift_mm'],
 '4. V4 fails to grasp either task on demo0002. On its separately chosen known-grasp demo0006, it grasps2/2 feeding but1/2 loading trials; neither loading trial completes. The loading trial with a grasp finishes near the old feeding square, with176degree yaw error. Loading geometry and instruction remain confounded in these physical trials.',
 '', 'What has not been established: a causal decomposition of numeric language vs target pixels vs fixture collision; statistical generalisation across many B scenes; whether V4 recognises the numeral throughout; benefit from any new loading training.',
 '', 'Recommended next controlled training contrast: use verified robot expert labels on a factorial set of starts and destinations, with the same start paired to multiple goals and both tasks represented. Keep an equal-update fixed-goal control and reserve complete start/goal combinations for evaluation. This tests goal conditioning rather than simply training longer on A. Preserve the no-training current24h repair track until its observer qualification is finished.',
 '', 'Scores: supported grasp requires bilateral contact and10mm lift for3 consecutive20Hz steps. Stable inside requires footprint, upright face, placement height, release and stability. Strict adds15mm centre and10degree directed yaw tolerance. Operational observer evidence is not strict certification.',
 '', 'Limitations: two seeds are not independent demonstrations; one chosen familiar parent in the factorial plus one V4 development anchor. Canonical50×35×30mm/80g provisional proxy retained to match training; this is not the catalogued physical DUPLO dimensions. RGB policies use simulated camera inputs with generated expert labels. V4 retains its original privileged causal held-marker input; broad12k has no object truth input.',
 '', 'Evidence: comparisons.csv, validation_summary.json, training-source-audit.json, factorial/evaluation-protocol.json, goal-probe/protocol.json, goal-probe/token-retention.json, v4-success-anchor/evaluation-protocol.json. Every trial retains action tensors, controller trace, trajectory, initial/final frames and video.',
 '', 'Reproduction (from repository root, existing environments; output paths refuse overwrite):',
 '.venv-sim/bin/python scripts/exp20261012_loading_factorial.py prepare',
 '.venv-sim/bin/python scripts/exp20261012_loading_factorial.py evaluate --method broad-pt-12000',
 '.venv-sim/bin/python scripts/exp20261012_loading_factorial.py evaluate --method v4-m6',
 '.venv-sim/bin/python scripts/exp20261012_loading_factorial.py evaluate --method expert',
 '.venv-sim/bin/python scripts/exp20261012_loading_goal_probe.py',
 '.venv-sim/bin/python scripts/exp20261012_loading_v4_anchor.py',
 '.venv-sim/bin/python scripts/exp20261012_report.py',
 'Check CLI --help against the code snapshot; frozen result directories must never be deleted to rerun. Choose a new output root in a copied script or explicit new revision.']
 (OUT/'loading/REPORT.txt').write_text('\n'.join(text)+'\n')
 from PIL import Image,ImageDraw,ImageFont
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
 canvas=Image.new('RGB',(1024,1170),'#15191d');draw=ImageDraw.Draw(canvas)
 for i,(method,task,label) in enumerate([('broad-pt-12000','feeding','Broad12k / feeding: strict success'),('broad-pt-12000','loading','Broad12k / loading: returns to feeding square'),('expert','feeding','Expert / feeding: strict success'),('expert','loading','Expert / loading: strict success')]):
  folder=OUT/'loading/factorial/evaluation'/method/f'b01-familiar-{task}-seed7417';im=Image.open(folder/'final.png').crop((0,0,512,512));x=(i%2)*512;y=(i//2)*585+60;canvas.paste(im,(x,y));draw.text((x+10,y-42),label,font=font,fill='white')
 canvas.save(OUT/'loading/final-placement-comparison.png')
 code=OUT/'code';code.mkdir(exist_ok=True)
 for f in (ROOT/'scripts').glob('exp20261012_*.py'):shutil.copy2(f,code/f.name)
 print(json.dumps(summary))
if __name__=='__main__':main()
