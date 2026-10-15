"""Independent saved-trace scoring for INIT01; no inference or physics."""
import os,sys,json,gzip,time,copy,csv,hashlib,itertools
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont
from supervisor.feeding_loading.evaluation import score_episode
O=Path('/var/tmp/typhon-init01-20261015')
A=Path(os.environ.get('TYPHON_INIT01_ANALYSIS_OUTPUT',str(O)))
A.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
p=read(O/'protocol.json');assert read(O/'complete.json')['completed'];assert read(O/'input-and-label-audit.json')['passed']
rows=[];bank=[];poses=[]
for parent in p['parents']:
 base=O/'continuations'/parent['id'];states=[(base/b/'prefix/terminal-state.pkl').read_bytes() for b in p['branches']]
 assert len(set(states))==1 and states[0]==(O/'qualification'/parent['id']/'terminal-state.pkl').read_bytes()
 pre=read(base/'L-policy/prefix/prefix-qualification.json');poses.append({'parent':parent['id'],**pre['terminal_truth']})
 paired={}
 for branch in p['branches']:
  d=base/branch/'episode';ep=read(d/'episode.json');cfg=parent['entry']['config'] if branch=='F-preserve' else parent['physical_config']
  score=score_episode(ep['initial_truth'],ep['final_truth'],ep['stability_truths'],cfg,ep['attempts'])
  assert score==ep['evaluation']
  if branch=='L-policy':trace=[json.loads(x) for x in gzip.open(d/'controller_trace.jsonl.gz','rt')]
  else:trace=read(d/'controller-rich.json')
  truth=[x['object_truth'] for x in trace];ini=ep['initial_truth'];eef=np.asarray([ini['eef_position_m']]+[x['eef_xyz'] for x in trace]);obj=np.asarray([t['object_position_m'] for t in truth])
  rise=obj[:,2]-ini['object_position_m'][2]
  supported=[bool(t['bilateral_grasp_contact'] and z>=.07 and x['object_tilt_deg']<=10) for x,t,z in zip(trace,truth,rise)]
  longest=0;run=0
  for hit in supported:run=run+1 if hit else 0;longest=max(longest,run)
  row={'parent':parent['id'],'condition':branch,'strict_success':score['success'],'close_command_steps':sum(x['command'][-1]>0 for x in trace),'contact_steps':sum(t['grasp_contact'] for t in truth),'bilateral_contact_steps':sum(t['bilateral_grasp_contact'] for t in truth),'max_height_gain_mm':float(rise.max()*1000),'upright_bilateral_lift_70mm':any(supported),'longest_upright_supported_lift_s':longest*.05,'min_eef_object_distance_mm':float(np.linalg.norm(eef[1:]-obj,axis=1).min()*1000),'object_transport_mm':float(np.linalg.norm(obj[-1,:2]-np.asarray(ini['object_position_m'])[:2])*1000),'position_error_mm':score['position_error_m']*1000,'yaw_error_deg':score['orientation_error_deg'],'tilt_deg':float(np.degrees(score['upright_error_rad'])),'eef_path_length_m':float(np.linalg.norm(np.diff(eef,axis=0),axis=1).sum()),'eef_max_displacement_mm':float(np.linalg.norm(eef-eef[0],axis=1).max()*1000),'joint_limit_excursion_steps':sum(x['nearest_joint_limit_rad']<0 for x in trace),'maximum_joint_limit_excursion_rad':float(max(0.,-min(x['nearest_joint_limit_rad'] for x in trace))),'control_steps':len(trace),'simulated_seconds':len(trace)*.05,'directory':str(d)}
  rows.append(row);paired[branch]=row
  if branch!='L-policy':
   q=read(d/'qualification.json');assert q['strict_success']==row['strict_success'] and q['upright_bilateral_lift_70mm']==row['upright_bilateral_lift_70mm']
   bank.append({'id':parent['id']+'-'+branch,'parent_trajectory':parent['entry']['id'],'source_demo_id':parent['id'],'parent_actions_sha256':parent['actions_sha256'],'parent_npz_sha256':parent['entry']['sha256'],'handover_state_sha256':pre['semantic_state_sha256'],'condition':branch,'task':cfg['task']['type'],'instruction':ep['instruction'],'absolute_path':str(d/'candidate.npz'),'sha256':hashlib.sha256((d/'candidate.npz').read_bytes()).hexdigest(),'complete_30_action_windows':q['complete_30_action_windows'],'candidate_positive_loading':q['candidate_positive'],'valid_preservation_reference':bool(branch=='F-preserve' and row['strict_success'] and row['contact_steps']==0),'joint_limit_excursion_steps':row['joint_limit_excursion_steps'],'phase_nonconvergence':q['phase_nonconvergence'],'status':'candidate only; not trained','action_origin':'generated privileged Panda loading expert or supplied open-gripper hold, not measured human actions'})
 for e in bank:
  if e['source_demo_id']==parent['id']:e['native_policy_strict_success']=paired['L-policy']['strict_success'];e['native_policy_supported_lift']=paired['L-policy']['upright_bilateral_lift_70mm']
xy=np.asarray([x['object_position_m'][:2] for x in poses]);yaw=np.asarray([x['object_yaw_rad'] for x in poses])
dist=[float(np.linalg.norm(xy[i]-xy[j])*1000) for i,j in itertools.combinations(range(4),2)]
spread={'parents':poses,'x_range_mm':float(np.ptp(xy[:,0])*1000),'y_range_mm':float(np.ptp(xy[:,1])*1000),'maximum_pairwise_xy_mm':max(dist),'yaw_range_deg':float(np.degrees(np.ptp(yaw))),'independent_sessions_not_established':True,'interpretation':'four training parent IDs converging on the same feeding station; not four broad geometric regimes'}
rescues=[e['source_demo_id'] for e in bank if e['candidate_positive_loading'] and not e['native_policy_strict_success'] and any(z['source_demo_id']==e['source_demo_id'] and z['valid_preservation_reference'] for z in bank)]
manifest={'candidate_only':True,'trained':False,'active_training_manifest_modified':False,'source_split':'four LOAD04 training families only','entries':bank,'eligible_rescued_parents':rescues,'engineering_admission_gate_passed':len(rescues)>=2,'independent_generalisation_claim':False,'pose_spread':spread}
dump(A/'candidate-manifest.json',manifest);dump(A/'independent-scores.json',rows)
with (A/'comparisons.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'passed':True,'prefixes_reproduced':4,'physical_continuations':12,'native_loading_successes':sum(r['strict_success'] for r in rows if r['condition']=='L-policy'),'qualified_teacher_loading':len(rescues),'preservation_successes':sum(r['strict_success'] for r in rows if r['condition']=='F-preserve'),'candidate_training_admission':len(rescues)>=2,'new_training':False,'new_holdout_families':0,'pose_spread':{k:v for k,v in spread.items() if k!='parents'},'input_label_audit_passed':True,'all_matched_branch_semantic_states_exact':True,'epoch':time.time()}
dump(A/'validation_summary.json',summary)
font=ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',16)
sheet=Image.new('RGB',(1560,4*315+55),'#10141c');draw=ImageDraw.Draw(sheet);draw.text((15,12),'INIT01: final states | FL loading | unchanged loading expert | completed-feeding hold',font=font,fill='white')
for i,parent in enumerate(p['parents']):
 for j,b in enumerate(p['branches']):
  d=O/'continuations'/parent['id']/b/'episode';im=Image.open(d/'final.png').crop((0,0,512,512)).resize((280,280))
  x=j*520;y=55+i*315;sheet.paste(im,(x,y));row=next(x for x in rows if x['parent']==parent['id'] and x['condition']==b)
  label=f"{parent['id']} {b}\nstrict={row['strict_success']}\ncontact={row['contact_steps']}\nerror={row['position_error_mm']:.1f} mm"
  draw.multiline_text((x+285,y+20),label,font=font,fill='white',spacing=7)
sheet.save(A/'final-contact-sheet.png')
print(json.dumps(summary))
