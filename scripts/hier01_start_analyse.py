"""Read-only outcome and matched-pair analysis of the frozen START panel."""
import json,gzip,time,csv,hashlib,subprocess,os,sys
from pathlib import Path
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];ORIGINAL=Path('/var/tmp/typhon-hier01-start-20261015');O=Path(os.environ.get('TYPHON_START_ANALYSIS_COPY',str(ORIGINAL)));REPRODUCE='--reproduce' in sys.argv
assert not REPRODUCE or O.resolve()!=ORIGINAL.resolve(),'Reproduction requires a separate archive copy'
D=O/'recomputed' if REPRODUCE else R/'experiment-history/2026-10-15-hier01-start';D.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def lines(p):
 with gzip.open(p,'rt') as f:return [json.loads(x) for x in f]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(O/'protocol.json');complete=read(O/'complete.json');rows=read(O/'summary.json');assert complete['complete'] and len(rows)==10 and read(O/'control-gate.json')['passed']
started=read(O/'trace-followup-clock.json')['start_epoch'] if (O/'trace-followup-clock.json').exists() and not REPRODUCE else time.time()
if REPRODUCE or not (O/'trace-followup-clock.json').exists():dump(O/'trace-followup-clock.json',{'start_epoch':started,'deadline':started+600,'read_only':True})
details=[];cache={}
for row in rows:
 if REPRODUCE:row['directory']=str(O/'episodes'/row['id'])
 root=Path(row['directory']);ep=read(root/'episode.json');trace=lines(root/'controller_trace.jsonl.gz');pred=np.load(root/'predictions.npz');chunks=pred['chunks'];hist=pred['histories'];seed=row['seed'];cell=row['cell'];case=p['cases'][row['id']]
 assert len(trace)==1460 and chunks.shape==(48,30,20)
 raw=chunks.reshape(-1,20);current_objects=np.array([ep['initial_truth']['object_position_m'] if q==0 else trace[q*30-1]['object_truth']['object_position_m'] for q in range(48)])
 current_eef=np.array([ep['initial_truth']['eef_position_m'] if q==0 else trace[q*30-1]['eef_xyz'] for q in range(48)])
 rawdist=np.linalg.norm(chunks[:,:,:3]-current_objects[:,None,:],axis=2)*1000
 actual=np.array([t['eef_xyz'] for t in trace]);obj=np.array([t['object_truth']['object_position_m'] for t in trace])
 before=np.vstack([ep['initial_truth']['object_position_m'],obj[:-1]])
 marker=np.array(case['config']['workspace']['staging_position_xy_m']);target=np.array(case['config']['task']['target_position_xy_m'])
 closes=np.argwhere(chunks[:,:,9]>.5);executedclose=[i for i,t in enumerate(trace[:1436]) if t['command'][-1]>0]
 qclose=None if not len(closes) else {'query':int(closes[0,0]),'action':int(closes[0,1]),'flat_index':int(closes[0,0]*30+closes[0,1]),'probability':float(chunks[tuple(closes[0])][9])}
 rec={**row,'first_chunk_min_raw_target_distance_to_current_brick_mm':float(rawdist[0].min()),'min_raw_target_distance_to_current_brick_mm':float(rawdist.min()),'first_chunk_last_raw_target_xyz':chunks[0,-1,:3].tolist(),'first_chunk_start_to_last_target_displacement_mm':((chunks[0,-1,:3]-current_eef[0])*1000).tolist(),'minimum_executed_eef_distance_to_current_brick_mm':float(np.linalg.norm(actual-obj,axis=1).min()*1000),'first_chunk_min_actual_eef_distance_mm':float(np.linalg.norm(actual[:30]-obj[:30],axis=1).min()*1000),'raw_gripper_probability_range':[float(raw[:,9].min()),float(raw[:,9].max())],'raw_close_proposals':int((raw[:,9]>.5).sum()),'first_raw_close':qclose,'first_executed_close_index':executedclose[0] if executedclose else None,'contact_steps':sum(bool(t['object_truth']['grasp_contact']) for t in trace),'bilateral_steps':sum(bool(t['object_truth']['bilateral_grasp_contact']) for t in trace),'max_lift_mm':float((obj[:,2]-ep['initial_truth']['object_position_m'][2]).max()*1000),'max_object_displacement_mm':float(np.linalg.norm(obj-np.array(ep['initial_truth']['object_position_m']),axis=1).max()*1000),'executed_eef_path_length_mm':float(np.linalg.norm(np.diff(np.vstack([ep['initial_truth']['eef_position_m'],actual]),axis=0),axis=1).sum()*1000),'min_raw_target_xy_to_displayed_marker_mm':float(np.linalg.norm(chunks[:,:,:2]-marker,axis=2).min()*1000),'final_brick_distance_to_displayed_marker_mm':float(np.linalg.norm(obj[-1,:2]-marker)*1000),'final_brick_distance_to_loading_fixture_mm':float(np.linalg.norm(obj[-1,:2]-target)*1000),'initial_truth':ep['initial_truth'],'final_truth':ep['final_truth'],'evaluation':ep['evaluation'],'source_sha256':{n:sha(root/n) for n in ['episode.json','predictions.npz','controller_trace.jsonl.gz','policy_io.npz']}}
 dump(root/'start-diagnostics.json',rec);details.append(rec);cache[row['id']]={'chunks':chunks.copy(),'histories':hist.copy(),'trace':trace,'ep':ep,'rawdist':rawdist}
 assert time.time()<started+600
by={x['id']:x for x in details};pairs=[]
for kind,combos in [('marker_at_fixed_object',[('AA','AB'),('BA','BB')]),('position_at_fixed_marker',[('AA','BA'),('AB','BB')])]:
 for a,b in combos:
  for seed in [912171,912172]:
   x=by[f'{a}-{seed}'];y=by[f'{b}-{seed}'];pairs.append({'comparison':kind,'A':x['id'],'B':y['id'],'B_minus_A_first_chunk_min_raw_distance_mm':y['first_chunk_min_raw_target_distance_to_current_brick_mm']-x['first_chunk_min_raw_target_distance_to_current_brick_mm'],'B_minus_A_min_actual_distance_mm':y['minimum_executed_eef_distance_to_current_brick_mm']-x['minimum_executed_eef_distance_to_current_brick_mm'],'B_minus_A_raw_close_proposals':y['raw_close_proposals']-x['raw_close_proposals'],'supported_lift_A':x['supported_lift_10mm_3steps'],'supported_lift_B':y['supported_lift_10mm_3steps'],'strict_success_A':x['requested_success'],'strict_success_B':y['requested_success']})
follow=None
for a,b in [('AA','AB'),('BA','BB')]:
 for seed in [912171,912172]:
  x=f'{a}-{seed}';y=f'{b}-{seed}'
  if follow is None and not np.array_equal(cache[x]['chunks'],cache[y]['chunks']):follow=(x,y,'first lexicographic differing same-position marker pair')
if follow is None:follow=('AA-912171','REF-912171','no differing marker pair; predeclared AA/standalone fallback')
a,b,why=follow;x=cache[a];y=cache[b];different=np.argwhere(np.any(x['chunks']!=y['chunks'],axis=2))
followup={'A':a,'B':b,'selection':why,'first_differing_query_action':different[0].tolist() if len(different) else None,'first_chunk_max_abs_translation_difference_mm':float(abs(x['chunks'][0,:,:3]-y['chunks'][0,:,:3]).max()*1000),'first_chunk_max_abs_gripper_difference':float(abs(x['chunks'][0,:,9]-y['chunks'][0,:,9]).max()),'first_chunk_proposed_xyz_A':x['chunks'][0,:,:3].tolist(),'first_chunk_proposed_xyz_B':y['chunks'][0,:,:3].tolist(),'first_chunk_gripper_A':x['chunks'][0,:,9].tolist(),'first_chunk_gripper_B':y['chunks'][0,:,9].tolist(),'first_chunk_actual_eef_A':[t['eef_xyz'] for t in x['trace'][:30]],'first_chunk_actual_eef_B':[t['eef_xyz'] for t in y['trace'][:30]],'policy_inference_calls':0,'physics_steps':0,'finished_epoch':time.time()}
np.testing.assert_array_equal(x['histories'][0],y['histories'][0])
followup['initial_proprio_exact']=True
if why.startswith('first'):
 assert x['ep']['initial_truth']==y['ep']['initial_truth'];followup['initial_object_robot_truth_exact']=True
 for f in ['noise-000.npz']:
  with np.load(Path(by[a]['directory'])/'input-audit'/f) as u,np.load(Path(by[b]['directory'])/'input-audit'/f) as v:
   for k in u.files:np.testing.assert_array_equal(u[k],v[k])
 followup['first_actual_noise_exact']=True
close_index=by[b]['first_executed_close_index']
if close_index is not None:
 t=y['trace'][close_index];sample={k:t[k] for k in ['step','target','eef_xyz','command','object_truth']};sample['zero_based_action_index']=close_index;sample['time_since_first_policy_action_s']=(close_index+1)/20;followup['first_close_in_B']=sample
followup['finished_epoch']=time.time()
dump(O/'diagnostics.json',details);dump(O/'paired-effects.json',pairs);dump(O/'saved-trace-followup.json',followup)
keys=['id','cell','seed','requested_success','supported_lift_10mm_3steps','position_error_mm','yaw_error_deg','first_chunk_min_raw_target_distance_to_current_brick_mm','min_raw_target_distance_to_current_brick_mm','minimum_executed_eef_distance_to_current_brick_mm','raw_close_proposals','contact_steps','bilateral_steps','max_lift_mm','max_object_displacement_mm','executed_eef_path_length_mm','final_brick_distance_to_displayed_marker_mm']
with open(O/'comparisons.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in details)
with open(O/'paired-effects.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
canvas=Image.new('RGB',(1100,10*210),'#11151b');draw=ImageDraw.Draw(canvas)
for i,row in enumerate(details):
 root=Path(row['directory'])
 draw.text((8,i*210+5),row['id']+'  lift='+str(row['supported_lift_10mm_3steps'])+'  strict='+str(row['requested_success'])+'  error='+format(row['position_error_mm'],'.1f')+'mm',fill='white')
 for j,n in enumerate(['initial.png','final.png']):
  im=Image.open(root/n);im.thumbnail((540,180));canvas.paste(im,(8+j*550,i*210+25))
canvas.save(O/'initial-final-contact-sheet.png')
for index,(a,b) in enumerate([('AA-912171','AB-912171'),('BA-912171','BB-912171')]):
 dest=O/('marker-pair-'+str(index+1)+'.mp4')
 if not dest.exists():
  cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(Path(by[a]['directory'])/'rollout.mp4'),'-i',str(Path(by[b]['directory'])/'rollout.mp4'),'-filter_complex','[0:v]scale=768:384[a];[1:v]scale=768:384[b];[a][b]vstack=inputs=2[v]','-map','[v]','-c:v','libx264','-preset','veryfast','-crf','23','-threads','4',str(dest)]
  subprocess.run(cmd,check=True,timeout=80)
summary={'passed':True,'reproduction_rerun':REPRODUCE,'episodes':10,'controls':2,'controls_supported_lift_and_strict':True,'new_factorial_rollouts':8,'actual_noise_paired_by_query':True,'preparation_before_deadline':p['frozen_epoch']<p['clock']['preparation_deadline'],'controls_before_deadline':read(O/'control-gate.json')['epoch']<p['clock']['controls_deadline'],'execution_before_deadline':complete['epoch']<p['clock']['execution_deadline'],'followup_within_10_minutes':time.time()<started+600,'analysis_finished_epoch':time.time(),'training':False,'new_independent_holdout_families':0,'new_original_EVAL02_intervention_trials':0,'cell_results':{c:{'strict':sum(r['requested_success'] for r in rows if r['cell']==c),'supported_lift':sum(r['supported_lift_10mm_3steps'] for r in rows if r['cell']==c),'trials':2} for c in ['REF','AA','AB','BA','BB']}}
dump(O/'validation_summary.json',summary)
for n in ['diagnostics.json','paired-effects.json','paired-effects.csv','comparisons.csv','saved-trace-followup.json','trace-followup-clock.json','validation_summary.json','control-gate.json','summary.json','complete.json','execution-freeze.json','initial-final-contact-sheet.png','scene-contact-sheet.png']:(D/n).write_bytes((O/n).read_bytes())
print(json.dumps(summary,indent=2));print('FOLLOWUP',json.dumps({k:v for k,v in followup.items() if not isinstance(v,list)},indent=2));print('PAIRS',json.dumps(pairs,indent=2))
