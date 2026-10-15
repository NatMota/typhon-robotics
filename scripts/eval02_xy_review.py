"""Post-run derived checks, never changes actor or frozen run source."""
import os,sys,json,copy
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')]
from eval02_xy import OUT,read,dump,requested_score
p=read(OUT/'protocol.json');rows=read(OUT/'summary.json');config=p['case']['config']
ref=read(OUT/'episodes/C-912171/episode.json')
checks=[]
for name,changes in [('sideways',{'object_quat_wxyz':[.7071067811865476,.7071067811865476,0.,0.]}),('elevated',{'object_position_m':[.125,-.1,.9]}),('unreleased',{'gripper_open':False})]:
 ep=copy.deepcopy(ref);ep['final_truth']['object_position_m']=[.125,-.1,.815]
 ep['final_truth'].update(changes)
 _,score=requested_score(ep,config,[.125,-.1]);assert not score['success']
 checks.append({'test':name,'passed':True,'criteria':score['criteria']})
for row in rows:
 ep=read(OUT/'episodes'/row['id']/'episode.json')
 a,b=requested_score(ep,config,[.125,-.12]);assert a['feeding']['success']==b['success']
checks.append({'test':'requested_centre_equals_original_predicate_for_all6','passed':True})
gains=[]
for seed in p['policy_seeds']:
 minus=next(r for r in rows if r['policy_seed']==seed and r['condition']=='Yminus')
 plus=next(r for r in rows if r['policy_seed']==seed and r['condition']=='Yplus')
 dy=plus['final_xy_m'][1]-minus['final_xy_m'][1]
 gains.append({'policy_seed':seed,'extreme_endpoint_separation_y_mm':dy*1000,'requested_separation_y_mm':40.,'finite_difference_y_gain':dy/.04})
dump(OUT/'post-run-scorer-checks.json',{'passed':True,'analysis_only_after_rollouts':True,'tests':checks})
dump(OUT/'extreme-steering-gain.json',gains)
print(json.dumps(gains))
