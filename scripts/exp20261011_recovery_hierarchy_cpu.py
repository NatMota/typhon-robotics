"""Independent saved-RGB audit; no simulator and no runtime mutation."""
import json,gzip,sys,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO),str(REPO/'scripts')]
from exp20261011_hierarchy_runtime_v2 import ObservableBlock,goal_evidence,PrimitiveAdapter
from exp20261011_hierarchy_runtime import pose_target
from supervisor.baseline.xvla import to_action
OUT=Path('/var/tmp/typhon-research-20261011/hierarchy/independent-primitive-audit-v1')
def load(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')
rows=[]
for d in sorted((OUT/'episodes').iterdir()):
 sc=load(d/'scenario.json');e=load(d/'episode.json');h=load(d/'hierarchy-summary.json');events=load(d/'observable-phase.json')['events'];a=np.load(d/'policy_io.npz');cfg=sc['config'];truth=e['final_truth']
 obs={'rgb':{k:a[k][-1] for k in ['primary','secondary']},'proprio':{'eef_position_m':a['eef_position_m'][-1],'eef_quat_xyzw':a['eef_quat_xyzw'][-1],'gripper_joint_position_m':truth['gripper_joint_position_m']}}
 hypotheses={}
 for name,carry in [('runtime_carry_plane',True),('resting_plane',False)]:
  sensor=ObservableBlock(cfg);b=sensor.step(obs,0,False,carry);g=goal_evidence(b,cfg)
  hypotheses[name]={'block':b,'goal':g,'xy_error_mm':None if b['status']=='unknown' else float(np.linalg.norm(np.array(b['position_m'][:2])-truth['object_position_m'][:2])*1000)}
 qevents=[r for r in events if r['object']['status']=='unknown'];r={'source_id':d.name,'physical_success':e['success'],'pickup_70mm':h['bilateral_pickup_70mm'],'last_phase':h['final_phase'],'completion_claimed':h['completion_claimed'],'position_error_mm':h['position_error_mm'],'yaw_error_deg':h['yaw_error_deg'],'unknown_updates':len(qevents),'total_updates':len(events),'first_unknown_step':qevents[0]['step'] if qevents else None,'first_unknown':qevents[0]['object'] if qevents else None,'source_age_final_s':events[-1]['source_age_s'],'transitions':h['transitions'],'final_RGB_hypotheses':hypotheses,'uses_evaluator_truth_only_for_diagnostic_error':True}
 rows.append(r)
 print(json.dumps({'id':d.name,'success':e['success'],'phase':h['final_phase'],'hypotheses':{n:{'goal':x['goal'],'xy_error_mm':x['xy_error_mm']} for n,x in hypotheses.items()}}))
save(OUT/'saved-rgb-plane-audit.json',{'method':'Same final saved model-query RGB and measured EEF; only projection plane toggled. Evaluation truth is error reference only. No actor or data outputs modified.','rows':rows})
# Meaningful algebraic contrast: premature OPEN coincident with valid translation and rotation.
cfg=load(OUT/'episodes/personal_demo_0003/scenario.json')['config'];obs={'proprio':{'eef_position_m':[0.,0.,1.0],'eef_quat_xyzw':Rotation.from_euler('xyz',[math.pi,0,0]).as_quat()}}
r=Rotation.from_quat(obs['proprio']['eef_quat_xyzw']).as_matrix();raw=pose_target([.02,-.01,1.0],Rotation.from_euler('z',.10).as_matrix()@r,False)
e={'phase':'orient','gripper_span_m':.034,'goal':{'complete':False},'held':True,'object':{'status':'observed'}}
a,event=PrimitiveAdapter(cfg).from_vla(raw,obs,e);griponly=raw.copy();griponly[9]=1
save(OUT/'release-veto-algebra.json',{'case':'Fresh observed held object with incomplete orientation; numeric target combines +20mmX,-10mmY,+0.1radZ and OPEN','raw_command':to_action(raw,obs,cfg),'current_full_pose_veto_command':to_action(a,obs,cfg),'proposed_gripper_only_veto_command':to_action(griponly,obs,cfg),'scope':'Algebraic interface audit only, not a physical rollout. Unknown-pose/hazard safe hold must remain a separate condition. No runtime changed.','current_event':event})
