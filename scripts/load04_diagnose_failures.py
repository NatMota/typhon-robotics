"""Post-hoc failure-phase audit of saved logs; no simulator execution."""
from load04_common import *
import numpy as np,csv,math
rows=read(OUT/'analysis/episodes.json')
def tilt(q,xyzw=False):
 if xyzw:x,y,z,w=q
 else:w,x,y,z=q
 return float(np.degrees(np.arccos(np.clip(1-2*(x*x+y*y),-1,1))))
def root(r):return OUT/'evaluation'/r['model']/r['visibility']/r['scene']/r['id']
summary=[]
for r in rows:
 if r['requested_task']!='loading' or r['scene']!='demo_0012':continue
 path=root(r);tr=[json.loads(x) for x in gzip.open(path/'controller_trace.jsonl.gz','rt')];ts=[json.loads(x) for x in gzip.open(path/'trajectory.jsonl.gz','rt')];acts={x['step']:x for x in ts}
 init=read(path/'episode.json')['initial_truth'];h0=init['object_position_m'][2]
 samples=[]
 for x in tr:
  t=x['object_truth'];q=t['object_quat_wxyz'];a=acts[x['step']]
  samples.append({'step':x['step'],'sim_time_s':t['sim_time_s'],'object_tilt_deg':tilt(q),'object_yaw_deg':math.degrees(t['object_yaw_rad']),
   'height_above_initial_mm':(t['object_position_m'][2]-h0)*1000,'bilateral_contact':t['bilateral_grasp_contact'],'gripper_open':t['gripper_open'],
   'command_open':a['applied_action'][-1]<0,'actual_eef_down_error_deg':180-tilt(x['eef_quat_xyzw'],True),
   'command_eef_down_error_deg':float(np.degrees(np.arccos(np.clip(-np.asarray(x['controller_goal_rotation'])[2,2],-1,1)))),
   'contact_pairs':[(c['geom1'],c['geom2']) for c in x['contacts']]})
 def first(test,start=0):return next((x for x in samples if x['step']>=start and test(x)),None)
 grasp=first(lambda x:x['bilateral_contact'])
 lift=first(lambda x:x['bilateral_contact'] and x['height_above_initial_mm']>=10)
 release=first(lambda x:x['command_open'],lift['step']+1) if lift else None
 events={'first_bilateral_contact':grasp,'first_supported_10mm_lift':lift,'first_open_command_after_lift':release,
  'first_tilt_over_15deg':first(lambda x:x['object_tilt_deg']>15),'first_tilt_over_45deg':first(lambda x:x['object_tilt_deg']>45),
  'first_tilt_over_75deg':first(lambda x:x['object_tilt_deg']>75),'last':samples[-1]}
 failure=r['model']=='B' and not r['requested_success']
 record={'model':r['model'],'visibility':r['visibility'],'seed':r['seed'],'source':str(path),'strict':r['requested_success'],'failure':failure,'events':events,
  'failed_criteria':[k for k,v in r['scores']['loading']['criteria'].items() if not v],
  'maximum_command_eef_down_error_deg':max(x['command_eef_down_error_deg'] for x in samples),
  'maximum_actual_eef_down_error_deg':max(x['actual_eef_down_error_deg'] for x in samples)}
 summary.append(record)
 dump(OUT/'analysis/failure-phases'/(r['model']+'-'+r['id']+'.json'),{'record':record,'samples':samples})
dump(OUT/'analysis/failure-phase-summary.json',summary)
print(json.dumps([{'model':x['model'],'visibility':x['visibility'],'seed':x['seed'],'strict':x['strict'],'events':{k:None if v is None else {j:v[j] for j in ['step','object_tilt_deg','command_eef_down_error_deg','actual_eef_down_error_deg','height_above_initial_mm']} for k,v in x['events'].items()}} for x in summary],indent=2))
