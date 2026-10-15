"""INIT08 saved-trace analysis and pre-contact packet selection. No model/physics calls."""
import os,sys,json,time,gzip,csv,traceback
from pathlib import Path
W=Path(__file__).resolve().parents[1];os.chdir(W);sys.path[:0]=[str(W),str(W/'scripts'),str(W/'.m5-deps'),str(W/'.m6-deps')]
import numpy as np,torch
from transformers import AutoProcessor
from init04_prepare import prepare,read,dump,sha,lines
from load04_common import array_sha
O=Path('/var/tmp/typhon-init08-20261015');NEW=Path('/var/tmp/typhon-init07-20261015');OLD=Path('/var/tmp/typhon-init02-20261015')
CASES=['init-demo_0003-loading-912172','init-demo_0005-loading-912171']
WEIGHTS={'Q0':'54c86e744e0c66cd5a55f81b56b949cd71000ed01726125550e878dcfaed081c','Q1':'26d573eb90f791d9af86bbff30e138f51c5aa658ac7abefe76787ba352363a7a'}
def contact(row):
 return any('cube'in c['geom1']+c['geom2']and'gripper'in c['geom1']+c['geom2']for c in row['contacts'])
def streak(a):
 best=now=0
 for x in a:now=now+1 if x else 0;best=max(best,now)
 return best
def first(a):return next((i for i,x in enumerate(a)if x),None)
def snap(tr,i,initial,dt):
 if i is None:return None
 r=tr[i];s=r['object_truth'];prev=initial['eef_position_m']if i==0 else tr[i-1]['eef_xyz']
 return {'local_index':i,'control_tick':r['step'],'raw_target':r['target'],'bounded_command':r['command'],'actual_eef_xyz':r['eef_xyz'],'actual_eef_quat_xyzw':r['eef_quat_xyzw'],'controller_goal_xyz':r['controller_goal_xyz'],'controller_goal_rotation':r['controller_goal_rotation'],'eef_velocity_m_s':((np.asarray(r['eef_xyz'])-prev)/dt).tolist(),'gripper_joint_position_m':s['gripper_joint_position_m'],'aperture_m':abs(s['gripper_joint_position_m'][0]-s['gripper_joint_position_m'][1]),'object_position_m':s['object_position_m'],'object_tilt_deg':r['object_tilt_deg'],'bilateral':s['bilateral_grasp_contact'],'grasp_contact':s['grasp_contact'],'object_gripper_contacts':[c for c in r['contacts_full']if 'cube'in c['geom1']+c['geom2']and'gripper'in c['geom1']+c['geom2']],'finger_geoms':r['finger_geoms'],'position_error_mm':r['position_error_mm'],'orientation_error_deg':r['orientation_error_deg'],'joint_positions':r['joint_positions'],'nearest_joint_limit_rad':r['nearest_joint_limit_rad']}
def main():
 torch.set_num_threads(4);clock=read(O/'clock.json');assert time.time()<clock['trace_selection_deadline']
 sources={};traces={};events=[];phase_rows=[]
 for case in CASES:
  for model in ['T','Q0','Q1']:
   d=(OLD if model=='T'else NEW)/'episodes'/model/case
   for n in ['episode.json','summary.json','controller_trace.jsonl.gz','trajectory.jsonl.gz','policy_io.npz','predictions.npz','inference.json']:sources[str(d/n)]=sha(d/n)
   ep=read(d/'episode.json');tr=lines(d/'controller_trace.jsonl.gz');assert len(tr)==1460
   assert [r['step']for r in tr]==list(range(ep['initial_truth']['control_steps']+1,ep['initial_truth']['control_steps']+1461))
   ts=[r['object_truth']for r in tr];dt=float(np.median(np.diff([s['sim_time_s']for s in ts])));assert abs(dt-.05)<1e-8
   touched=[contact(r)for r in tr];bilat=[s['bilateral_grasp_contact']for s in ts];close=[r['target'][9]>.5 for r in tr]
   rise=np.array([s['object_position_m'][2]-ep['initial_truth']['object_position_m'][2]for s in ts])*1000
   supported=[b and z>=10 for b,z in zip(bilat,rise)]
   fc=first(touched);fb=first(bilat);cl=first(close);fs=first([all(supported[i:i+3])for i in range(len(supported)-2)])
   loss=next((i for i in range(1,len(tr))if bilat[i-1]and not bilat[i]),None)
   opening=next((i for i in range(1,len(tr))if close[i-1]and not close[i]),None)
   phases={'precontact':fc-1 if fc is not None and fc>0 else None,'first_contact':fc,'first_close':cl,'first_bilateral':fb,'first_supported_lift':fs,'first_bilateral_loss':loss,'first_open_after_close':opening,'maximum_object_rise':int(rise.argmax()),'final':len(tr)-1}
   snaps={name:snap(tr,i,ep['initial_truth'],dt)for name,i in phases.items()}
   row={'model':model,'case':case,'strict_success':read(d/'summary.json')['requested_success'],'initial_tick':ep['initial_truth']['control_steps'],'first_contact_tick':None if fc is None else tr[fc]['step'],'first_close_tick':None if cl is None else tr[cl]['step'],'first_bilateral_tick':None if fb is None else tr[fb]['step'],'first_supported_lift_tick':None if fs is None else tr[fs]['step'],'contact_ticks':sum(touched),'bilateral_ticks':sum(bilat),'max_bilateral_streak_ticks':streak(bilat),'max_supported_lift_streak_ticks':streak(supported),'maximum_object_rise_mm':float(rise.max()),'maximum_bilateral_object_rise_mm':float(max((z for z,b in zip(rise,bilat)if b),default=0)),'close_target_ticks':sum(close),'maximum_close_streak_ticks':streak(close),'dt_s':dt}
   if cl is not None:
    end=next((i for i in range(cl+1,len(tr))if not close[i]),len(tr));ref=tr[cl]['eef_xyz'][2]
    row.update(first_close_segment_ticks=end-cl,first_close_segment_requested_rise_mm=float((max(r['target'][2]for r in tr[cl:end])-ref)*1000),first_close_segment_actual_wrist_rise_mm=float((max(r['eef_xyz'][2]for r in tr[cl:end])-ref)*1000),first_close_segment_actual_object_rise_mm=float(rise[cl:end].max()),first_close_aperture_mm=snaps['first_close']['aperture_m']*1000)
   events.append({'summary':row,'phases':snaps});traces[model,case]=(d,ep,tr)
   for phase,s in snaps.items():
    if s is not None:phase_rows.append({'model':model,'case':case,'phase':phase,'control_tick':s['control_tick'],'target_xyz':json.dumps(s['raw_target'][:3]),'target_gripper_probability':s['raw_target'][9],'bounded_command':json.dumps(s['bounded_command']),'actual_xyz':json.dumps(s['actual_eef_xyz']),'actual_aperture_mm':s['aperture_m']*1000,'actual_velocity_m_s':json.dumps(s['eef_velocity_m_s']),'object_rise_mm':(s['object_position_m'][2]-ep['initial_truth']['object_position_m'][2])*1000,'object_tilt_deg':s['object_tilt_deg'],'bilateral_contact':s['bilateral'],'contact_geoms':json.dumps([[c['geom1'],c['geom2']]for c in s['object_gripper_contacts']]),'tracking_error_mm':s['position_error_mm']})
 dump(O/'trace-events.json',events)
 for name,rows in [('trace-summary.csv',[x['summary']for x in events]),('phase-table.csv',phase_rows)]:
  keys=list(dict.fromkeys(k for r in rows for k in r))
  with(O/name).open('w')as f:wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)
 proc=AutoProcessor.from_pretrained(W/'.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);proc.language_max_length=128
 cells={};selections=[];rec=O/'selected-records';rec.mkdir(exist_ok=False)
 for case in CASES:
  earliest=min(x['summary']['first_contact_tick']for x in events if x['summary']['case']==case and x['summary']['model']in ['Q0','Q1'])
  base=traces['Q0',case][1]['initial_truth']['control_steps'];assert base==traces['Q1',case][1]['initial_truth']['control_steps']
  eligible=[q for q in range(48)if base+30*q<earliest];assert eligible
  q=max(eligible);sel={'case':case,'earliest_contact_tick':earliest,'shared_query_index':q,'shared_query_tick':base+30*q,'original_policy_seed':int(case.rsplit('-',1)[1]),'states':[]}
  for model in ['Q0','Q1']:
   d,ep,tr=traces[model,case]
   with np.load(d/'policy_io.npz')as z:io={k:z[k].copy()for k in ['primary','secondary','eef_position_m','eef_quat_xyzw']}
   with np.load(d/'predictions.npz')as z:hist=z['histories'].copy();expected=z['chunks'][q].copy()
   ap=d/'input-audit'/f'input-{q:03d}.json';audit=read(ap);sources[str(ap)]=sha(ap)
   b=prepare(proc,io,hist,q,audit['instruction']);hashes={k:array_sha(v.numpy())for k,v in b.items()};assert hashes==audit['input_hashes'],(case,model,'input mismatch')
   fixed=ep['initial_truth']if q==0 else tr[q*30-1]['object_truth'];np.testing.assert_array_equal(io['eef_position_m'][q],fixed['eef_position_m'])
   assert not any(contact(r)for r in tr[:q*30])
   noises=[]
   for seed in [912171,912172]:
    nd=NEW/'episodes'/model/(case.rsplit('-',1)[0]+'-'+str(seed))/'input-audit'/f'noise-{q:03d}.npz'
    with np.load(nd)as z:n=z['draw_0'].copy()
    assert n.shape==(1,30,20)and n.dtype==np.float32
    noises.append(n);sources[str(nd)]=sha(nd)
   assert array_sha(noises[[912171,912172].index(sel['original_policy_seed'])])==audit['noise_hashes'][0]
   key=case+'-'+model;effective=b['proprio'].clone();effective[...,9]=0;effective[...,19]=0
   cells[key]={'inputs':b,'effective_proprio':effective,'noises':noises,'expected_executor':expected,'donor_model':model,'case':case,'query':q,'fixed_truth':fixed,'measured_xyz':io['eef_position_m'][q],'measured_quat':io['eef_quat_xyzw'][q],'original_noise_index':[912171,912172].index(sel['original_policy_seed'])}
   np.savez_compressed(rec/(key+'.npz'),primary=io['primary'][q],secondary=io['secondary'][q],history=hist[q],expected_executor=expected,measured_xyz=io['eef_position_m'][q],measured_quat=io['eef_quat_xyzw'][q],noise_912171=noises[0],noise_912172=noises[1])
   dump(rec/(key+'.json'),{'input_hashes':hashes,'instruction':audit['instruction'],'fixed_truth':fixed,'source_directory':str(d),'query':q})
   sel['states'].append({'key':key,'model':model,'input_hashes':hashes,'effective_proprio_sha256':array_sha(effective.numpy()),'noise_hashes':[array_sha(n)for n in noises]})
  a,b=(cells[case+'-'+m]for m in ['Q0','Q1'])
  assert all(np.array_equal(x,y)for x,y in zip(a['noises'],b['noises']))
  equal=all(torch.equal(a['inputs'][k],b['inputs'][k])for k in a['inputs'])
  sel['effective_packets_identical']=equal;sel['deduplicate']=equal
  selections.append(sel)
 torch.save({'cells':cells},O/'packets.pt')
 for p,value in sources.items():assert sha(p)==value
 assert time.time()<clock['trace_selection_deadline']
 protocol={'experiment':'INIT08','ticket':11079,'cases':CASES,'weights':{m:{'path':str(NEW/('train-'+m)/'step3000.pt'),'sha256':s}for m,s in WEIGHTS.items()},'selections':selections,'source_sha256':sources,'packets_sha256':sha(O/'packets.pt'),'freeze_epoch':time.time(),'maximum_outputs':16,'deduplicated_output_count':sum((1 if s['deduplicate']else 2)*2*2 for s in selections),'new_physics':0,'new_training':0,'generation':{'float32_weights':True,'bf16_autocast':True,'domain':29,'steps':10,'horizon':30,'native_previous_target_proprio':True},'trace_geometry_for_offline_reference_only':True}
 dump(O/'protocol.json',protocol);dump(O/'preparation-complete.json',{'passed':True,'epoch':time.time(),'six_trace_summaries':6,'selections':selections,'new_model_calls':0})
 print(json.dumps({'passed':True,'selection':selections,'summaries':[x['summary']for x in events]},indent=2))
if __name__=='__main__':
 try:main()
 except BaseException:dump(O/'PREPARATION_STOPPED.json',{'epoch':time.time(),'error':traceback.format_exc()});raise
