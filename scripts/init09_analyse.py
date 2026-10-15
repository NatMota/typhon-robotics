"""INIT09 kinematics/contact and retry audit from immutable recordings."""
import os,sys,json,gzip,csv,time,hashlib,traceback
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'scripts'))
from init09_prepare import O,CASES,read,dump,sha,trace,spans,contact
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
def quat(q):
 x,y,z,w=np.asarray(q,float)/np.linalg.norm(q)
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def angle(a,b):return float(np.degrees(np.arccos(np.clip((np.trace(a@b.T)-1)/2,-1,1))))
def rot6(a):
 x=np.asarray(a[3:6],float);x/=np.linalg.norm(x);y=np.asarray(a[6:9],float);y-=x*np.dot(x,y);y/=np.linalg.norm(y);return np.column_stack([x,y,np.cross(x,y)])
def csvout(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w')as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def first(a):return next((i for i,x in enumerate(a)if x),None)
def main():
 clock=read(O/'clock.json');assert time.time()<clock['analysis_deadline'];seal=read(O/'selection-freeze.json')
 assert seal['epoch']<clock['inventory_deadline']
 for f,h in seal['source_sha256'].items():assert sha(Path(f))==h
 cache={};queryrows=[];querydetail={};mismatch=0.;relmax=0.;rows=[];phase_rows=[];details=[];series={}
 for case in CASES:
  for model in ['T','Q0','Q1']:
   d=Path('/var/tmp/typhon-init02-20261015'if model=='T'else'/var/tmp/typhon-init07-20261015')/'episodes'/model/case
   tr=trace(d/'controller_trace.jsonl.gz');ep=read(d/'episode.json');base=ep['initial_truth']['control_steps'];inf=read(d/'inference.json')
   with np.load(d/'predictions.npz')as z:chunks=z['chunks'].copy();hist=z['histories'].copy()
   with np.load(d/'policy_io.npz')as z:io={k:z[k].copy()for k in ['primary','secondary','eef_position_m','eef_quat_xyzw']}
   for q in range(1,48):np.testing.assert_array_equal(hist[q,:9],chunks[q-1,-1,:9])
   cache[case,model]=(d,tr,ep,chunks,hist,io,inf)
   for q in range(48):
    st=ep['initial_truth']if q==0 else tr[q*30-1]['object_truth'];obj=np.array(st['object_position_m']);wr=io['eef_position_m'][q];target=chunks[q,:,:3]
    rq=rot6(chunks[q,0]);mq=quat(io['eef_quat_xyzw'][q]);m=inf[q]
    queryrows.append({'case':case,'model':model,'query':q,'observation_tick':base+q*30,'first_action_tick':base+q*30+1,'last_action_tick':base+(q+1)*30,'measured_wrist_xyz':json.dumps(wr.tolist()),'object_xyz':json.dumps(obj.tolist()),'object_tilt_deg':float(np.degrees(np.arccos(np.clip(quat(np.array(st['object_quat_wxyz'])[[1,2,3,0]])[2,2],-1,1)))),'object_speed_mm_s':float(np.linalg.norm(st['object_linear_velocity_m_s'])*1000),'measured_aperture_mm':abs(np.diff(st['gripper_joint_position_m'])[0])*1000,'predicted_first_xyz':json.dumps(target[0].tolist()),'predicted_last_xyz':json.dumps(target[-1].tolist()),'predicted_min_z_m':float(target[:,2].min()),'predicted_max_z_m':float(target[:,2].max()),'measured_distance_to_object_mm':float(np.linalg.norm(wr-obj)*1000),'predicted_min_distance_to_fixed_object_mm':float(np.linalg.norm(target-obj,axis=1).min()*1000),'first_close_waypoint':first(chunks[q,:,9]>.5),'first_target_rotation_departure_deg':angle(rq,mq),'object_crop_detected':m['crop_metadata']['detected'],'image_mask':json.dumps(m['image_mask'][0]),'current_observation_only':True})
    raw=hist[q].copy();native=raw.copy()
    for off in [0,10]:native[off+3:off+9]=raw[off+3:off+9][[0,3,1,4,2,5]]
    effective=native.copy();effective[[9,19]]=0
    querydetail[case+'-'+model+'-q'+str(q)]={'history_executor_order':raw.tolist(),'native_preprocess_proprio':native.tolist(),'effective_transformer_proprio':effective.tolist(),'measured_xyz':wr.tolist(),'measured_quat_xyzw':io['eef_quat_xyzw'][q].tolist(),'object_truth':st,'crop':m['crop_metadata'],'image_mask':m['image_mask'],'raw_chunk':chunks[q].tolist(),'instruction':read(d/'input-audit'/('input-%03d.json'%q))['instruction']}
   # Every relative transform is checked independently from recorded world poses.
   for r in tr:
    R=quat(r['eef_quat_xyzw']);B=quat(np.array(r['object_truth']['object_quat_wxyz'])[[1,2,3,0]])
    relative=R.T@(np.array(r['object_truth']['object_position_m'])-r['eef_xyz'])
    mismatch=max(mismatch,float(np.abs(relative-r['p_GO']).max()));relmax=max(relmax,float(np.abs(R.T@B-r['R_GO']).max()))
 assert mismatch<1e-8 and relmax<1e-8,(mismatch,relmax)
 for a in seal['attempts']:
  d,tr,ep,chunks,hist,io,inf=cache[a['case'],a['model']];s=a['window_start_index'];end=a['window_end_index_exclusive'];cut=tr[s:end];z0=ep['initial_truth']['object_position_m'][2]
  xyz=np.array([r['eef_xyz']for r in cut]);obj=np.array([r['object_truth']['object_position_m']for r in cut]);rel=np.array([r['p_GO']for r in cut]);ap=np.array([abs(np.diff(r['object_truth']['gripper_joint_position_m'])[0])*1000 for r in cut])
  bi=np.array([r['object_truth']['bilateral_grasp_contact']for r in cut]);rise=(obj[:,2]-z0)*1000;touch=np.array([contact(r)for r in cut])
  cl=a['close_segment']['start_index']-s;opening=a['close_segment']['end_index_exclusive']-s;fc=first(touch);fb=first(bi);supported=bi&(rise>=10)
  fs=first([all(supported[i:i+3])for i in range(len(cut)-2)]);loss=next((i for i in range((fb or 0)+1,len(cut))if bi[i-1]and not bi[i]),None)
  # Geometric lift onset, not a learned phase: >2mm wrist rise relative to first bilateral contact (or close if absent).
  ref=fb if fb is not None else cl
  lift=next((i for i in range(ref,len(cut))if xyz[i,2]-xyz[ref,2]>.002),None)
  phases={'window_start':0,'close':cl,'precontact':max(0,fc-1)if fc is not None else None,'first_contact':fc,'first_bilateral':fb,'wrist_rise_gt2mm':lift,'supported_lift':fs,'last_bilateral_before_loss':loss-1 if loss is not None else None,'support_loss':loss,'opening':opening if opening<len(cut)else None,'post_opening':len(cut)-1}
  snapshots={}
  for name,i in phases.items():
   if i is None:continue
   r=cut[i];cs=[c for c in r['contacts_full']if'cube'in c['geom1']+c['geom2']and'gripper'in c['geom1']+c['geom2']];pads=[c for c in cs if'pad'in c['geom1']+c['geom2']]
   pts=np.array([c['object_local_position']for c in pads])*1000 if pads else np.empty((0,3))
   prev=tr[s+i-1]if s+i>0 else None
   snap={'tick':r['step'],'raw_target':r['target'],'raw_rotation_matrix':rot6(r['target']).tolist(),'bounded_command':r['command'],'wrist_xyz':r['eef_xyz'],'wrist_quat_xyzw':r['eef_quat_xyzw'],'object_truth':r['object_truth'],'p_GO':r['p_GO'],'R_GO':r['R_GO'],'contacts':cs,'finger_geoms':r['finger_geoms'],'aperture_mm':float(ap[i]),'aperture_change_mm_s':None if prev is None else float((ap[i]-abs(np.diff(prev['object_truth']['gripper_joint_position_m'])[0])*1000)/.05),'recorded_actuator_force':None,'recorded_contact_force':None}
   snapshots[name]=snap
   phase_rows.append({'attempt':a['id'],'phase':name,'tick':r['step'],'probability_close':r['target'][9],'bounded_close_command':r['command'][-1],'aperture_mm':ap[i],'aperture_change_mm_s':snap['aperture_change_mm_s'],'object_rise_mm':rise[i],'object_tilt_deg':r['object_tilt_deg'],'relative_xyz_mm':json.dumps((rel[i]*1000).tolist()),'pad_contact_points':len(pads),'pad_local_z_min_mm':float(pts[:,2].min())if len(pts)else None,'pad_local_z_max_mm':float(pts[:,2].max())if len(pts)else None,'pad_local_x_min_mm':float(pts[:,0].min())if len(pts)else None,'pad_local_x_max_mm':float(pts[:,0].max())if len(pts)else None,'finger_geoms':json.dumps(sorted({c[k]for c in cs for k in ['geom1','geom2']if'gripper'in c[k]}))})
  row={'attempt':a['id'],'case':a['case'],'model':a['model'],'rule':a['rule'],'first_close_tick':cut[cl]['step'],'first_contact_tick':cut[fc]['step']if fc is not None else None,'first_bilateral_tick':cut[fb]['step']if fb is not None else None,'first_supported_lift_tick':cut[fs]['step']if fs is not None else None,'first_support_loss_tick':cut[loss]['step']if loss is not None else None,'first_open_tick':cut[opening]['step']if opening<len(cut)else None,'first_contact_wrist_above_object_mm':float((xyz[fc,2]-obj[fc,2])*1000)if fc is not None else None,'first_contact_lateral_offset_mm':float(np.linalg.norm((xyz[fc]-obj[fc])[:2])*1000)if fc is not None else None,'first_contact_aperture_mm':float(ap[fc])if fc is not None else None,'close_window_brick_rise_mm':float(rise[cl:min(opening,len(cut))].max()),'post_opening_tilt_deg':cut[-1]['object_tilt_deg'],'post_opening_object_shift_mm':float(np.linalg.norm(obj[-1]-ep['initial_truth']['object_position_m'])*1000)}
  if fb is not None and loss is not None:
   row.update(relative_translation_to_loss_mm=float(np.linalg.norm(rel[loss]-rel[fb])*1000),relative_translation_before_loss_mm=float(np.linalg.norm(rel[loss-1]-rel[fb])*1000),relative_rotation_to_loss_deg=angle(np.array(cut[loss]['R_GO']),np.array(cut[fb]['R_GO'])),wrist_translation_to_loss_mm=float(np.linalg.norm(xyz[loss]-xyz[fb])*1000),wrist_lateral_to_loss_mm=float(np.linalg.norm((xyz[loss]-xyz[fb])[:2])*1000),wrist_rotation_to_loss_deg=angle(quat(cut[loss]['eef_quat_xyzw']),quat(cut[fb]['eef_quat_xyzw'])),aperture_first_bilateral_mm=float(ap[fb]),aperture_at_loss_mm=float(ap[loss]),tilt_first_bilateral_deg=cut[fb]['object_tilt_deg'],tilt_at_loss_deg=cut[loss]['object_tilt_deg'],close_probability_at_loss=cut[loss]['target'][9])
  rows.append(row);details.append({'selection':a,'phases':snapshots})
  series[a['id']]={'tick':np.array([r['step']for r in cut]),'relative_xyz_mm':rel*1000,'aperture_mm':ap,'object_rise_mm':rise,'bilateral':bi,'tilt_deg':np.array([r['object_tilt_deg']for r in cut]),'first_contact_tick':cut[fc]['step']if fc is not None else cut[cl]['step']}
 csvout(O/'acquisition-table.csv',rows);csvout(O/'phase-measurements.csv',phase_rows);csvout(O/'query-timeline.csv',queryrows)
 dump(O/'acquisition-details.json',details);dump(O/'query-packets.json',querydetail);dump(O/'acquisition-summary.json',rows)
 np.savez_compressed(O/'kinematic-series.npz',**{k+'--'+n:v for k,s in series.items()for n,v in s.items()})
 media=O/'media';media.mkdir(exist_ok=True)
 fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
 for ci,case in enumerate(CASES):
  for model,label,color in [('T','first_success','#4c8b36'),('Q0','first_failed_close','#888888'),('Q0','first_later_success','#217ac0'),('Q1','first_brief_lift_loss','#d56c25')]:
   key=case+'-'+model+'-'+label;s=series[key];t=(s['tick']-s['first_contact_tick'])*.05;mask=(t>=-.5)&(t<=3.0)
   axes[ci,0].plot(t[mask],s['object_rise_mm'][mask],label=model+' '+label,color=color)
   axes[ci,1].plot(t[mask],s['aperture_mm'][mask],label=model+' '+label,color=color)
  for di,ylab in enumerate(['Block rise (mm)','Measured finger aperture (mm)']):
   axes[ci,di].set_title(case.split('-')[1]);axes[ci,di].set_xlabel('Seconds relative to first contact');axes[ci,di].set_ylabel(ylab);axes[ci,di].grid(alpha=.2)
 axes[0,0].legend(fontsize=7);fig.suptitle('Saved contact-aligned acquisition; samples every50ms')
 fig.savefig(media/'acquisition-comparison.png',dpi=130);plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
 for ci,case in enumerate(CASES):
  key=case+'-Q1-first_brief_lift_loss';s=series[key];r=next(r for r in rows if r['attempt']==key);t=(s['tick']-r['first_bilateral_tick'])*.05;mask=(t>=-.2)&(t<=1.1);ref=np.flatnonzero(s['tick']==r['first_bilateral_tick'])[0]
  for axis,col in enumerate(['#cc4c46','#3e8c53','#3f70bd']):axes[ci,0].plot(t[mask],(s['relative_xyz_mm'][:,axis]-s['relative_xyz_mm'][ref,axis])[mask],label='xyz'[axis],color=col)
  axes[ci,1].plot(t[mask],s['aperture_mm'][mask]);axes[ci,2].plot(t[mask],s['tilt_deg'][mask])
  for di,lab in enumerate(['Object shift in measured gripper frame (mm)','Finger aperture (mm)','Brick tilt (degrees)']):
   axes[ci,di].axvline((r['first_support_loss_tick']-r['first_bilateral_tick'])*.05,color='red',ls='--',label='Bilateral loss');axes[ci,di].set_title(case.split('-')[1]);axes[ci,di].set_xlabel('Seconds since first bilateral contact');axes[ci,di].set_ylabel(lab);axes[ci,di].grid(alpha=.2)
 axes[0,0].legend(fontsize=7);fig.suptitle('Q1: relative movement and support loss; no recorded contact force')
 fig.savefig(media/'q1-support-loss.png',dpi=130);plt.close(fig)
 # Only original saved camera observations, selected by frozen attempt/phase rules.
 image_rows=[]
 for a in seal['attempts']:
  d,tr,ep,chunks,hist,io,inf=cache[a['case'],a['model']]
  q=a['first_close_generating_query']
  image_rows.append((a['id'],q,io['primary'][q],io['secondary'][q],inf[q]['crop_metadata']))
 sheet=Image.new('RGB',(1024,552*len(image_rows)),'#101a26');draw=ImageDraw.Draw(sheet)
 for i,(key,q,p,s,crop)in enumerate(image_rows):
  draw.text((8,i*552+8),key+' q'+str(q)+' crop detected='+str(crop['detected']),fill='white')
  for j,img in enumerate([p,s]):
   im=Image.fromarray(img).convert('RGB');im.thumbnail((512,512));sheet.paste(im,(j*512,i*552+32))
 sheet.save(media/'attempt-camera-records.png')
 for f,h in seal['source_sha256'].items():assert sha(Path(f))==h
 assert time.time()<clock['analysis_deadline']
 dump(O/'analysis-validation.json',{'passed':True,'epoch':time.time(),'attempts':8,'queries':len(queryrows),'p_GO_world_transform_max_error_m':mismatch,'R_GO_world_transform_max_error':relmax,'source_sha256_unchanged':True,'new_physics':0,'new_model_calls':0,'new_training':0,'time_qualification':'All recorded pose/contact derived fields captured together after each env.step; finger qpos and velocities have simulator integration timing limitations, so no exact substep causal claim. Contact forces and actuator grip controls absent.'})
 print(json.dumps({'passed':True,'attempts':rows,'relative_transform_error':mismatch}))
if __name__=='__main__':
 try:main()
 except BaseException:dump(O/'ANALYSIS_STOPPED.json',{'epoch':time.time(),'error':traceback.format_exc()});raise
