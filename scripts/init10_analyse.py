"""Read-only analysis of four fixed INIT10 recordings. Never runs physics or inference."""
import argparse,json,gzip,hashlib,csv,time,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import imageio.v2 as imageio
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
R=a.source;O=a.output;O.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def dump(n,x):(O/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 with gzip.open(p,'rt') as f:return [json.loads(x) for x in f]
def same(x,y):
 if isinstance(x,dict):
  assert isinstance(y,dict) and set(x)==set(y),(set(x),set(y))
  return max([same(x[k],y[k]) for k in x]+[0.])
 if isinstance(x,list):
  assert isinstance(y,list) and len(x)==len(y)
  return max([same(u,v) for u,v in zip(x,y)]+[0.])
 if isinstance(x,(int,float)) and not isinstance(x,bool):
  assert np.isfinite(x) and np.isfinite(y);return abs(x-y)
 assert x==y,(x,y);return 0.
def sustained(flags):return any(all(flags[i:i+3]) for i in range(len(flags)-2))
def pad(c):return 'pad_collision' in c['geom1'] or 'pad_collision' in c['geom2']
def obj(c):return 'cube_g0' in (c['geom1'],c['geom2'])
def forbidden(c):return ('gripper' in c['geom1'] and any(z in c['geom2'] for z in ['table','fixture'])) or ('gripper' in c['geom2'] and any(z in c['geom1'] for z in ['table','fixture']))
f=read(R/'execution-freeze.json');summary=[];validation=[];loaded={};contacttable=[];timeseries=[]
for case in f['cases']:
 cid=case['id'];oldroot=R/'reference-inputs'/cid
 old=rows(oldroot/'controller_trace.jsonl.gz');z0=read(oldroot/'episode.json')['initial_truth']['object_position_m'][2]
 native=rows(R/'branches'/cid/'N/controller-force-trace.jsonl.gz')
 for cond in ['N','D5']:
  root=R/'branches'/cid/cond; rr=rows(root/'controller-force-trace.jsonl.gz');loaded[(cid,cond)]=rr
  assert len(rr)==60
  prefix=read(root/'prefix-validation.json')
  assert prefix['passed'] and prefix['maximum_rich_error']==0
  assert sha(root/'branch-state.pkl')==sha(R/'branches'/cid/'N/branch-state.pkl')
  extra=['original_target','applied_target','force_telemetry']
  orig=np.array([x['original_target'] for x in rr]);applied=np.array([x['applied_target'] for x in rr])
  assert np.array_equal(orig,np.array([x['target'] for x in old[60:120]]))
  expected=orig.copy();expected[:,2]-=.005 if cond=='D5' else 0
  assert np.array_equal(applied,expected) and np.array_equal(applied,np.array([x['target'] for x in rr]))
  nativeerr=max(same({k:v for k,v in x.items() if k not in extra},old[60+i]) for i,x in enumerate(rr)) if cond=='N' else None
  if cond=='N': assert nativeerr==0
  ctrl=np.array([x['force_telemetry']['ctrl'] for x in rr]);nctrl=np.array([x['force_telemetry']['ctrl'] for x in native]);assert np.array_equal(ctrl,nctrl)
  forces_finite=True;transformerror=0.;ortherr=0.;bad=0;forcecounts=0
  heights=[];tilts=[];bilat=[];apertures=[];padforces=[];actforces=[]
  for x in rr:
   ft=x['force_telemetry'];truth=x['object_truth'];b=truth['bilateral_grasp_contact']
   assert ft['activation']==[]
   for key in ['ctrl','actuator_force','finger_generalized_actuator_force','finger_qpos','finger_qvel']:
    assert np.isfinite(ft[key]).all()
   pf=[0.,0.]
   for c in ft['contacts']:
    forcecounts+=1;M=np.array(c['frame_rows_world']);w=np.array(c['wrench_contact']);assert np.isfinite(w).all()
    ortherr=max(ortherr,float(np.abs(M@M.T-np.eye(3)).max()))
    transformerror=max(transformerror,float(np.abs(M.T@w[:3]-c['force_world_contact_basis']).max()),float(np.abs(M.T@w[3:]-c['torque_world_contact_basis']).max()))
    bad+=int(forbidden(c))
    if pad(c) and obj(c):
     side=0 if 'finger1_pad' in c['geom1']+c['geom2'] else 1
     pf[side]+=max(0.,float(w[0]))
   h=(truth['object_position_m'][2]-z0)*1000;tilt=x['object_tilt_deg'];aperture=abs(ft['finger_qpos'][0]-ft['finger_qpos'][1])*1000
   heights.append(h);tilts.append(tilt);bilat.append(b);apertures.append(aperture);padforces.append(pf);actforces.append(ft['actuator_force'])
   ps=[c['object_local_position'][2]*1000 for c in x['contacts_full'] if pad(c) and obj(c)]
   ct={'case':cid,'condition':cond,'tick':x['step'],'bilateral':b,'object_rise_mm':h,'tilt_deg':tilt,'aperture_mm':aperture,'left_pad_normal_sum_N':pf[0],'right_pad_normal_sum_N':pf[1],'left_actuator_force':ft['actuator_force'][0],'right_actuator_force':ft['actuator_force'][1],'pad_local_z_min_mm':min(ps) if ps else None,'pad_local_z_max_mm':max(ps) if ps else None}
   timeseries.append(ct)
  assert bad==0 and transformerror<1e-12 and ortherr<1e-12
  first=next((i for i,b in enumerate(bilat) if b),None);last=max((i for i,b in enumerate(bilat) if b),default=None)
  firstrow=rr[first] if first is not None else None;ps=[c['object_local_position'][2]*1000 for c in firstrow['contacts_full'] if pad(c) and obj(c)] if firstrow else []
  gate10=sustained([b and h>=10 for b,h in zip(bilat,heights)])
  gate70=sustained([b and h>=70 and t<=10 for b,h,t in zip(bilat,heights,tilts)])
  held=bool(bilat[-1] and tilts[-1]<=10);gate=bool(gate70 and held and not bad)
  s=read(root/'summary.json')
  for key,val in [('supported_10mm_3samples',gate10),('upright_supported_70mm_3samples',gate70),('end_upright_bilateral',held),('retention_gate',gate)]:
   assert s[key]==val,(key,s[key],val)
  assert abs(s['maximum_brick_rise_mm']-max(heights))<1e-10 and s['full_task_success'] is None
  out={'case':cid,'condition':cond,'samples':len(rr),'native_max_rich_error':nativeerr,'first_bilateral_tick':rr[first]['step'] if first is not None else None,'last_bilateral_tick':rr[last]['step'] if last is not None else None,'bilateral_samples':sum(bilat),'supported_10mm_3samples':gate10,'upright_supported_70mm_3samples':gate70,'end_upright_bilateral':held,'retention_gate':gate,'peak_rise_mm':max(heights),'final_rise_mm':heights[-1],'final_tilt_deg':tilts[-1],'first_pad_local_z_min_mm':min(ps) if ps else None,'first_pad_local_z_max_mm':max(ps) if ps else None,'first_wrist_height_above_object_mm':(firstrow['eef_xyz'][2]-firstrow['object_truth']['object_position_m'][2])*1000 if firstrow else None,'first_bilateral_aperture_mm':apertures[first] if first is not None else None,'final_aperture_mm':apertures[-1],'first_bilateral_left_actuator_force':actforces[first][0] if first is not None else None,'first_bilateral_right_actuator_force':actforces[first][1] if first is not None else None,'median_left_pad_normal_sum_during_bilateral_N':float(np.median(np.array(padforces)[bilat,0])) if any(bilat) else None,'median_right_pad_normal_sum_during_bilateral_N':float(np.median(np.array(padforces)[bilat,1])) if any(bilat) else None,'forbidden_contact_samples':bad,'full_task_success':None}
  summary.append(out)
  validation.append({'case':cid,'condition':cond,'qualified_prefix':True,'paired_branch_semantic_bytes_equal':True,'target_change_exactly_declared':True,'actuator_ctrl_identical_to_N':True,'native_rich_error':nativeerr,'force_contact_count':forcecounts,'force_transform_max_error':transformerror,'contact_frame_orthonormal_max_error':ortherr,'all_finite':True,'independent_outcome_match':True})
dump('summary.json',summary)
dump('validation.json',{'passed':True,'branches':validation,'source_unchanged':all(sha(R/'reference-inputs'/Path(p).parent.name/Path(p).name)==h for p,h in f['source_sha256'].items()),'new_simulation_steps':0,'new_model_calls':0})
assert read(O/'validation.json')['source_unchanged']
for name,data in [('summary.csv',summary),('timeseries.csv',timeseries)]:
 with (O/name).open('w',newline='') as g:
  w=csv.DictWriter(g,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
plt.rcParams.update({'font.size':9})
fig,axs=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
for i,case in enumerate(f['cases']):
 cid=case['id']
 for cond,color in [('N','#c65d29'),('D5','#1678ba')]:
  rr=[x for x in timeseries if x['case']==cid and x['condition']==cond];t=np.arange(60)/20
  for j,key in enumerate(['object_rise_mm','tilt_deg','aperture_mm']):
   axs[i,j].plot(t,[x[key] for x in rr],color=color,label=cond)
  axs[i,3].plot(t,[x['left_pad_normal_sum_N'] for x in rr],color=color,label=cond+' left')
  axs[i,3].plot(t,[x['right_pad_normal_sum_N'] for x in rr],color=color,linestyle='--',alpha=.7,label=cond+' right')
 for j,title in enumerate(['Brick rise (mm)','Brick tilt (degrees)','Finger aperture (mm)','Pad normal-force sum (N)']):
  axs[i,j].set_title(cid.split('-')[1]+' · '+title);axs[i,j].set_xlabel('Seconds from tested segment start');axs[i,j].grid(alpha=.2);axs[i,j].legend()
 axs[i,0].axhline(70,c='grey',lw=.7,ls=':');axs[i,1].axhline(10,c='grey',lw=.7,ls=':')
fig.suptitle('INIT10: fixed 5 mm lower target path; same saved Q1 actions and grip command. Two cases only.')
fig.savefig(O/'contact-lift-force.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
for i,case in enumerate(f['cases']):
 for cond,col in [('N','#c65d29'),('D5','#1678ba')]:
  rr=[x for x in timeseries if x['case']==case['id'] and x['condition']==cond];t=np.arange(60)/20
  for j,key in enumerate(['left_actuator_force','right_actuator_force']):axs[i,j].plot(t,[x[key] for x in rr],label=cond,color=col)
 for j in range(2):axs[i,j].set_title(case['id'].split('-')[1]+' · '+['Left','Right'][j]+' actuator force');axs[i,j].set_xlabel('Seconds');axs[i,j].legend();axs[i,j].grid(alpha=.2)
fig.suptitle('Prospective actuator-force telemetry; identical ctrl sequence, different physical response')
fig.savefig(O/'actuator-forces.png',dpi=150);plt.close(fig)
# New N media for case0005 is inconsistent with its exact physical replay.
# Never substitute it silently: use the archived native footage whose full
# first120 rich samples are identical, and explicitly label that provenance.
media=[]
sheet=Image.new('RGB',(2048,2*(512+64)),(22,24,29));dr=ImageDraw.Draw(sheet)
for i,case in enumerate(f['cases']):
 cid=case['id'];oldroot=R/'reference-inputs'/cid
 ep=read(oldroot/'episode.json');begin=ep['control_step_start']
 ticks=[begin]+[t for t in range(begin+1,begin+121) if t%4==0]
 assert len(ticks)==31
 paths=[oldroot/'rollout.mp4',R/'branches'/cid/'D5/rollout.mp4']
 readers=[imageio.get_reader(str(p)) for p in paths]
 framesets=[[rd.get_data(k) for k in range(31)] for rd in readers]
 for rd in readers:rd.close()
 q1rows=rows(oldroot/'controller_trace.jsonl.gz')[:120]
 nv=rows(R/'branches'/cid/'N/trajectory.jsonl.gz')
 assert len(nv)==120
 # Explicit matching step / physical state for the full selected native segment.
 assert all(x['step']==y['step'] and same(x['truth'],y['object_truth'])==0 for x,y in zip(nv,q1rows))
 out=O/(cid+'-historical-N-vs-D5.mp4')
 with imageio.get_writer(str(out),fps=5,macro_block_size=1) as writer:
  for k,tick in enumerate(ticks):
   panel=Image.new('RGB',(2048,576),(22,24,29));d=ImageDraw.Draw(panel)
   for j,label in enumerate(['N: archived native video; replay state trace exact','D5: new recorded video; target Z minus 5 mm']):
    panel.paste(Image.fromarray(framesets[j][k]),(j*1024,64))
    d.text((j*1024+14,10),label,fill='white')
    d.text((j*1024+14,30),'control tick '+str(tick)+' | '+('common prefix' if tick<=begin+60 else 'tested acquisition / lift'),fill='white')
   writer.append_data(np.array(panel))
 for j,label in enumerate(['N: archived native frame','D5: new recorded frame']):
  sheet.paste(Image.fromarray(framesets[j][-1]),(j*1024,i*576+64))
  dr.text((j*1024+14,i*576+12),cid+' | '+label,fill='white')
  dr.text((j*1024+14,i*576+34),'Last common recorded tick '+str(ticks[-1])+'; not a placement endpoint',fill='white')
 media.append({'case':cid,'frame_ticks':ticks,'native_video_source':str(paths[0]),'native_video_sha256':sha(paths[0]),'D5_video_source':str(paths[1]),'D5_video_sha256':sha(paths[1]),'native_first120_physical_samples_exact':True,'native_new_capture_used':False,'new_native_case0005_media_contradicts_state_trace':cid.endswith('912171'),'video':out.name,'frames':31,'fps':5,'terminal_sample_is_not_necessarily_last_control_tick':True})
sheet.save(O/'final-contact-sheet.png')
dump('media-provenance.json',{'qualified_for_presentation':True,'method':'Repackage existing video only; no render or physical step. Historical N explicitly labelled and matched to exact new replay states. Original four recordings and first failed QA preserved.','defect':'New case0005 N final image shows held brick and duplicated secondary views despite exact trace showing empty grip and 90-degree brick tilt. Multiple live render contexts are a suspected cause, not a proven root cause. No model consumed these new pictures.','comparisons':media})
dump('analysis-receipt.json',{'source':str(R),'new_physics':False,'new_model_calls':0,'script_sha256':sha(Path(__file__)),'generated_files':[x.name for x in sorted(O.iterdir()) if x.is_file()],'epoch':time.time()})
print(json.dumps({'summary':summary,'validation':read(O/'validation.json'),'output':str(O)}))
