"""Read-only INIT08 result analysis. No model calls, optimization or simulation."""
import os,sys,json,csv,gzip,hashlib,time,shutil
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[1];O=Path('/var/tmp/typhon-init08-20261015')
os.chdir(W);sys.path[:0]=[str(W),str(W/'scripts')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def writecsv(p,rows):
 with p.open('w')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 assert read(O/'complete.json')['passed']
 protocol=read(O/'protocol.json')
 measurements=read(O/'measurements.json');events=read(O/'trace-events.json')
 with np.load(O/'outputs.npz')as z:outputs={k:z[k].copy()for k in z.files}
 assert len(outputs)==len(measurements)==16
 for f,h in protocol['source_sha256'].items():assert sha(Path(f))==h,f
 by={r['id']:r for r in measurements};checks=[]
 for k,a in outputs.items():
  r=by[k];meta=read(O/'selected-records'/(r['case']+'-'+r['donor']+'.json'))
  with np.load(O/'selected-records'/(r['case']+'-'+r['donor']+'.npz'))as z:measured=z['measured_xyz'].copy()
  obj=np.asarray(meta['fixed_truth']['object_position_m']);closing=np.flatnonzero(a[:,9]>.5)
  assert np.array_equal(a[:,:3],np.asarray(r['xyz_path_m']))
  assert np.array_equal(a[:,9],np.asarray(r['gripper_probabilities']))
  assert (int(closing[0])if len(closing)else None)==r['first_close_waypoint']
  xy=np.linalg.norm(a[:,:2]-obj[:2],axis=1).min()*1000;lift=(a[:,2].max()-measured[2])*1000
  np.testing.assert_allclose([xy,lift],[r['minimum_target_xy_distance_to_object_mm'],r['maximum_target_z_above_measured_mm']],rtol=0,atol=1e-9)
  rotations=np.asarray(r['full_rotation_path'])
  np.testing.assert_allclose(rotations@rotations.transpose(0,2,1),np.eye(3)[None].repeat(30,0),rtol=0,atol=2e-5)
  checks.append({'id':k,'passed':True,'output_sha256':sha(O/'outputs.npz')})
 pairs=[];states=[];noise=[]
 for case in protocol['cases']:
  for donor in ['Q0','Q1']:
   for ni in [0,1]:
    keys=[m+'--'+case+'-'+donor+'--n'+str(ni)for m in ['Q0','Q1']]
    x,y=[by[k]for k in keys];a,b=[outputs[k]for k in keys]
    pairs.append({'case':case,'donor':donor,'noise':ni,'Q0_first_close':x['first_close_waypoint'],'Q1_first_close':y['first_close_waypoint'],'Q1_minus_Q0_close_waypoints':y['first_close_waypoint']-x['first_close_waypoint'],'xyz_mean_difference_mm':float(np.linalg.norm(b[:,:3]-a[:,:3],axis=1).mean()*1000),'xyz_max_difference_mm':float(np.linalg.norm(b[:,:3]-a[:,:3],axis=1).max()*1000),'Q0_min_target_xy_mm':x['minimum_target_xy_distance_to_object_mm'],'Q1_min_target_xy_mm':y['minimum_target_xy_distance_to_object_mm'],'Q0_target_lift_mm':x['maximum_target_z_above_measured_mm'],'Q1_target_lift_mm':y['maximum_target_z_above_measured_mm']})
  for model in ['Q0','Q1']:
   for ni in [0,1]:
    x,y=[by[model+'--'+case+'-'+d+'--n'+str(ni)]for d in ['Q0','Q1']]
    states.append({'case':case,'model':model,'noise':ni,'S0_first_close':x['first_close_waypoint'],'S1_first_close':y['first_close_waypoint'],'S0_target_lift_mm':x['maximum_target_z_above_measured_mm'],'S1_target_lift_mm':y['maximum_target_z_above_measured_mm'],'S0_min_xy_mm':x['minimum_target_xy_distance_to_object_mm'],'S1_min_xy_mm':y['minimum_target_xy_distance_to_object_mm']})
   for donor in ['Q0','Q1']:
    a,b=[outputs[model+'--'+case+'-'+donor+'--n'+str(i)]for i in [0,1]]
    noise.append({'case':case,'model':model,'donor':donor,'max_xyz_difference_mm':float(np.linalg.norm(a[:,:3]-b[:,:3],axis=1).max()*1000),'max_gripper_probability_difference':float(np.abs(a[:,9]-b[:,9]).max())})
 writecsv(O/'matched-weight-contrasts.csv',pairs);writecsv(O/'matched-state-contrasts.csv',states);writecsv(O/'noise-contrasts.csv',noise)
 summary=[]
 for donor in ['Q0','Q1']:
  for model in ['Q0','Q1']:
   rows=[r for r in measurements if r['checkpoint']==model and r['donor']==donor]
   summary.append({'input_state':donor,'model':model,'outputs':len(rows),'first_close_waypoints':sorted(set(r['first_close_waypoint']for r in rows)),'target_xy_distance_mm_min':min(r['minimum_target_xy_distance_to_object_mm']for r in rows),'target_xy_distance_mm_max':max(r['minimum_target_xy_distance_to_object_mm']for r in rows),'target_lift_mm_min':min(r['maximum_target_z_above_measured_mm']for r in rows),'target_lift_mm_max':max(r['maximum_target_z_above_measured_mm']for r in rows)})
 dump(O/'contrast-summary.json',{'groups':summary,'noise_max_xyz_difference_mm':max(r['max_xyz_difference_mm']for r in noise),'limits':'Two deliberately selected lost cases; proposals are not physically executed and cannot establish grasp success. Donor packets differ jointly in images and previous-target proprioception. No mechanism isolated within that package.'})
 media=O/'media';media.mkdir(exist_ok=True)
 fig,axes=plt.subplots(2,3,figsize=(15,7),constrained_layout=True)
 for ci,case in enumerate(protocol['cases']):
  for mi,model in enumerate(['T','Q0','Q1']):
   base=Path('/var/tmp/typhon-init02-20261015'if model=='T'else'/var/tmp/typhon-init07-20261015')/'episodes'/model/case
   ep=read(base/'episode.json')
   with gzip.open(base/'controller_trace.jsonl.gz','rt')as f:tr=[json.loads(s)for s in f][:450]
   initial=ep['initial_truth'];t=np.array([r['step']-initial['control_steps']for r in tr])*.05
   z0=initial['object_position_m'][2];w0=initial['eef_position_m'][2]
   wrist=(np.array([r['eef_xyz'][2]for r in tr])-w0)*1000
   target=(np.array([r['target'][2]for r in tr])-w0)*1000
   brick=(np.array([r['object_truth']['object_position_m'][2]for r in tr])-z0)*1000
   bi=np.array([r['object_truth']['bilateral_grasp_contact']for r in tr])
   ax=axes[ci,mi];ax.plot(t,target,label='Requested wrist delta Z',color='#7b5cc4',lw=1);ax.plot(t,wrist,label='Measured wrist delta Z',color='#1685aa',lw=1);ax.plot(t,brick,label='Brick rise',color='#d97614',lw=1.5)
   ax.fill_between(t,0,1,where=bi,transform=ax.get_xaxis_transform(),color='#59aa68',alpha=.12,label='Both pads in contact')
   ax.set_title(model+' / '+case.split('-')[1]);ax.set_xlabel('Seconds from task start');ax.set_ylabel('Vertical change (mm)');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=7,loc='lower right')
 fig.suptitle('Archived physical execution: first 22.5 seconds, no new rollouts')
 fig.savefig(media/'acquisition-traces.png',dpi=130);plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
 for ci,case in enumerate(protocol['cases']):
  for di,donor in enumerate(['Q0','Q1']):
   ax=axes[ci,di]
   for model,color in [('Q0','#2864b4'),('Q1','#d35f20')]:
    for ni,ls in [(0,'-'),(1,'--')]:
     r=by[model+'--'+case+'-'+donor+'--n'+str(ni)]
     p=np.array(r['xyz_path_m'])
     with np.load(O/'selected-records'/(case+'-'+donor+'.npz'))as z:ref=z['measured_xyz']
     ax.plot(np.arange(30),(p[:,2]-ref[2])*1000,color=color,ls=ls,label=model+' noise '+str(ni))
     ix=r['first_close_waypoint'];ax.scatter(ix,(p[ix,2]-ref[2])*1000,color=color,marker='x',s=40)
   ax.set_title(case.split('-')[1]+' / '+donor+' input');ax.set_xlabel('Predicted waypoint (zero based)');ax.set_ylabel('Target Z above measured wrist (mm)');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8)
 fig.suptitle('Unexecuted action proposals; crosses indicate first CLOSE')
 fig.savefig(media/'proposal-height-and-close.png',dpi=130);plt.close(fig)
 sheet=Image.new('RGB',(1024,1104),'#101a26');draw=ImageDraw.Draw(sheet)
 for ci,case in enumerate(protocol['cases']):
  for di,donor in enumerate(['Q0','Q1']):
   with np.load(O/'selected-records'/(case+'-'+donor+'.npz'))as z:a=z['primary'].copy()
   assert a.ndim==3 and a.shape[-1]==3
   im=Image.fromarray(a).convert('RGB');im.thumbnail((512,512))
   x=di*512;y=ci*552;draw.text((x+8,y+8),case+' / '+donor+' pre-contact',fill='white');sheet.paste(im,(x,y+32))
 sheet.save(media/'precontact-inputs.png')
 for f,h in protocol['source_sha256'].items():assert sha(Path(f))==h,f
 dump(O/'analysis-validation.json',{'passed':True,'epoch':time.time(),'outputs_independently_rechecked':16,'source_hashes_unchanged':True,'new_model_calls':0,'new_physics':0,'new_training':0,'figures':[{'path':str(p.relative_to(O)),'sha256':sha(p)}for p in sorted(media.glob('*.png'))]})
 print(json.dumps({'passed':True,'groups':summary,'noise_max_xyz_difference_mm':max(r['max_xyz_difference_mm']for r in noise)}))
if __name__=='__main__':main()
