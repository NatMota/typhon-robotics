"""Independent motion-summary, provenance and perturbation checks; keep audit-v1 immutable."""
from pathlib import Path
import argparse,json,hashlib,math
import cv2,numpy as np
import video06_motion as m
def arr(x):return np.array(x,np.float64)
def metrics(w):
 t=w['trace'];initial={i:p for i,p in zip(t[0]['ids'],t[0]['points'])};prev=initial.copy();paths={i:0. for i in initial};maxpoint=0.
 for step in t[1:]:
  for i,p in zip(step['ids'],step['points']):
   p=arr(p);maxpoint=max(maxpoint,float(np.linalg.norm(p-arr(initial[i]))));paths[i]+=float(np.linalg.norm(p-arr(prev[i])));prev[i]=p
 final=t[-1];ang=scale=rms=None
 if len(final['ids'])>=3:
  p=arr([initial[i]for i in final['ids']]);q=arr(final['points']);pc=p-p.mean(0);qc=q-q.mean(0);u,sv,vt=np.linalg.svd(pc.T@qc);fix=np.eye(2);fix[-1,-1]=np.linalg.det(u@vt);rot=u@fix@vt;scale=float(np.sum(sv*np.diag(fix))/max(1e-9,np.sum(pc*pc)));pred=scale*pc@rot+q.mean(0);rms=float(np.sqrt(np.mean(np.sum((pred-q)**2,axis=1))));ang=float(np.degrees(np.arctan2(rot[0,1],rot[0,0])))
 return {'max_any_tracked_feature_displacement_px':maxpoint if w['qualified']else None,'max_any_tracked_feature_path_px':max(paths.values())if w['qualified']and paths else None,'apparent_2d_similarity_angle_clockwise_deg':ang if w['qualified']else None,'apparent_similarity_scale':scale if w['qualified']else None,'similarity_rms_px':rms if w['qualified']else None}
def tile(im,label,width=540,height=300):
 canvas=np.full((height,width,3),18,np.uint8);s=min(width/im.shape[1],(height-30)/im.shape[0]);mini=cv2.resize(im,(round(im.shape[1]*s),round(im.shape[0]*s)));x=(width-mini.shape[1])//2;y=30+(height-30-mini.shape[0])//2;canvas[y:y+mini.shape[0],x:x+mini.shape[1]]=mini;cv2.putText(canvas,label,(6,20),cv2.FONT_HERSHEY_SIMPLEX,.47,(240,240,240),1);return canvas
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--analysis-dir',default='analysis-v3');p.add_argument('--audit-dir',default='audit-v2');a=p.parse_args();audit=a.root/a.audit_dir;out=a.root/a.analysis_dir;out.mkdir(parents=True,exist_ok=False)
cfg=json.loads((audit/'protocol.json').read_text());d=Path(cfg['input_data']);rows=json.loads((d/'source-index.json').read_text())['samples'];summary=json.loads((audit/'summary.json').read_text());sel=json.loads((audit/'selection.json').read_text());derived=[];checks=[];pairs=[];videos=[]
for seq in sel:
 tr=json.loads((audit/'traces'/f'{seq["sequence_id"]}.json').read_text());ss=next(s for s in summary if s['sequence_id']==seq['sequence_id'])
 for w in tr['windows']:
  mm=metrics(w);strict=bool(w['qualified']and w['background']['qualified']and mm['max_any_tracked_feature_displacement_px']<=3 and mm['max_any_tracked_feature_path_px']<=5 and w['background']['path_px']<=2)
  derived.append({'sequence_id':seq['sequence_id'],'kind':seq['kind'],'role':seq['role'],'t0':w['t0'],'t1':w['t1'],'midpoint':bool(ss['midpoint']and w['start']==ss['midpoint']['start']),'qualified':w['qualified'],'n_seed':w['seed_count'],'n_end':w['end_count'],'median_translation_max_px':w['max_displacement_px'],'median_translation_path_px':w['path_px'],'background_qualified':w['background']['qualified'],'background_path_px':w['background']['path_px'],'old_endpoint_displacement_px':w['old_endpoint_displacement_px'],**mm,'strict_low_motion_candidate':strict})
 for pr in tr['pairs']:
  pairs.append({'sequence_id':seq['sequence_id'],'kind':seq['kind'],**{k:pr[k]for k in ['row0','row1','time0','time1','qualified','n_seed','n_retained','old_valid_both','median_delta_px','old_centroid_delta_px','flow_minus_old_delta_norm_px']}})
 if seq['kind']=='stationary_control':
  r=rows[seq['row_indices'][0]];im=cv2.imread(str(d/r['frame_path']));g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);mask,_=m.blue_mask(im,seq['view'],cfg)
  for delta in [(0.,0.),(4.,-3.),(-4.,3.)]:
   M=np.array([[1,0,delta[0]],[0,1,delta[1]]],np.float32);h,w=g.shape;nextim=cv2.warpAffine(im,M,(w,h),flags=cv2.INTER_LINEAR);nmask,_=m.blue_mask(nextim,seq['view'],cfg);n=cv2.cvtColor(nextim,cv2.COLOR_BGR2GRAY);res=m.estimate_pair(g,n,mask,nmask,cfg)
   err=float(np.linalg.norm(arr(res['median_delta_px'])-arr(delta)))if res['qualified']else None
   assert not res['qualified']or err<.35,(seq['sequence_id'],delta,res,err)
   if delta==(0.,0.):assert res['qualified'],seq['sequence_id']
   checks.append({'sequence_id':seq['sequence_id'],'known_delta_px':list(delta),'measured_delta_px':res['median_delta_px'],'error_px':err,'qualified':res['qualified']})
  blank=np.zeros_like(g);res=m.estimate_pair(g,blank,mask,blank,cfg);assert not res['qualified']and res['median_delta_px']is None
 if seq['kind']=='held':
  path=audit/'previews'/f'{seq["sequence_id"]}.mp4';cap=cv2.VideoCapture(str(path));n=0
  while True:
   ok,frame=cap.read()
   if not ok:break
   n+=1
  cap.release();assert n==len(seq['row_indices']);videos.append({'file':str(path.relative_to(a.root)),'frames':n,'sha256':m.sha(path)})
m.save(out/'window-measurements.json',derived);m.save(out/'pair-measurements.json',pairs)
m.save(out/'validation.json',{'known_image_translation_checks':len(checks),'qualified_translation_checks':sum(x['qualified']for x in checks),'rejected_translation_checks':sum(not x['qualified']for x in checks),'max_translation_error_px':max(x['error_px']for x in checks if x['qualified']),'blank_occlusion_checks':12,'preview_videos_decoded':len(videos),'total_preview_frames':sum(x['frames']for x in videos),'flow_units':'pixels in850px-long-edge images; no metric or3D validity','pointwise_qualification':'max surviving-feature displacement and accumulated path, not group median translation, so rotation cannot cancel into a false pause','check_details':checks,'videos':videos})
# One fixed crop for all times in each row. Source context and absolute displacement remain visible.
panels=[];prov=[]
for did,view in [('demo_0012','camera_a'),('demo_0013','camera_a'),('demo_0015','camera_a'),('demo_0015','camera_b')]:
 seq=next(s for s in sel if s['sequence_id']==f'{did}-{view}-held');ss=next(s for s in summary if s['sequence_id']==seq['sequence_id']);mid=ss['midpoint'];js=[mid['start'],(mid['start']+mid['end'])//2,mid['end']];rr=[rows[seq['row_indices'][j]]for j in js];ims=[cv2.imread(str(d/r['frame_path']))for r in rr];boxes=[]
 for im in ims:
  mask,_=m.blue_mask(im,view,cfg);pts=cv2.findNonZero(mask)
  if pts is not None:
   x,y,w,h=cv2.boundingRect(pts);boxes.append([x,y,x+w,y+h])
 boxes=arr(boxes);h,w=ims[0].shape[:2];x0=max(0,int(boxes[:,0].min())-70);y0=max(0,int(boxes[:,1].min())-65);x1=min(w,int(boxes[:,2].max())+70);y1=min(h,int(boxes[:,3].max())+65);tiles=[]
 for r,im in zip(rr,ims):
  tiles.append(tile(im[y0:y1,x0:x1],f'{did} {view} {r["clip_time_s"]:.2f}s',width=360,height=260));prov.append({'sample_id':r['sample_id'],'frame_sha256':r['frame_sha256'],'fixed_crop_xyxy':[x0,y0,x1,y1]})
 panels.append(np.hstack(tiles))
cv2.imwrite(str(out/'fixed-crop-motion-card.jpg'),np.vstack(panels),[cv2.IMWRITE_JPEG_QUALITY,95]);m.save(out/'motion-card-provenance.json',prov)
# Highest old-centroid / feature-flow disagreement, fixed review triplets, all outcomes retained.
top=sorted([p for p in pairs if p['kind']=='held'and p['flow_minus_old_delta_norm_px']is not None],key=lambda p:p['flow_minus_old_delta_norm_px'],reverse=True)[:6];panels=[]
for pr in top:
 tiles=[]
 for i in [pr['row0'],pr['row1']]:
  r=rows[i];im=cv2.imread(str(d/r['frame_path']));mask,_=m.blue_mask(im,r['view'],cfg)
  for pt in m.features(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),mask,cfg):cv2.circle(im,tuple(np.rint(pt).astype(int)),2,(255,255,0),-1)
  if r['valid_xy']:
   pt=np.rint(arr(r['xy_normalized'])*arr(r['image_wh'])).astype(int);cv2.drawMarker(im,tuple(pt),(255,0,255),cv2.MARKER_CROSS,18,2)
  tiles.append(tile(im,f'{r["demo_id"]} {r["view"]} t={r["clip_time_s"]:.2f}',540,330))
 panel=np.hstack(tiles);foot=np.full((35,1080,3),18,np.uint8);cv2.putText(foot,f'Flow vs old centre delta difference: {pr["flow_minus_old_delta_norm_px"]:.2f}px | cyan glyph corners, magenta old centre',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1);panels.append(np.vstack([panel,foot]))
cv2.imwrite(str(out/'largest-disagreement.jpg'),np.vstack(panels),[cv2.IMWRITE_JPEG_QUALITY,94]);m.save(out/'largest-disagreement.json',top)
print(json.dumps({'window_count':len(derived),'strict_low_motion_held':sum(x['strict_low_motion_candidate']for x in derived if x['kind']=='held'),'strict_low_motion_control':sum(x['strict_low_motion_candidate']for x in derived if x['kind']!='held'),'translation_checks':len(checks),'max_error':max(x['error_px']for x in checks if x['qualified'])}))
