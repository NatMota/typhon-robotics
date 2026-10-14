"""Offline visible-glyph motion audit. No image point is a 3D object pose."""
from pathlib import Path
import argparse,hashlib,json,math,time,subprocess
import cv2,numpy as np
cv2.setNumThreads(1);cv2.setRNGSeed(121206)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def roi_mask(shape,box):
 h,w=shape;mask=np.zeros(shape,np.uint8);x0,y0,x1,y1=box;mask[round(y0*h):round(y1*h),round(x0*w):round(x1*w)]=255;return mask
def blue_mask(im,view,cfg):
 c=cfg['tracking'];hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);m=cv2.inRange(hsv,np.array(c['blue_hsv_min']),np.array(c['blue_hsv_max']))
 m=cv2.bitwise_and(m,roi_mask(m.shape,cfg['reviewed_spatial_masks'][view]['glyph_roi_normalized']))
 n,lab,st,cent=cv2.connectedComponentsWithStats(m,8)
 if n<2:return np.zeros_like(m),0
 k=1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]));area=int(st[k,cv2.CC_STAT_AREA])
 if area<10:return np.zeros_like(m),area
 raw=(lab==k).astype(np.uint8)*255;pad=c['support_dilation_px'];return cv2.dilate(raw,np.ones((pad*2+1,pad*2+1),np.uint8)),area
def features(gray,mask,cfg):
 c=cfg['tracking'];p=cv2.goodFeaturesToTrack(gray,maxCorners=c['max_corners'],qualityLevel=c['quality_level'],minDistance=c['min_distance_px'],mask=mask,blockSize=3)
 return np.empty((0,2),np.float32)if p is None else p.reshape(-1,2)
def flow(g0,g1,p,mask1,cfg):
 c=cfg['tracking']
 if len(p)==0:return p,np.zeros(0,bool),{'fb_error_px':[],'lk_error':[]}
 kw=dict(winSize=(c['window_px'],)*2,maxLevel=c['max_level'],criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
 q,st,e=cv2.calcOpticalFlowPyrLK(g0,g1,p.reshape(-1,1,2),None,**kw)
 if q is None:return p,np.zeros(len(p),bool),{'fb_error_px':[None]*len(p),'lk_error':[None]*len(p)}
 back,bst,_=cv2.calcOpticalFlowPyrLK(g1,g0,q,None,**kw)
 q=q.reshape(-1,2);back=back.reshape(-1,2);err=np.linalg.norm(back-p,axis=1);le=e.reshape(-1)
 h,w=g1.shape;xy=np.rint(q).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h);support=np.zeros(len(p),bool);support[inside]=mask1[xy[inside,1],xy[inside,0]]>0
 ok=(st.ravel()>0)&(bst.ravel()>0)&np.isfinite(q).all(axis=1)&(err<=c['fb_error_max_px'])&(le<=c['lk_error_max'])&support
 return q,ok,{'fb_error_px':[float(x)if st.ravel()[i] and bst.ravel()[i] and math.isfinite(x)else None for i,x in enumerate(err)],'lk_error':[float(x)if st.ravel()[i] and math.isfinite(x)else None for i,x in enumerate(le)],'forward_status':st.ravel().tolist(),'backward_status':bst.ravel().tolist()}
def sufficient(n,n0,cfg):return n>=cfg['tracking']['min_retained_points']and n>=n0*cfg['tracking']['min_retained_fraction']
def vec(x):return [float(v)for v in x]
def estimate_pair(g0,g1,m0,m1,cfg):
 p=features(g0,m0,cfg);q,ok,diag=flow(g0,g1,p,m1,cfg);n=int(ok.sum());good=sufficient(n,len(p),cfg)
 return {'seed_points':p.tolist(),'next_points':q.tolist(),'retained':ok.tolist(),**diag,'n_seed':len(p),'n_retained':n,'qualified':good,'median_delta_px':vec(np.median(q[ok]-p[ok],axis=0))if good else None}
def persistent(gs,ms,ts,cfg):
 p=features(gs[0],ms[0],cfg);orig=p.copy();ids=np.arange(len(p));n0=len(p);trace=[{'n':n0,'ids':ids.tolist(),'points':p.tolist(),'median_from_first_px':[0.,0.]}];qual=sufficient(n0,n0,cfg);path=0.;lastdelta=np.zeros(2);maxdisp=0.
 for j in range(1,len(gs)):
  q,ok,diag=flow(gs[j-1],gs[j],p,ms[j],cfg);q=q[ok];ids=ids[ok];n=len(ids)
  good=sufficient(n,n0,cfg)and ts[j]-ts[j-1]<=cfg['low_motion_rule']['maximum_pair_gap_s'];qual=qual and good
  delta=np.median(q-orig[ids],axis=0)if n else None
  # Use the same surviving features on both sides of each step, avoiding re-centring from dropped points.
  step=np.median(q-p[ok],axis=0)if good else None
  if step is not None:path+=float(np.linalg.norm(step))
  if delta is not None:maxdisp=max(maxdisp,float(np.linalg.norm(delta)))
  trace.append({'n':n,'ids':ids.tolist(),'points':q.tolist(),'median_from_first_px':vec(delta)if delta is not None else None,'qualified_step':good})
  p=q
 return {'seed_count':n0,'end_count':len(ids),'qualified':qual,'max_displacement_px':maxdisp if qual else None,'path_px':path if qual else None,'end_delta_px':trace[-1]['median_from_first_px']if qual else None,'trace':trace}
def run(a):
 begin=time.monotonic();cfg=json.loads(a.protocol.read_text());d=Path(cfg['input_data']);out=a.output;out.mkdir(parents=True,exist_ok=False)
 assert sha(d/'source-index.json')==cfg['source_index_sha256']
 rows=json.loads((d/'source-index.json').read_text())['samples'];sel=json.loads(a.selection.read_text())
 save(out/'protocol.json',cfg);save(out/'selection.json',sel);out.joinpath('previews').mkdir();out.joinpath('traces').mkdir()
 allsum=[];allsources={};midpanels=[]
 for s in sel:
  rr=[rows[i]for i in s['row_indices']];ims=[];ms=[];areas=[];gs=[];bg=[]
  for r in rr:
   assert sha(d/r['frame_path'])==r['frame_sha256'];im=cv2.imread(str(d/r['frame_path']));assert list(im.shape[1::-1])==r['image_wh']
   m,area=blue_mask(im,s['view'],cfg);ims.append(im);ms.append(m);areas.append(area);gs.append(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY));bg.append(roi_mask(im.shape[:2],cfg['reviewed_spatial_masks'][s['view']]['background_roi_normalized']))
   allsources[r['sample_id']]={k:r[k]for k in ['sample_id','demo_id','view','recording_id','source_frame_index','source_time_s','clip_time_s','frame_path','frame_sha256','image_wh']}
  ts=[r['clip_time_s']for r in rr];pairs=[];overlays=[]
  for j in range(len(rr)-1):
   p=estimate_pair(gs[j],gs[j+1],ms[j],ms[j+1],cfg);b=estimate_pair(gs[j],gs[j+1],bg[j],bg[j+1],cfg);r0,r1=rr[j:j+2];wh=np.array(r0['image_wh'])
   olddelta=vec((np.array(r1['xy_normalized'])-np.array(r0['xy_normalized']))*wh)if r0['valid_xy']and r1['valid_xy']else None
   p.update({'row0':s['row_indices'][j],'row1':s['row_indices'][j+1],'time0':ts[j],'time1':ts[j+1],'old_centroid_delta_px':olddelta,'old_valid_both':olddelta is not None,'blue_area0':areas[j],'blue_area1':areas[j+1],'background':b})
   p['flow_minus_old_delta_norm_px']=float(np.linalg.norm(np.array(p['median_delta_px'])-np.array(olddelta)))if p['qualified']and olddelta is not None else None;pairs.append(p)
  windows=[]
  starts=list(range(max(0,len(rr)-6)))
  for start in starts:
   end=min(range(start+1,len(rr)),key=lambda j:abs(ts[j]-ts[start]-.6))
   if abs(ts[end]-ts[start]-.6)>.055:continue
   w=persistent(gs[start:end+1],ms[start:end+1],ts[start:end+1],cfg);b=persistent(gs[start:end+1],bg[start:end+1],ts[start:end+1],cfg)
   w.update({'start':start,'end':end,'t0':ts[start],'t1':ts[end],'background':b})
   r0,r1=rr[start],rr[end];w['old_endpoint_displacement_px']=float(np.linalg.norm((np.array(r1['xy_normalized'])-np.array(r0['xy_normalized']))*np.array(r0['image_wh'])))if r0['valid_xy']and r1['valid_xy']else None
   w['low_motion_candidate']=bool(w['qualified']and b['qualified']and w['max_displacement_px']<=cfg['low_motion_rule']['maximum_supported_feature_displacement_px']and w['path_px']<=cfg['low_motion_rule']['maximum_supported_feature_path_px']and b['path_px']<=cfg['low_motion_rule']['maximum_background_path_px'])
   windows.append(w)
  midpoint=sum(s['requested_interval_s'])/2;mid=min(windows,key=lambda w:abs((w['t0']+w['t1'])/2-midpoint))if windows else None
  summary={**{k:s[k]for k in ['sequence_id','demo_id','view','kind','role']},'frames':len(rr),'old_visible_frames':sum(r['valid_xy']for r in rr),'blue_supported_frames':sum(x>=10 for x in areas),'pair_count':len(pairs),'qualified_pairs':sum(p['qualified']for p in pairs),'median_pair_difference_px':float(np.median([p['flow_minus_old_delta_norm_px']for p in pairs if p['flow_minus_old_delta_norm_px']is not None]))if any(p['flow_minus_old_delta_norm_px']is not None for p in pairs)else None,'rolling_windows':len(windows),'qualified_windows':sum(w['qualified']for w in windows),'low_motion_candidates':sum(w['low_motion_candidate']for w in windows),'midpoint':{k:v for k,v in mid.items()if k not in ['trace','background']}if mid else None}
  save(out/'traces'/f'{s["sequence_id"]}.json',{'sequence':s,'pairs':pairs,'windows':windows});allsum.append(summary)
  if s['kind']=='held':
   # Full-resolution source images with an added header; colours: cyan flow, magenta old centre.
   path=out/'previews'/f'{s["sequence_id"]}.mp4';h,w=ims[0].shape[:2];vw=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10.,(w,h+64));assert vw.isOpened()
   for j,im in enumerate(ims):
    canvas=cv2.copyMakeBorder(im,64,0,0,0,cv2.BORDER_CONSTANT,value=(18,18,18));p=pairs[min(j,len(pairs)-1)]
    for pt in features(gs[j],ms[j],cfg):cv2.circle(canvas,(round(float(pt[0])),round(float(pt[1]))+64),2,(255,255,0),-1)
    r=rr[j]
    if r['valid_xy']:
     x,y=np.array(r['xy_normalized'])*np.array(r['image_wh']);cv2.drawMarker(canvas,(round(x),round(y)+64),(255,0,255),cv2.MARKER_CROSS,14,2)
    cv2.putText(canvas,f'{s["demo_id"]} {s["view"]} t={ts[j]:.2f}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(245,245,245),1)
    cv2.putText(canvas,'cyan glyph corners | magenta old visible-centre',(8,43),cv2.FONT_HERSHEY_SIMPLEX,.41,(210,210,210),1)
    vw.write(canvas);overlays.append(canvas)
   vw.release()
   if mid:
    tiles=[]
    for j in [mid['start'],(mid['start']+mid['end'])//2,mid['end']]:
     pts=cv2.findNonZero(ms[j]);x,y,bw,bh=cv2.boundingRect(pts)if pts is not None else(0,0,im.shape[1],im.shape[0]);cx=x+bw/2;cy=y+bh/2;half=max(bw,bh)*1.25+12
     crop=ims[j][max(0,int(cy-half)):min(h,int(cy+half)),max(0,int(cx-half)):min(w,int(cx+half))]
     # Crops are review-only and follow the glyph. Full-frame overlays preserve absolute displacement.
     tile=np.full((285,350,3),18,np.uint8);scale=min(340/crop.shape[1],230/crop.shape[0]);tiny=cv2.resize(crop,(round(crop.shape[1]*scale),round(crop.shape[0]*scale)));xx=(350-tiny.shape[1])//2;yy=45+(230-tiny.shape[0])//2;tile[yy:yy+tiny.shape[0],xx:xx+tiny.shape[1]]=tiny
     cv2.putText(tile,f'{s["demo_id"]} {s["view"]} t={ts[j]:.2f}',(5,22),cv2.FONT_HERSHEY_SIMPLEX,.47,(240,240,240),1);tiles.append(tile)
    midpanels.append(np.hstack(tiles))
  print('AUDIT',s['sequence_id'],summary['qualified_pairs'],'/',len(pairs),'windows',summary['low_motion_candidates'],'/',len(windows),flush=True)
 for k in range(0,len(midpanels),4):cv2.imwrite(str(out/f'midpoint-crops-{k//4+1}.jpg'),np.vstack(midpanels[k:k+4]),[cv2.IMWRITE_JPEG_QUALITY,95])
 save(out/'summary.json',allsum);save(out/'source-provenance.json',list(allsources.values()));save(out/'environment.json',{'opencv':cv2.__version__,'numpy':np.__version__,'seconds':time.monotonic()-begin,'script_sha256':sha(__file__),'source_index_sha256':cfg['source_index_sha256'],'source_frames_verified':len(allsources),'precision':'float32 opticalflow','gpu_used':False})
 print('COMPLETE',len(allsum),'seconds',time.monotonic()-begin,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
