"""Paired cadence comparison over identical source endpoints, with explicit lost support."""
from pathlib import Path
import argparse,json,hashlib,time,math
import cv2,numpy as np
import video06_motion as m
def small(w):return {k:v for k,v in w.items()if k!='trace'}
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def compare(a):
 cfg=json.loads(a.protocol.read_text());old=Path(cfg['previous_root'])/'audit-v2';oldcfg=json.loads((old/'protocol.json').read_text());data=a.data;out=a.output;out.mkdir(parents=True,exist_ok=False);(out/'traces').mkdir();(out/'previews').mkdir();begin=time.monotonic()
 assert m.sha(Path(m.__file__))==cfg['motion_script_sha256'];idx=json.loads((data/'index.json').read_text());frames=idx['frames'];seqs=idx['sequences'];summaries=[];pairrows=[];winrows=[];cache={};panels={};native_count=0
 for s in seqs:
  keys=s['native_frame_keys'];common=s['common_frame_keys'];rr=[frames[k]for k in keys];position={k:i for i,k in enumerate(keys)};common_ix=[position[k]for k in common];ims=[];gs=[];ms=[];bgs=[];newbgs=[]
  for r in rr:
   if r['frame_path']not in cache:assert m.sha(data/r['frame_path'])==r['frame_sha256'];cache[r['frame_path']]=True
   im=cv2.imread(str(data/r['frame_path']));ims.append(im);gs.append(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY));ms.append(m.blue_mask(im,s['view'],oldcfg)[0]);bgs.append(m.roi_mask(im.shape[:2],oldcfg['reviewed_spatial_masks'][s['view']]['background_roi_normalized']));newbgs.append(m.roi_mask(im.shape[:2],cfg['background_reference_v2'][s['view']]))
  ts=[r['source_time_s']-s['source_start_s']for r in rr];g10=[gs[i]for i in common_ix];m10=[ms[i]for i in common_ix];b10=[bgs[i]for i in common_ix];nb10=[newbgs[i]for i in common_ix];t10=[ts[i]for i in common_ix]
  prior=json.loads((old/'traces'/f'{s["sequence_id"]}.json').read_text());prs=[];ws=[]
  for j in range(len(common_ix)-1):
   lo,hi=common_ix[j:j+2];base=m.estimate_pair(g10[j],g10[j+1],m10[j],m10[j+1],oldcfg);nat=m.persistent(gs[lo:hi+1],ms[lo:hi+1],ts[lo:hi+1],oldcfg)
   expected=prior['pairs'][j];assert base['qualified']==expected['qualified']and base['median_delta_px']==expected['median_delta_px']
   delta=float(np.linalg.norm(np.array(base['median_delta_px'])-np.array(nat['end_delta_px'])))if base['qualified']and nat['qualified']else None
   pr={'sequence_id':s['sequence_id'],'demo_id':s['demo_id'],'view':s['view'],'kind':s['kind'],'role':s['role'],'common_pair_index':j,'start_frame_key':common[j],'end_frame_key':common[j+1],'t0':ts[lo],'t1':ts[hi],'native_frames':hi-lo+1,'baseline_qualified':bool(base['qualified']and ts[hi]-ts[lo]<=oldcfg['low_motion_rule']['maximum_pair_gap_s']),'baseline_existing_code_qualified':base['qualified'],'baseline_gap_valid':bool(ts[hi]-ts[lo]<=oldcfg['low_motion_rule']['maximum_pair_gap_s']),'native_qualified':nat['qualified'],'baseline_delta_px':base['median_delta_px'],'native_delta_px':nat['end_delta_px'],'endpoint_disagreement_px':delta,'baseline_seed_count':base['n_seed'],'baseline_end_count':base['n_retained'],'native_seed_count':nat['seed_count'],'native_end_count':nat['end_count']};prs.append({'measurement':pr,'baseline':base,'native':nat});pairrows.append(pr)
  for j,w in enumerate(prior['windows']):
   start,end=w['start'],w['end'];lo,hi=common_ix[start],common_ix[end]
   base=m.persistent(g10[start:end+1],m10[start:end+1],t10[start:end+1],oldcfg);nat=m.persistent(gs[lo:hi+1],ms[lo:hi+1],ts[lo:hi+1],oldcfg)
   assert small(base)=={k:w[k]for k in small(base)},(s['sequence_id'],j)
   nb=m.persistent(g10[start:end+1],nb10[start:end+1],t10[start:end+1],oldcfg);nn=m.persistent(gs[lo:hi+1],newbgs[lo:hi+1],ts[lo:hi+1],oldcfg)
   ob=w['background'];on=m.persistent(gs[lo:hi+1],bgs[lo:hi+1],ts[lo:hi+1],oldcfg)
   delta=float(np.linalg.norm(np.array(base['end_delta_px'])-np.array(nat['end_delta_px'])))if base['qualified']and nat['qualified']else None
   wr={'sequence_id':s['sequence_id'],'demo_id':s['demo_id'],'view':s['view'],'kind':s['kind'],'role':s['role'],'window_index':j,'common_start_index':start,'common_end_index':end,'t0':ts[lo],'t1':ts[hi],'native_frames':hi-lo+1,'baseline':small(base),'native':small(nat),'endpoint_disagreement_px':delta,'background_old_10hz':small(ob),'background_old_native':small(on),'background_edge_10hz':small(nb),'background_edge_native':small(nn)}
   ws.append({'measurement':wr,'baseline':base,'native':nat,'background_edge_10hz':nb,'background_edge_native':nn});winrows.append(wr)
  summary={'sequence_id':s['sequence_id'],'demo_id':s['demo_id'],'view':s['view'],'kind':s['kind'],'role':s['role'],'common_frames':len(common),'native_frames':len(keys),'pairs':len(prs),'baseline_pair_qualified':sum(x['measurement']['baseline_qualified']for x in prs),'native_pair_qualified':sum(x['measurement']['native_qualified']for x in prs),'windows':len(ws),'baseline_window_qualified':sum(x['baseline']['qualified']for x in ws),'native_window_qualified':sum(x['native']['qualified']for x in ws),'new_background_10hz_qualified':sum(x['background_edge_10hz']['qualified']for x in ws),'new_background_native_qualified':sum(x['background_edge_native']['qualified']for x in ws)}
  summaries.append(summary);save(out/'traces'/f'{s["sequence_id"]}.json',{'sequence':s,'pairs':prs,'windows':ws})
  if s['kind']=='held':
   h,iw=ims[0].shape[:2];fps=1/np.median(np.diff(ts));writer=cv2.VideoWriter(str(out/'previews'/f'{s["sequence_id"]}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),float(fps),(iw,h+64));assert writer.isOpened()
   for j,im in enumerate(ims):
    canvas=cv2.copyMakeBorder(im,64,0,0,0,cv2.BORDER_CONSTANT,value=(18,18,18))
    for pt in m.features(gs[j],ms[j],oldcfg):cv2.circle(canvas,(int(round(pt[0])),int(round(pt[1]))+64),2,(255,255,0),-1)
    x0,y0,x1,y1=cfg['background_reference_v2'][s['view']];cv2.rectangle(canvas,(int(x0*iw),64+int(y0*h)),(int(x1*iw),64+int(y1*h)),(180,120,40),1)
    cv2.putText(canvas,f'{s["demo_id"]} {s["view"]} t={ts[j]:.3f}',(6,23),cv2.FONT_HERSHEY_SIMPLEX,.49,(245,245,245),1);cv2.putText(canvas,'Native frames | cyan: glyph | blue: background',(6,46),cv2.FONT_HERSHEY_SIMPLEX,.40,(220,220,220),1);writer.write(canvas)
   writer.release();native_count+=len(ims)
   # Five exact samples, one fixed image crop per view across the entire held interval.
   boxes=[]
   for mask in ms:
    pts=cv2.findNonZero(mask)
    if pts is not None:
     x,y,bw,bh=cv2.boundingRect(pts);boxes.append([x,y,x+bw,y+bh])
   bb=np.array(boxes);x0=max(0,int(bb[:,0].min())-70);y0=max(0,int(bb[:,1].min())-65);x1=min(iw,int(bb[:,2].max())+70);y1=min(h,int(bb[:,3].max())+65);tiles=[]
   for j in sorted(set(int(round(x))for x in np.linspace(0,len(ims)-1,5))):
    im=ims[j][y0:y1,x0:x1];canvas=np.full((260,320,3),18,np.uint8);scale=min(310/im.shape[1],222/im.shape[0]);im=cv2.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)));xx=(320-im.shape[1])//2;yy=34+(222-im.shape[0])//2;canvas[yy:yy+im.shape[0],xx:xx+im.shape[1]]=im;cv2.putText(canvas,f'{s["view"]} t={ts[j]:.3f}',(5,22),cv2.FONT_HERSHEY_SIMPLEX,.47,(240,240,240),1);tiles.append(canvas)
   panels.setdefault(s['demo_id'],[]).append(np.hstack(tiles))
  print('COMPARE',s['sequence_id'],summary['baseline_pair_qualified'],summary['native_pair_qualified'],'/',len(prs),'windows',summary['baseline_window_qualified'],summary['native_window_qualified'],'/',len(ws),flush=True)
 for did,lines in panels.items():cv2.imwrite(str(out/f'{did}-phase-review.jpg'),np.vstack(lines),[cv2.IMWRITE_JPEG_QUALITY,95])
 save(out/'summary.json',summaries);save(out/'pair-measurements.json',pairrows);save(out/'window-measurements.json',winrows);save(out/'protocol.json',cfg)
 save(out/'validation.json',{'sequence_count':len(summaries),'source_jpeg_hashes_reverified':len(cache),'baseline_pair_results_identical':len(pairrows),'baseline_window_results_identical':len(winrows),'native_preview_frames':native_count,'seconds':time.monotonic()-begin,'opencv':cv2.__version__,'numpy':np.__version__,'script_sha256':m.sha(__file__),'motion_script_sha256':m.sha(m.__file__),'gpu_used':False})
 print('COMPLETE',time.monotonic()-begin,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);compare(p.parse_args())
