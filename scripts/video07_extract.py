"""Sequential native-frame extraction, retaining exact VIDEO06 query endpoints."""
from pathlib import Path
import argparse,collections,json,hashlib,subprocess,time
import cv2,numpy as np
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
cfg=json.loads(a.protocol.read_text());data=Path(cfg['input_data']);old=Path(cfg['previous_root']);root=Path(cfg['raw_project_root']);out=a.output;out.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
assert sha(data/'source-index.json')==cfg['source_index_sha256']
rows=json.loads((data/'source-index.json').read_text())['samples'];raw={r['recording_id']:r for r in json.loads((data/'input-manifest.json').read_text())['raw']}
sel=json.loads((old/'audit-v2/selection.json').read_text());needed=collections.defaultdict(set);old_by=collections.defaultdict(list);sequences=[]
for s in sel:
 rr=[rows[i]for i in s['row_indices']];rid=rr[0]['recording_id'];f0=rr[0]['source_frame_index'];f1=rr[-1]['source_frame_index'];assert all(r['recording_id']==rid for r in rr)
 ids=list(range(f0,f1+1));needed[rid].update(ids);sequences.append({**s,'recording_id':rid,'source_start_s':rr[0]['source_start_s'],'native_frame_keys':[f'{rid}/{i}'for i in ids],'common_frame_keys':[f'{rid}/{r["source_frame_index"]}'for r in rr]})
 for r in rr:old_by[(rid,r['source_frame_index'])].append(r)
frames={};provenance=[];checked=0
for rid,ids in needed.items():
 meta=raw[rid];path=root/meta['relative_path'];assert sha(path)==meta['sha256']
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)]))
 pts=[float(f['best_effort_timestamp_time'])for f in probe['frames']];assert pts==meta['frame_pts_s'];assert np.all(np.diff(pts)>0)
 primary=rid in ['recording_0005','recording_0008'];cap=cv2.VideoCapture(str(path));assert cap.isOpened();folder=out/'frames'/rid;folder.mkdir(parents=True)
 for fid in range(max(ids)+1):
  ok,im=cap.read();assert ok,(rid,fid)
  if fid not in ids:continue
  if primary:im=cv2.rotate(im,cv2.ROTATE_90_COUNTERCLOCKWISE)
  digest=hashlib.sha256(im.tobytes()).hexdigest();target=folder/f'{fid:06d}.jpg';assert cv2.imwrite(str(target),im,[cv2.IMWRITE_JPEG_QUALITY,cfg['jpeg_quality']]);jh=sha(target)
  for r in old_by[(rid,fid)]:
   assert digest==r['decoded_bgr_sha256'];assert jh==r['frame_sha256'],(r['sample_id'],'JPEG encoder mismatch');assert abs(pts[fid]-r['source_time_s'])<1e-9;checked+=1
  frames[f'{rid}/{fid}']={'recording_id':rid,'source_frame_index':fid,'source_time_s':pts[fid],'canonical_rotate_ccw':primary,'image_wh':list(im.shape[1::-1]),'decoded_bgr_sha256':digest,'frame_sha256':jh,'frame_path':str(target.relative_to(out))}
 cap.release();provenance.append({'recording_id':rid,'raw_path':str(path),'raw_sha256':meta['sha256'],'selected_frames':len(ids),'pts_verified':len(pts),'minimum_source_gap_s':float(np.min(np.diff(pts))),'median_source_gap_s':float(np.median(np.diff(pts)))})
 print('DECODED',rid,len(ids),flush=True)
assert checked==342
save(out/'index.json',{'schema':'video07-native-observations-v1','frames':frames,'sequences':sequences});save(out/'source-provenance.json',provenance);save(out/'protocol.json',cfg)
save(out/'validation.json',{'raw_files_verified':len(provenance),'native_frames':len(frames),'common_frames_byte_identical':checked,'sequence_count':len(sequences),'no_seeking':True,'raw_pts_verified':True,'seconds':time.monotonic()-begin,'opencv':cv2.__version__,'numpy':np.__version__,'script_sha256':sha(__file__)})
print('EXTRACTION',len(frames),'common',checked,'seconds',time.monotonic()-begin)
