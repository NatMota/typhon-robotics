"""Read-only paired-original frame evidence for existing VIDEO01-C acceptance review."""
import argparse,json,hashlib,sys,subprocess,math
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from video09_source import save,sha
from video09_analyze import fit
def prepare(m0,out):
 out.mkdir(parents=True,exist_ok=False)
 manifest=json.loads((m0/'data/personal/m0/dataset_manifest.json').read_text());demos=manifest['demonstrations']
 assert len(demos)==16 and all(d['demo_id']!='demo_0017' for d in demos)
 protocol=json.loads((ROOT/'configs/experiments/video01-c-review/protocol.json').read_text());save(out/'protocol.json',protocol)
 requests={};metadata={};rawmeta={}
 for demo in demos:
  did=demo['demo_id'];duration=min(demo[v]['end_s']-demo[v]['start_s'] for v in ['camera_a','camera_b'])
  times=np.linspace(0,duration,7);metadata[did]={'demo_id':did,'manual_category':demo['attempt_type'],'sync_uncertainty_s':demo['sync_uncertainty_s'],
      'boundary_uncertainty_s':demo['boundary_uncertainty_s'],'annotations':demo['manual_annotations'],
      'scenario_review':demo.get('manual_scenario_review'),'task_semantics_review':demo.get('task_semantics_review'),
      'reviewed_frame_requests':[],'metadata_source':str(m0/'data/personal/m0/demonstrations'/did/'metadata.json')}
  for k,t in enumerate(times):
   for view in ['camera_a','camera_b']:
    v=demo[view];path=m0/'data/personal/m0/raw'/v['filename']
    key=str(path);rawmeta[key]={'recording_id':v['recording_id'],'sha256':v['source_sha256']}
    requests.setdefault(key,[]).append({'demo_id':did,'view':view,'sample_index':k,'relative_time_requested_s':float(t),
      'source_time_requested_s':float(v['start_s']+t),'transform':v.get('clip_transform','none'),'recording_id':v['recording_id']})
 evidence=[]
 for filename,req in requests.items():
  path=Path(filename);assert sha(path)==rawmeta[filename]['sha256']
  pts=[float(x['best_effort_timestamp_time']) for x in json.loads(subprocess.check_output(
   ['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',filename]))['frames']]
  wanted={}
  for r in req:
   idx=int(np.argmin(abs(np.asarray(pts)-r['source_time_requested_s'])));wanted.setdefault(idx,[]).append(r)
  cap=cv2.VideoCapture(filename)
  for idx in range(max(wanted)+1):
   ok,im=cap.read();assert ok
   if idx not in wanted:continue
   record=rawmeta[filename]['recording_id'];p=out/'frames'/record/f'{idx:06d}.jpg';p.parent.mkdir(parents=True,exist_ok=True)
   cv2.imwrite(str(p),im,[cv2.IMWRITE_JPEG_QUALITY,92])
   for r in wanted[idx]:
    row={**r,'raw_path':filename,'raw_sha256':rawmeta[filename]['sha256'],'source_frame_index':idx,'source_time_decoded_s':pts[idx],
     'time_quantization_error_s':pts[idx]-r['source_time_requested_s'],'decoded_bgr_sha256':hashlib.sha256(im.tobytes()).hexdigest(),
     'frame_path':str(p.relative_to(out)),'saved_jpeg_sha256':sha(p),'image_is_derivative':True}
    evidence.append(row);metadata[r['demo_id']]['reviewed_frame_requests'].append(row)
  cap.release()
 for did,demo in metadata.items():
  panels=[]
  for k in range(7):
   pair=[]
   for view in ['camera_a','camera_b']:
    r=next(r for r in demo['reviewed_frame_requests'] if r['sample_index']==k and r['view']==view)
    im=cv2.imread(str(out/r['frame_path']))
    if r['transform']=='rotate_90_degrees_counterclockwise':im=np.rot90(im).copy()
    canvas=fit(im,300,200);cv2.rectangle(canvas,(0,0),(300,26),(18,18,18),-1)
    cv2.putText(canvas,f'{view} raw {r["source_time_decoded_s"]:.3f}s f{r["source_frame_index"]}',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1);pair.append(canvas)
   pane=np.vstack(pair);cv2.putText(pane,f'{did} sample {k}',(8,393),cv2.FONT_HERSHEY_SIMPLEX,.4,(220,220,220),1);panels.append(pane)
  panels.append(np.full_like(panels[0],20));sheet=np.vstack([np.hstack(panels[:4]),np.hstack(panels[4:])])
  p=out/'contact_sheets'/(did+'.jpg');p.parent.mkdir(exist_ok=True);cv2.imwrite(str(p),sheet,[cv2.IMWRITE_JPEG_QUALITY,85])
 save(out/'review_frame_index.json',evidence);save(out/'source_ledger_inputs.json',list(metadata.values()))
 save(out/'source_manifest.json',{'curated_manifest_sha256':sha(m0/'data/personal/m0/dataset_manifest.json'),
  'raw_sources':[{'path':p,**r} for p,r in rawmeta.items()],'raw_preserved':True,'unique_raw_recordings':len(rawmeta),
  'curated_demo_count':16,'excluded_demo':'demo_0017','paired_instants':112,'view_samples':len(evidence),
  'scope':'These sampled instants are inspection evidence, not a claim that every intervening frame was reviewed.'})
 print(json.dumps({'demos':[(d['demo_id'],d['manual_category']) for d in metadata.values()],'view_samples':len(evidence),'originals_verified':len(rawmeta)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--m0',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.m0,a.output)
