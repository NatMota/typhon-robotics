"""Measure the actual 224px model preprocessing, not a guessed display resize."""
import argparse,json,sys,hashlib
from pathlib import Path
import cv2,numpy as np
from transformers import AutoProcessor
def main(a):
 a.output.mkdir(parents=True,exist_ok=False)
 idx=json.loads((a.data/'source-index.json').read_text());rows=[r for r in idx['samples']if r['view']=='camera_a'and r['valid_xy']and r['yaw_clockwise_deg']is not None]
 proc=AutoProcessor.from_pretrained(str(a.base_model),trust_remote_code=True,local_files_only=True).image_processor
 pc=proc.to_dict();assert pc['size']=={'height':224,'width':224} and not pc.get('do_center_crop',False),pc
 (a.output/'processor.json').write_text(json.dumps(pc,indent=2)+'\n')
 selected=[]
 for did in ['demo_0001','demo_0006','demo_0007','demo_0018']:
  rr=[r for r in rows if r['demo_id']==did]
  selected.extend([rr[0]['sample_id'],rr[-1]['sample_id']])
 measurements=[];panels={}
 for r in rows:
  im=cv2.imread(str(a.data/r['frame_path']));bx,by,bw,bh=r['bbox_xywh']
  for method in ['full_rgb','object_crop']:
   crop=r['crop_xyxy']if method=='object_crop'else[0,0,im.shape[1],im.shape[0]];x0,y0,x1,y1=crop;im2=im[y0:y1,x0:x1]
   px=proc([cv2.cvtColor(im2,cv2.COLOR_BGR2RGB)],return_tensors='np')['pixel_values'][0]
   rgb=np.clip(np.moveaxis(px,0,-1)*np.array(proc.image_std)+np.array(proc.image_mean),0,1)
   rgb=np.round(rgb*255).astype(np.uint8);bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
   mask=cv2.inRange(cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV),np.array([95,95,45]),np.array([130,255,255]))
   x=int(max(0,(bx-x0)/im2.shape[1]*224));y=int(max(0,(by-y0)/im2.shape[0]*224));xe=int(min(224,np.ceil((bx+bw-x0)/im2.shape[1]*224)));ye=int(min(224,np.ceil((by+bh-y0)/im2.shape[0]*224)))
   support=np.zeros_like(mask);support[y:ye,x:xe]=mask[y:ye,x:xe];pts=cv2.findNonZero(support)
   box=cv2.boundingRect(pts)if pts is not None else(0,0,0,0)
   measurements.append({'sample_id':r['sample_id'],'method':method,'blue_extent_width_px':box[2],'blue_extent_height_px':box[3],'blue_pixels':int(np.count_nonzero(support))})
   if r['sample_id']in selected:
    tile=np.full((266,224,3),18,np.uint8);tile[:224]=bgr
    cv2.putText(tile,method,(4,240),cv2.FONT_HERSHEY_SIMPLEX,.42,(240,240,240),1)
    cv2.putText(tile,f"blue extent {box[2]}x{box[3]} px",(4,258),cv2.FONT_HERSHEY_SIMPLEX,.42,(240,240,240),1)
    panels[(r['sample_id'],method)]=tile
 (a.output/'glyph-pixel-measurements.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in measurements))
 contact=[]
 for sid in selected:
  p=np.hstack([panels[(sid,'full_rgb')],panels[(sid,'object_crop')]]);header=np.full((28,448,3),18,np.uint8);cv2.putText(header,sid,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.46,(240,240,240),1);contact.append(np.vstack([header,p]))
 cv2.imwrite(str(a.output/'actual-model-inputs.jpg'),np.vstack(contact),[cv2.IMWRITE_JPEG_QUALITY,92])
 stats={}
 for m in ['full_rgb','object_crop']:
  rr=[r for r in measurements if r['method']==m];stats[m]={k:np.quantile([r[k]for r in rr],[.05,.5,.95]).tolist()for k in ['blue_extent_width_px','blue_extent_height_px','blue_pixels']}
 (a.output/'summary.json').write_text(json.dumps({'primary_samples_with_observed_yaw':len(rows),'model_input_wh':[224,224],'blue_extent_quantiles_p05_p50_p95':stats,'limitations':'Colour extent is pixel support, not glyph legibility or semantic recognition. Each sample is observed with known template yaw; missing/occluded glyph frames are excluded. Crop and full view use the actual pinned processor.'},indent=2)+'\n')
 print(json.dumps(stats),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--base-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
