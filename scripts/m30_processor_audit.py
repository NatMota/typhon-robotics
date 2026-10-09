"""Inspect actual checkpoint processor tensors and vision entry; no optimization."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import time
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor
from supervisor.adaptation.observations import views, object_crop


def glyph(rgb):
    r,g,b=rgb.astype(np.int16).transpose(2,0,1)
    m=((b>100)&(b-r>55)&(b-g>35)&(r<120)).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(m,8)
    if n>1:m=(labels==(1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
    ys,xs=np.where(m); out=np.zeros((96,96),bool)
    if not len(xs):return out,[0,0],0
    crop=m[ys.min():ys.max()+1,xs.min():xs.max()+1];h,w=crop.shape;s=84/max(h,w)
    small=cv2.resize(crop,(max(1,round(w*s)),max(1,round(h*s))),interpolation=cv2.INTER_NEAREST)
    y=(96-small.shape[0])//2;x=(96-small.shape[1])//2;out[y:y+small.shape[0],x:x+small.shape[1]]=small>0
    return out,[w,h],int(m.sum())


def main(source,out,versions):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    checkpoint=Path('.cache-xvla/checkpoint')
    processor=AutoProcessor.from_pretrained(checkpoint,trust_remote_code=True,local_files_only=True)
    processor.language_max_length=128
    model,info=AutoModel.from_pretrained(checkpoint,trust_remote_code=True,local_files_only=True,output_loading_info=True,torch_dtype=torch.float32)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=model.cuda().eval()
    captured=[]
    original=model.vlm._encode_image
    def encode(x):
        captured.append({'shape':list(x.shape),'dtype':str(x.dtype),'range':[float(x.min()),float(x.max())]})
        return original(x)
    model.vlm._encode_image=encode
    raw_template=np.asarray(Image.open('assets/duplo_31110pb074_v2/numeral_2_mask.png').convert('L'))>127
    fake=np.zeros((*raw_template.shape,3),np.uint8);fake[raw_template]=[0,0,255]
    templates=[glyph(np.rot90(fake,k))[0] for k in range(4)]
    records=json.loads((source/'manifest.json').read_text())['records']; rows=[]; timings=[]
    for version in versions:
        folder=out/version;folder.mkdir()
        sheet=Image.new('RGB',(4*224,3*256),'white');draw=ImageDraw.Draw(sheet)
        for index,record in enumerate(records):
            rgb={v:np.asarray(Image.open(source/f'{record["id"]}-{v}.png').convert('RGB')) for v in ['primary','secondary']}
            inputs=processor(images=views(rgb,version),language_instruction='Place the orange block in the staging square with blue 2 upright.')
            pixels=inputs['image_input']
            # Save de-normalized *actual* tensors, not an independent resize approximation.
            mean=torch.tensor(processor.image_processor.image_mean).view(3,1,1)
            std=torch.tensor(processor.image_processor.image_std).view(3,1,1)
            for vi,name in enumerate(['primary','secondary','detail'] if version=='O2' else ['primary','secondary']):
                reconstructed=((pixels[0,vi]*std+mean)*255).round().clamp(0,255).byte().permute(1,2,0).numpy()
                Image.fromarray(reconstructed).save(folder/f'{record["id"]}-{name}.png')
                if vi==(2 if version=='O2' else 0):
                    mask,bounds,count=glyph(reconstructed)
                    scores=[float((mask&t).sum()/max(1,(mask|t).sum())) for t in templates];rank=np.argsort(scores)[::-1]
                    angle=90*int(rank[0]);margin=scores[rank[0]]-scores[rank[1]]
                    row={**record,'observation_version':version,'bounds':bounds,'blue_pixels':count,
                        'predicted_yaw':angle,'margin':margin,'tensor_shape':list(pixels.shape),'image_mask':inputs['image_mask'].tolist(),
                        'crop_metadata':object_crop(rgb['primary'])[1] if version=='O2' else None}
                    if record['type']=='orientation_fixture':
                        row['legibility_pass']=min(bounds)>=7 and max(bounds)>=12 and count>=30 and angle==record['known_fixture_yaw'] and margin>=.05
                        sheet.paste(Image.fromarray(reconstructed),((index%4)*224,(index//4)*256+30))
                        draw.text(((index%4)*224+3,(index//4)*256+3),record['id'],fill='black')
                    rows.append(row)
            if index==0:
                inputs={k:v.cuda() for k,v in inputs.items()};inputs['proprio']=torch.zeros((1,20),device='cuda');inputs['domain_id']=torch.tensor([3],device='cuda')
                for _ in range(3):
                    torch.cuda.synchronize();t=time.perf_counter()
                    with torch.inference_mode():actions=model.generate_actions(**inputs,steps=10)
                    torch.cuda.synchronize();timings.append({'version':version,'seconds':time.perf_counter()-t,'action_shape':list(actions.shape)})
        sheet.save(folder/'four-orientations-three-positions.png')
    names={n:p.numel() for n,p in model.named_parameters() if 'soft_prompt' in n}
    report={'checkpoint_revision':'129e71460678b7236cee6fc9707f09d9fa0c3590','model_class':str(type(model)),
        'processor_class':str(type(processor)),'loading_info':info,'vision_encoder_entries':captured,
        'processor':processor.image_processor.to_dict(),'timings':timings,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
        'soft_prompt_parameters':names,'rows':rows,'raw_contract':'uint8 RGB HWC 512; MuJoCo vertical flip occurs once in environment.frame',
        'image_order':['primary','secondary','RGB crop for O2, masked zero padding otherwise'],'summary':{v:{'passed':sum(r.get('legibility_pass',False) for r in rows if r['observation_version']==v),'total':12} for v in versions}}
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['summary','vision_encoder_entries','soft_prompt_parameters','timings','peak_cuda_bytes']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--versions',nargs='+',default=['O0','O1']);a=p.parse_args();main(a.source,a.output,a.versions)
