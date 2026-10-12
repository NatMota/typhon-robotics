"""Frozen image features on the same dense phone frames, with an explicit causal crop."""
import argparse, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np,torch
from transformers import AutoProcessor
from supervisor.adaptation.m6_joint import load_model

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    index=json.loads((a.data/'source-index.json').read_text());rows=index['samples'];config=json.loads((a.data/'protocol.json').read_text())
    model=load_model(a.base_model);model.requires_grad_(False);model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    processor=AutoProcessor.from_pretrained(str(a.base_model),trust_remote_code=True,local_files_only=True)
    feature={};elapsed={};shapes={};start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for method in config['feature_methods']:
        fs=[];time0=time.monotonic();shapes[method]=[]
        for first in range(0,len(rows),config['feature_batch']):
            images=[]
            for r in rows[first:first+config['feature_batch']]:
                p=a.data/r['frame_path'];raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['frame_sha256']
                im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                if method=='current_observed_object_square_crop'and r['crop_xyxy']is not None:
                    x0,y0,x1,y1=r['crop_xyxy'];im=im[y0:y1,x0:x1]
                images.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
            px=processor.image_processor(images,return_tensors='pt')['pixel_values'].cuda()
            if not shapes[method]:shapes[method]=list(px.shape)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                x=model.vlm._encode_image(px).float().mean(1)
            x=torch.nn.functional.normalize(x,dim=1);fs.append(x.cpu().numpy())
        feature[method]=np.concatenate(fs);elapsed[method]=time.monotonic()-time0
        assert np.isfinite(feature[method]).all()
        print('FEATURE_COMPLETE',method,len(rows),elapsed[method],flush=True)
    np.savez_compressed(a.output/'features.npz',**feature,sample_ids=np.array([r['sample_id']for r in rows]),crop_valid=np.array([r['crop_xyxy']is not None for r in rows]))
    summary={'schema':'video02-features-v1','base_model':str(a.base_model.resolve()),'base_model_sha256':sha(a.base_model/'model.safetensors'),'input_index_sha256':sha(a.data/'source-index.json'),'script_sha256':sha(__file__),'frame_count':len(rows),'crop_valid_count':sum(r['crop_xyxy']is not None for r in rows),'crop_fallback':'full frame when current object is not valid; invalid-crop comparisons must be masked','sequence_embedding':'per-image visual-token mean, L2; temporal summarization is separate','feature_shape':list(feature['full_rgb'].shape),'pixel_shapes':shapes,'seconds_by_method':elapsed,'seconds_after_load':time.monotonic()-start,'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),'training_updates':0,'geometric_expert_labels':False}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--base-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
