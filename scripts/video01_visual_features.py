"""Frozen pretrained phone-image embeddings; offline correspondence, no actions."""
import argparse, collections, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2, numpy as np, torch
from transformers import AutoProcessor
from supervisor.adaptation.m6_joint import load_model

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-index',type=Path,required=True);p.add_argument('--project-root',type=Path,required=True);p.add_argument('--base-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    index=json.loads(a.source_index.read_text());samples=index['samples'];model=load_model(a.base_model);model.requires_grad_(False);assert not any(q.requires_grad for q in model.parameters())
    processor=AutoProcessor.from_pretrained(str(a.base_model),trust_remote_code=True,local_files_only=True)
    model_hash=sha(a.base_model/'model.safetensors');features=[];start=time.perf_counter();shapes=[]
    torch.cuda.reset_peak_memory_stats()
    for first in range(0,len(samples),16):
        batch=samples[first:first+16];images=[]
        for s in batch:
            raw=(a.project_root/s['frame_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==s['frame_sha256']
            im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if s['canonical_rotate_ccw']:im=cv2.rotate(im,cv2.ROTATE_90_COUNTERCLOCKWISE)
            images.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        pixels=processor.image_processor(images,return_tensors='pt')['pixel_values'].cuda();shapes.append(list(pixels.shape))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):f=model.vlm._encode_image(pixels).float().mean(dim=1)
        features.append(f.cpu().numpy());print('FEATURE_FRAMES',first+len(batch),len(samples),flush=True)
    x=np.concatenate(features);assert len(x)==len(samples) and np.isfinite(x).all();x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    np.savez_compressed(a.output/'frozen-image-features.npz',features=x,sample_ids=np.asarray([s['sample_id'] for s in samples]))
    groups=collections.defaultdict(list)
    for i,s in enumerate(samples):groups[(s['demo_id'],s['view'])].append(i)
    rng=np.random.default_rng(121212);stats=[];means={}
    for key,ids in groups.items():
        v=x[ids].mean(axis=0);means[key]=v/max(np.linalg.norm(v),1e-8)
    for d in index['demonstrations']:
        did=d['demo_id'];aa=groups[(did,'camera_a')];bb=groups[(did,'camera_b')];ta=np.array([samples[i]['clip_time_s'] for i in aa]);tb=np.array([samples[i]['clip_time_s'] for i in bb]);scores=x[aa]@x[bb].T;best=scores.argmax(axis=1)
        err=np.abs(ta-tb[best]);permutations=[]
        for _ in range(100):permutations.append(float(np.mean(np.abs(ta-rng.permutation(tb)[best]))))
        pooled_candidates={other['demo_id']:float(means[(did,'camera_a')]@means[(other['demo_id'],'camera_b')]) for other in index['demonstrations']}
        matched=max(pooled_candidates,key=pooled_candidates.get)
        # Pooling is permutation invariant: explicitly verify rather than claim intention.
        shuffled=x[rng.permutation(aa)].mean(axis=0);pooled_difference=float(np.max(np.abs(shuffled-x[aa].mean(axis=0))))
        stats.append({'demo_id':did,'manual_category':d['manual_category'],'primary_queries':len(aa),'secondary_candidates':len(bb),'sync_uncertainty_s':d['sync_uncertainty_s'],'mean_nearest_feature_time_error_s':float(err.mean()),'median_nearest_feature_time_error_s':float(np.median(err)),'fraction_within_075s':float(np.mean(err<=.75)),'shuffled_timestamp_mean_error_s':float(np.mean(permutations)),'reversed_timestamp_mean_error_s':float(np.mean(np.abs(ta-tb[::-1][best]))),'static_first_frame_time_error_s':float(np.mean(np.abs(ta-tb[0]))),'pooled_nearest_take_id':matched,'pooled_correct_take':matched==did,'pooled_time_shuffle_max_feature_difference':pooled_difference})
    summary={'source_index_sha256':sha(a.source_index),'model_sha256':model_hash,'base_model':str(a.base_model.resolve()),'model_hub_manifest':json.loads((a.base_model/'hub_manifest.json').read_text()) if (a.base_model/'hub_manifest.json').exists() else None,'samples':len(samples),'feature_shape':list(x.shape),'image_tensor_shapes':shapes,'feature_path':'vlm._encode_image -> mean over visual tokens -> L2 normalize','fit_on_personal_data':False,'robot_actions_generated':False,'seconds_after_model_load':time.perf_counter()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'seed':121212,'pairs':stats,'pooled_take_top1':sum(r['pooled_correct_take'] for r in stats),'limits':['Matching searches the entire paired human take, including future frames. It is an offline representation diagnostic, not a causal control result.','16 takes come from four recording pairs; shared background can explain pooled take matching.','Mean pooling removes temporal order and is not an intention encoder by itself.','No robot, simulator truth, geometric teacher action, text prompt, fitted action decoder or metric depth is used.','A 0.75-second scoring tolerance combines sparse sampling with documented synchronization uncertainty; report continuous timing error too.','Hand occlusion and repeated stationary poses make image-to-time correspondence ambiguous.']}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('FEATURE_COMPLETE',len(samples),'TOP1',summary['pooled_take_top1'],flush=True)
if __name__=='__main__':main()
