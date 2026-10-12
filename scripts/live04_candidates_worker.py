"""V4 candidate worker using unchanged production equations and retained noise."""
import argparse, json, pickle, sys, time, traceback, types
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v3_v4_same_input_20261011 import ExplicitNoiseTorch

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args()
    wire=sys.stdout.buffer;sys.stdout=sys.stderr
    def reply(x):pickle.dump(x,wire,protocol=4);wire.flush()
    try:
        import torch
        from transformers import AutoProcessor
        from supervisor.adaptation.m7_training import load_checkpoint,generate_requested,HISTORY_AUDIT
        from supervisor.adaptation.m7_eval_worker import prepare_inputs
        t=time.perf_counter();model,projection,restored=load_checkpoint(a.checkpoint,ROOT/'.cache-xvla/checkpoint')
        config=restored['config'];assert config['condition']=='crosshair' and config['history_mode']=='sparse' and restored['domain_id']==29
        metadata=json.loads(json.dumps({k:restored[k] for k in ['schema','base_revision','step','config']},default=str));del restored
        model.eval();projection.eval()
        processor=AutoProcessor.from_pretrained(ROOT/'.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);processor.language_max_length=256
        reply(dict(ready=True,load_time_s=time.perf_counter()-t,checkpoint=a.checkpoint,model_state_strict_loaded=True,**metadata))
        while True:
            payload=pickle.load(sys.stdin.buffer)
            if payload['op']=='close':break
            if payload['op']=='reset':
                torch.manual_seed(payload['seed']);torch.cuda.manual_seed_all(payload['seed']);HISTORY_AUDIT.clear();reply({'reset':True});continue
            try:
                start=time.perf_counter()
                seeds=payload.pop('candidate_seeds',None)
                if seeds is not None:assert len(seeds) in (1,4) and all(type(s) is int for s in seeds)
                inputs,held,valid,annotation=prepare_inputs(processor,payload,config,'cuda')
                timed_all=[];noise_all=[];latencies=[]
                # Production single-example path is retained. No batching-dependent
                # numerical change or changed video/history representation.
                for seed in seeds or [None]:
                    if seed is not None:torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                    noise=torch.randn(1,len(payload['request']['anchor_times_s']),20,device=inputs['proprio'].device)
                    proxy=ExplicitNoiseTorch(torch,noise)
                    local=types.FunctionType(generate_requested.__code__,dict(generate_requested.__globals__,torch=proxy),generate_requested.__name__,generate_requested.__defaults__,generate_requested.__closure__)
                    torch.cuda.synchronize();begin=time.perf_counter()
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        timed=local(model,projection,inputs,payload['request'],held,valid,steps=10)
                    torch.cuda.synchronize();assert proxy.calls==1
                    latencies.append(time.perf_counter()-begin);timed_all.append(timed);noise_all.append(noise.cpu().numpy())
                reply({'timed':timed_all[0],'timed_candidates':timed_all,'initial_noise':np.concatenate(noise_all),
                    'inference_s':sum(latencies),'candidate_inference_s':latencies,'preprocess_and_inference_s':time.perf_counter()-start,
                    'image_mask':inputs['image_mask'].cpu().tolist(),'input_shapes':{k:list(v.shape) for k,v in inputs.items()},
                    'history_times_s':inputs.get('history_times_s',torch.tensor([])).cpu().tolist(),'history_camera_ids':inputs.get('history_camera_ids',torch.tensor([])).cpu().tolist(),
                    'annotation':annotation,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'candidate_seeds':seeds})
            except Exception:reply({'error':traceback.format_exc()})
    except Exception:reply({'error':traceback.format_exc()});raise
if __name__=='__main__':main()
