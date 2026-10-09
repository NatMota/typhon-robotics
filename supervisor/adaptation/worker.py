"""Trusted local inference worker; action logic remains in frozen M2 adapter."""
import argparse
import pickle
import sys
import time
import traceback
wire=sys.stdout.buffer;sys.stdout=sys.stderr


def reply(x):
    pickle.dump(x,wire,protocol=4);wire.flush()


try:
    import torch
    import numpy as np
    from transformers import AutoProcessor
    from .train import load_model, PROMPT, tensor_hash
    from .observations import views, object_crop
    p=argparse.ArgumentParser();p.add_argument('--version',required=True);p.add_argument('--checkpoint');a=p.parse_args()
    start=time.perf_counter();model=load_model();numerical_mode='fp32'
    assert all(p.requires_grad for p in model.parameters()) and not model.training
    if a.checkpoint:
        checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        assert checkpoint['base_revision']=='129e71460678b7236cee6fc9707f09d9fa0c3590'
        assert checkpoint['observation_version']==a.version
        if checkpoint.get('schema')=='m5-v1':
            from .m5 import restore
            model=restore(model,checkpoint);numerical_mode=checkpoint['numerical_mode']
        else:
            with torch.no_grad():dict(model.named_parameters())[PROMPT].copy_(checkpoint['prompt'].cuda())
            assert tensor_hash(dict(model.named_parameters())[PROMPT])==tensor_hash(checkpoint['prompt'])
    processor=AutoProcessor.from_pretrained('.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True)
    processor.language_max_length=128
    reply({'ready':True,'load_time_s':time.perf_counter()-start,'checkpoint':a.checkpoint,'version':a.version,
        'inference_configuration':{'parameter_requires_grad':'allTrue','module_mode':'eval','execution':'torch.inference_mode','dtype':numerical_mode}})
    while True:
        request=pickle.load(sys.stdin.buffer)
        if request['op']=='close':break
        if request['op']=='reset':
            torch.manual_seed(request['seed']);torch.cuda.manual_seed_all(request['seed']);reply({'reset':True});continue
        try:
            start=time.perf_counter()
            images=views(dict(zip(['primary','secondary'],request['images'])),a.version)
            inputs=processor(images=images,language_instruction=request['instruction'])
            inputs['proprio']=torch.as_tensor(request['proprio'],dtype=torch.float32).unsqueeze(0)
            inputs['domain_id']=torch.tensor([3],dtype=torch.long)
            inputs={k:v.cuda() for k,v in inputs.items()}
            torch.cuda.synchronize();t=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=numerical_mode=='bf16'):action=model.generate_actions(**inputs,steps=10)
            torch.cuda.synchronize();seconds=time.perf_counter()-t
            result=action.squeeze(0).float().cpu().numpy()
            if result.shape!=(30,20) or not np.isfinite(result).all():raise ValueError('Invalid model output')
            reply({'actions':result,'inference_s':seconds,'preprocess_and_inference_s':time.perf_counter()-start,
                'input_shapes':{k:list(v.shape) for k,v in inputs.items()},'image_mask':inputs['image_mask'].cpu().tolist(),
                'crop_metadata':object_crop(request['images'][0])[1] if a.version=='O2' else None,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        except Exception:reply({'error':traceback.format_exc()})
except Exception:
    reply({'error':traceback.format_exc()});sys.exit(1)
