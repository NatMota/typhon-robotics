"""Diagnose fresh-model numerical reproducibility without further training."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import torch
from transformers import AutoProcessor
from supervisor.adaptation.train import load_model,PROMPT,tensor_hash
from supervisor.adaptation.observations import views


def main(data,checkpoint,out):
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((data/'dataset.json').read_text())
    e=next(e for e in manifest['episodes'] if e['id']=='personal_demo_0001')
    with np.load(data/e['path']) as z:arrays={k:z[k] for k in z.files}
    processor=AutoProcessor.from_pretrained('.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True)
    processor.language_max_length=128
    inputs=processor(images=views({k:arrays[k][0] for k in ['primary','secondary']},'O2'),language_instruction=e['instruction'])
    inputs.update(proprio=torch.from_numpy(arrays['proprio'][:1]),domain_id=torch.tensor([3]))
    inputs={k:v.cuda() for k,v in inputs.items()}
    state=torch.load(checkpoint,weights_only=False,map_location='cpu')
    outputs={};buffers=[]
    for run in range(2):
        model=load_model()
        with torch.no_grad():dict(model.named_parameters())[PROMPT].copy_(state['prompt'].cuda())
        assert tensor_hash(dict(model.named_parameters())[PROMPT])==tensor_hash(state['prompt'])
        buffers.append({n:tensor_hash(b) for n,b in model.named_buffers()})
        for mode in ['default','prompt_only_grad','all_frozen']:
            for n,p in model.named_parameters():p.requires_grad_(mode=='default' or (mode=='prompt_only_grad' and n==PROMPT))
            torch.manual_seed(7381)
            with torch.inference_mode():outputs[f'{run}-{mode}']=model.generate_actions(**inputs,steps=10).cpu().numpy()
        del model;torch.cuda.empty_cache()
    ref=outputs['0-default'];result={'prompt_reload_hash_equal':True,'fresh_buffers_equal':buffers[0]==buffers[1],
        'max_difference_from_first':{k:float(abs(v-ref).max()) for k,v in outputs.items()},
        'scope':'same checkpoint, one fixed input and noise seed; no optimization'}
    (out/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'predictions.npz',**outputs);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.data,a.checkpoint,a.output)
