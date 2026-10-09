"""100-update native-forward plumbing diagnostic; no cached vision features."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import time
import numpy as np
import torch
from transformers import AutoProcessor
from supervisor.adaptation.train import load_model, PROMPT, tensor_hash
from supervisor.adaptation.observations import views


def main(data,out):
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((data/'dataset.json').read_text());episode=next(e for e in manifest['episodes'] if e['id']=='personal_demo_0001' and e['split']=='P_train')
    with np.load(data/episode['path']) as archive:arrays={k:archive[k] for k in archive.files}
    valid=np.flatnonzero(arrays['frame_action_indices']+30<=len(arrays['targets']))
    processor=AutoProcessor.from_pretrained('.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);processor.language_max_length=128
    model=load_model();params=dict(model.named_parameters())
    for n,p in params.items():p.requires_grad_(n==PROMPT)
    before={n:tensor_hash(p) for n,p in params.items()};original_prompt=params[PROMPT].detach().clone()
    def batch(indices):
        images=[views({k:arrays[k][i] for k in ['primary','secondary']},'O2') for i in indices]
        inputs=processor(images=images,language_instruction=[episode['instruction']]*len(indices))
        inputs['proprio']=torch.from_numpy(arrays['proprio'][indices]);inputs['domain_id']=torch.full((len(indices),),3,dtype=torch.long)
        inputs['action']=torch.from_numpy(np.stack([arrays['targets'][int(arrays['frame_action_indices'][i]):int(arrays['frame_action_indices'][i])+30] for i in indices]))
        return {k:v.cuda() for k,v in inputs.items()}
    probe=batch(valid[np.linspace(0,len(valid)-1,4,dtype=int)])
    captured={}
    handle=model.transformer.register_forward_hook(lambda m,a,o:captured.update(prediction=o.detach()))
    def measure():
        torch.manual_seed(73123)
        with torch.no_grad():losses=model(**probe)
        pred=captured['prediction'];target=probe['action'];result={k:float(v) for k,v in losses.items()}
        for name,start in [('active_arm',0),('padded_arm',10)]:
            result[name+'_xyz_mse']=float((pred[...,start:start+3]-target[...,start:start+3]).square().mean())
            result[name+'_rot6d_mse']=float((pred[...,start+3:start+9]-target[...,start+3:start+9]).square().mean())
            result[name+'_gripper_bce']=float(torch.nn.functional.binary_cross_entropy_with_logits(pred[...,start+9],target[...,start+9]))
        return result
    probe_before=measure();rng=np.random.default_rng(17);torch.manual_seed(17)
    optimizer=torch.optim.AdamW([params[PROMPT]],lr=.001,weight_decay=0.)
    rows=[];start_all=time.perf_counter()
    for step in range(100):
        inputs=batch(rng.choice(valid,4));optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();start=time.perf_counter();losses=model(**inputs);loss=sum(losses.values())
        assert torch.isfinite(loss);loss.backward();grad=params[PROMPT].grad
        assert grad is not None and torch.isfinite(grad).all()
        norm=float(grad.norm());torch.nn.utils.clip_grad_norm_([params[PROMPT]],1.);optimizer.step();torch.cuda.synchronize()
        rows.append({'step':step+1,'seconds':time.perf_counter()-start,'gradient_norm':norm,**{k:float(v) for k,v in losses.items()}})
        if (step+1)%20==0:print(step+1,rows[-1],flush=True)
    training_seconds=time.perf_counter()-start_all
    probe_after=measure();handle.remove()
    changed=[n for n,p in params.items() if tensor_hash(p)!=before[n]]
    domain_delta=(params[PROMPT]-original_prompt).reshape(30,-1).abs().max(dim=1).values.tolist()
    assert changed==[PROMPT] and domain_delta[3]>0 and all(v==0 for i,v in enumerate(domain_delta) if i!=3)
    state={'prompt':params[PROMPT].detach().cpu(),'base_revision':'129e71460678b7236cee6fc9707f09d9fa0c3590','observation_version':'O2','seed':17,'updates':100,'split':'single_personal_demo_0001_sanity'}
    torch.save(state,out/'prompt.pt')
    prediction_input={k:v[:1] for k,v in probe.items() if k!='action'}
    # Match the existing rollout worker. Inference-mode kernels still depend on
    # parameter flags: the isolated diagnostic measured a 9.4e-6 difference.
    # No optimization follows this change; the prompt-only audit above is final.
    for p in model.parameters():p.requires_grad_(True)
    torch.manual_seed(7381)
    with torch.inference_mode():expected=model.generate_actions(**prediction_input,steps=10).cpu()
    torch.manual_seed(7381)
    with torch.inference_mode():repeat=model.generate_actions(**prediction_input,steps=10).cpu()
    repeat_error=float((expected-repeat).abs().max())
    peak=torch.cuda.max_memory_allocated()
    del optimizer,params,model;torch.cuda.empty_cache()
    loaded=load_model();saved=torch.load(out/'prompt.pt',weights_only=False)
    with torch.no_grad():dict(loaded.named_parameters())[PROMPT].copy_(saved['prompt'].cuda())
    torch.manual_seed(7381)
    with torch.inference_mode():actual=loaded.generate_actions(**prediction_input,steps=10).cpu()
    reload_error=float((expected-actual).abs().max())
    reload_parameters_equal=all(tensor_hash(p)==(tensor_hash(saved['prompt']) if n==PROMPT else before[n]) for n,p in loaded.named_parameters())
    report={'passed':reload_error==0 and repeat_error==0 and reload_parameters_equal,'training_episode':episode['id'],'source_sha256':episode['sha256'],'updates':100,'batch_size':4,'sampled_windows':400,
        'valid_complete_windows':len(valid),'terminal_padding_used':False,'native_forward':True,'vision_feature_cache_used':False,
        'fixed_probe_before':probe_before,'fixed_probe_after':probe_after,'changed_parameter_names':changed,'domain_delta':domain_delta,
        'optimizer_exposed_parameters':983040,'effective_changed_parameters':32768,'frozen_parameters_hash_unchanged':True,
        'checkpoint_reload_action_max_error':reload_error,'mean_update_s':float(np.mean([r['seconds'] for r in rows])),
        'same_model_repeat_max_error':repeat_error,'all_reloaded_parameters_equal':reload_parameters_equal,
        'reload_inference_parameter_flags':'all requires_grad=True, matching unchanged rollout worker; inference_mode disables autograd',
        'training_wall_s':training_seconds,'peak_cuda_bytes':peak,'lr':.001,'weight_decay':0.,
        'checkpoint_sha256':tensor_hash(saved['prompt'])}
    (out/'sanity.json').write_text(json.dumps(report,indent=2)+'\n');(out/'updates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print(json.dumps(report,indent=2))
    np.savez_compressed(out/'reload_predictions.npz',expected=expected.numpy(),repeat=repeat.numpy(),reloaded=actual.numpy())
    assert report['passed'], 'Reload/repeat audit failed; measurements preserved in sanity.json'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.data,a.output)
