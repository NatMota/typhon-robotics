"""Prompt-only adaptation through the pinned model's native flow-matching loss."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
from .observations import views
from .validation import validate_model

PROMPT='transformer.soft_prompt_hub.weight'


def load_model():
    torch.set_num_threads(4)
    model,info=AutoModel.from_pretrained('.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True,
        torch_dtype=torch.float32,output_loading_info=True)
    if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise RuntimeError(info)
    return model.cuda().eval()


def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def train_native(dataset,out,split,updates,seed,lr,batch,initial=None,ids=None):
    """Fixed-budget prompt-only native forward; no visual feature cache."""
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(dataset.read_text())
    selected=[e for e in manifest['episodes'] if e['split'] in split.split(',') and (not ids or e['id'] in ids)]
    if not selected:raise ValueError('Empty training set')
    episodes=[];ledger=[]
    for e in selected:
        path=dataset.parent/e['path']
        with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==e['sha256']
        with np.load(path) as archive:arrays={k:archive[k] for k in archive.files}
        valid=np.flatnonzero(arrays['frame_action_indices']+30<=len(arrays['targets']))
        if len(valid)<48:raise ValueError('Insufficient complete windows for matched48-window budget')
        chosen=valid[np.linspace(0,len(valid)-1,48,dtype=int)]
        episodes.append((e,arrays,chosen))
        ledger.append({'id':e['id'],'split':e['split'],'source_sha256':e['sha256'],'frame_indices':chosen.tolist(),'windows':48,'terminal_padding':False})
    model=load_model();params=dict(model.named_parameters())
    if initial:
        state=torch.load(initial,map_location='cpu',weights_only=False)
        assert state['base_revision']=='129e71460678b7236cee6fc9707f09d9fa0c3590' and state['observation_version']=='O2'
        with torch.no_grad():params[PROMPT].copy_(state['prompt'].cuda())
    for n,p in params.items():p.requires_grad_(n==PROMPT)
    before={n:tensor_hash(p) for n,p in params.items()};original=params[PROMPT].detach().clone()
    processor=AutoProcessor.from_pretrained('.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);processor.language_max_length=128
    validation_before=validate_model(model,processor,dataset,out,'update0')
    def make_batch(samples):
        images=[];instructions=[];targets=[];proprios=[]
        for eidx,widx in samples:
            e,a,chosen=episodes[eidx];i=int(chosen[widx]);j=int(a['frame_action_indices'][i])
            images.append(views({k:a[k][i] for k in ['primary','secondary']},'O2'));instructions.append(e['instruction'])
            targets.append(a['targets'][j:j+30]);proprios.append(a['proprio'][i])
        result=processor(images=images,language_instruction=instructions)
        result.update(action=torch.from_numpy(np.stack(targets)),proprio=torch.from_numpy(np.stack(proprios)),domain_id=torch.full((len(samples),),3,dtype=torch.long))
        return {k:v.cuda() for k,v in result.items()}
    probe=make_batch([(i%len(episodes),w) for i,w in enumerate([0,16,31,47])]);captured={}
    handle=model.transformer.register_forward_hook(lambda m,a,o:captured.update(prediction=o.detach()))
    def measure():
        torch.manual_seed(73123)
        with torch.no_grad():losses=model(**probe)
        pred=captured['prediction'];target=probe['action'];r={k:float(v) for k,v in losses.items()}
        for name,start in [('active_arm',0),('padded_arm',10)]:
            r[name+'_xyz_mse']=float((pred[...,start:start+3]-target[...,start:start+3]).square().mean())
            r[name+'_rot6d_mse']=float((pred[...,start+3:start+9]-target[...,start+3:start+9]).square().mean())
            r[name+'_gripper_bce']=float(torch.nn.functional.binary_cross_entropy_with_logits(pred[...,start+9],target[...,start+9]))
        return r
    probe_before=measure();rng=np.random.default_rng(seed);torch.manual_seed(seed)
    optimizer=torch.optim.AdamW([params[PROMPT]],lr=lr,weight_decay=0.);loss_log=[];sample_log=[];start_all=time.perf_counter()
    for step in range(updates):
        samples=[(int(rng.integers(len(episodes))),int(rng.integers(48))) for _ in range(batch)]
        sample_log.append(samples);inputs=make_batch(samples)
        optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
        losses=model(**inputs);loss=sum(losses.values());assert torch.isfinite(loss)
        loss.backward();grad=params[PROMPT].grad;assert grad is not None and torch.isfinite(grad).all()
        norm=float(grad.norm());torch.nn.utils.clip_grad_norm_([params[PROMPT]],1.);optimizer.step();torch.cuda.synchronize()
        row={'step':step+1,'seconds':time.perf_counter()-start,'gradient_norm':norm,**{k:float(v) for k,v in losses.items()}}
        loss_log.append(row)
        if (step+1)%100==0:print(row,flush=True)
        if (step+1)%250==0 and step+1<updates:
            torch.save({'prompt':params[PROMPT].detach().cpu(),'optimizer':optimizer.state_dict(),'step':step+1,
                'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
                'selection_use':False},out/f'recovery-update{step+1}.pt')
    train_seconds=time.perf_counter()-start_all;probe_after=measure();handle.remove()
    validation_after=validate_model(model,processor,dataset,out,f'update{updates}')
    changed=[n for n,p in params.items() if tensor_hash(p)!=before[n]]
    delta=(params[PROMPT]-original).reshape(30,-1).abs().max(dim=1).values.tolist()
    assert changed==[PROMPT] and delta[3]>0 and all(v==0 for i,v in enumerate(delta) if i!=3)
    state={'prompt':params[PROMPT].detach().cpu(),'base_revision':'129e71460678b7236cee6fc9707f09d9fa0c3590','observation_version':'O2','seed':seed,'updates':updates,'split':split}
    torch.save(state,out/'prompt.pt');prediction_input={k:v[:1] for k,v in probe.items() if k!='action'}
    for p in model.parameters():p.requires_grad_(True)  # same flags as unchanged inference worker
    torch.manual_seed(7381)
    with torch.inference_mode():expected=model.generate_actions(**prediction_input,steps=10).cpu()
    peak=torch.cuda.max_memory_allocated();del optimizer,params,model;torch.cuda.empty_cache()
    loaded=load_model();restored=torch.load(out/'prompt.pt',map_location='cpu',weights_only=False)
    with torch.no_grad():dict(loaded.named_parameters())[PROMPT].copy_(restored['prompt'].cuda())
    assert tensor_hash(dict(loaded.named_parameters())[PROMPT])==tensor_hash(state['prompt'])
    torch.manual_seed(7381)
    with torch.inference_mode():actual=loaded.generate_actions(**prediction_input,steps=10).cpu()
    error=float((expected-actual).abs().max())
    report={'passed':error==0,'updates':updates,'batch_size':batch,'sampled_action_windows':updates*batch,'actions_per_window':30,'seed':seed,'lr':lr,'weight_decay':0.,
        'execution_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
        'training_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'initial_checkpoint':str(initial) if initial else 'unchanged_XVLA','initial_checkpoint_sha256':hashlib.sha256(initial.read_bytes()).hexdigest() if initial else None,
        'selected_episodes':ledger,'fixed_probe_before':probe_before,'fixed_probe_after':probe_after,'changed_parameter_names':changed,'per_domain_max_delta':delta,
        'unique_window_counts':{e['id']:len({w for batch_samples in sample_log for i,w in batch_samples if i==j}) for j,e in enumerate(selected)},
        'shared_validation_files':['validation-update0.json',f'validation-update{updates}.json'],
        'optimizer_exposed_parameters':983040,'effective_changed_domain_parameters':32768,'all_other_parameters_hash_unchanged':True,
        'native_forward':True,'vision_feature_cache_used':False,'terminal_padding_used':False,'checkpoint_selection':'fixed final update; no performance selection',
        'mean_update_s':float(np.mean([r['seconds'] for r in loss_log])),'training_wall_s':train_seconds,'peak_cuda_bytes':peak,
        'checkpoint_reload_action_max_error':error,'checkpoint_sha256':hashlib.sha256((out/'prompt.pt').read_bytes()).hexdigest(),
        'observation_version':'O2','loss_implementation':'unchanged pinned XVLA.forward flow-noise and action_space.compute_loss'}
    (out/'training.json').write_text(json.dumps(report,indent=2)+'\n');(out/'losses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in loss_log));print(json.dumps(report,indent=2))
    (out/'sampling.json').write_text(json.dumps({'episode_order':[e['id'] for e in selected],'sample_pairs_episode_window':sample_log})+'\n')
    assert report['passed'],'Reload check failed; diagnostic saved'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',required=True);p.add_argument('--updates',type=int,required=True);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--batch',type=int,default=4);p.add_argument('--initial',type=Path);p.add_argument('--ids',nargs='+')
    a=p.parse_args();train_native(a.dataset,a.output,a.split,a.updates,a.seed,a.lr,a.batch,a.initial,a.ids)
