"""Bounded M5 source adaptation; isolated checkpoints, matched sampling, no actor truth."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import sys, json, hashlib, time, argparse, contextlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.m5-deps'))
import numpy as np
import torch
from transformers import AutoProcessor
from .train import load_model, tensor_hash
from .observations import views
REV='129e71460678b7236cee6fc9707f09d9fa0c3590'
OUT=Path('results/personal-m5-20261009')
DATA=Path('frozen_results/personal-m3-20261009/data-canonical/dataset.json')
LORA={'r':8,'lora_alpha':16,'target_modules':'all-linear','lora_dropout':0.,'bias':'none'}

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def amp(mode):return torch.autocast('cuda',dtype=torch.bfloat16,enabled=mode=='bf16')

def group(name):
    if 'lora_' in name:return 'vlm_lora' if '.vlm.' in name else 'policy_lora'
    if name.endswith('transformer.soft_prompt_hub.weight'):return 'prompt'
    if '.transformer.action_encoder.' in name or '.transformer.action_decoder.' in name:return 'heads'
    return 'frozen'

def wrap(model):
    from peft import LoraConfig,get_peft_model
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(1705)
        model=get_peft_model(model,LoraConfig(**LORA))
    return model

def inference(model,inputs,mode):
    flags=[p.requires_grad for p in model.parameters()];training=model.training
    for p in model.parameters():p.requires_grad_(True)
    model.eval()
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(7381);torch.cuda.manual_seed_all(7381)
        with torch.inference_mode(),amp(mode):answer=model.generate_actions(**inputs,steps=10).float().cpu()
    for p,flag in zip(model.parameters(),flags):p.requires_grad_(flag)
    model.train(training)
    return answer

def restore(model,state):
    assert state['schema']=='m5-v1' and state['base_revision']==REV and state['observation_version']=='O2'
    assert state['lora_config']==LORA
    model=wrap(model);params=dict(model.named_parameters())
    expected={n for n in params if group(n)!='frozen'}
    assert set(state['parameters'])==expected
    with torch.no_grad():
        for n,t in state['parameters'].items():
            assert params[n].shape==t.shape,n
            params[n].copy_(t.to(params[n].device))
    for p in params.values():p.requires_grad_(True)
    return model.eval()

def dataset():
    manifest=json.loads(DATA.read_text());e=next(e for e in manifest['episodes'] if e['id']=='personal_demo_0001')
    assert e['split']=='P_train'
    path=DATA.parent/e['path'];assert sha(path)==e['sha256']
    with np.load(path) as f:a={k:f[k] for k in f.files}
    valid=np.flatnonzero(a['frame_action_indices']+30<=len(a['targets']))
    chosen=valid[np.linspace(0,len(valid)-1,48,dtype=int)]
    proc=AutoProcessor.from_pretrained('.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);proc.language_max_length=128
    def batch(indices):
        frames=[int(chosen[w]) for w in indices];starts=[int(a['frame_action_indices'][i]) for i in frames]
        b=proc(images=[views({k:a[k][i] for k in ['primary','secondary']},'O2') for i in frames],language_instruction=[e['instruction']]*len(frames))
        b.update(action=torch.from_numpy(np.stack([a['targets'][j:j+30] for j in starts])),proprio=torch.from_numpy(a['proprio'][frames]),domain_id=torch.full((len(frames),),3,dtype=torch.long))
        return {k:v.cuda() for k,v in b.items()}
    return e,a,chosen,batch

def protocol():
    e,a,chosen,_=dataset()
    phases={'preflight':(0,24),'approach':(24,124),'descend':(124,158),'close':(158,176),'lift':(176,276),'transfer':(276,328),'place':(328,344),'release':(344,364),'retreat':(364,380),'return_and_stability':(380,446)}
    starts=a['frame_action_indices'][chosen]
    coverage={n:{'range_action_index':[lo,hi],'window_starts':int(((starts>=lo)&(starts<hi)).sum()),'overlapping_windows':int(((starts<hi)&(starts+30>lo)).sum())} for n,(lo,hi) in phases.items()}
    assert all(coverage[n]['overlapping_windows']>0 for n in ['approach','close','lift','transfer','release'])
    dump(OUT/'source_protocol.json',{'episode':e,'npz_sha256':e['sha256'],'frame_indices':chosen.tolist(),'action_starts':starts.tolist(),'complete_windows':48,'terminal_padding':False,'phase_coverage':coverage,'phase_evidence':'M15 executor preflight24; expert phase lengths100,34,100,52,16,16; close18/release20; source truth steps minus40 reset. Action gripper transitions158/344 corroborate boundaries.','seed':17,'effective_batch':4,'updates':3000,'selection':'fixed final checkpoints; one source fitting diagnostic, no broader generalization claim','mode':'native forward in train mode; generate_actions eval; no image feature cache','data_origin':'personal observed initial state; generated geometric-expert robot targets, not human robot-action labels'})
    import copy
    core=[]
    for i,seed in enumerate([7317,7318,7319]):
        core.append({'scenario_id':f'm5-source-{i+1:02d}','group':'seen_source','task':'feeding','seed':seed,'config':copy.deepcopy(e['config']),'attempt_limit':2,'source_demo_id':e['id']})
    m=json.loads(DATA.read_text())
    for id in ['personal_demo_0006','personal_demo_0014','personal_demo_0016']:
        v=next(x for x in m['episodes'] if x['id']==id);assert v['split']=='P_validation'
        core.append({'scenario_id':'m5-validation-'+id[-4:],'group':'development_validation','task':'feeding','seed':7417,'config':copy.deepcopy(v['config']),'attempt_limit':2,'source_demo_id':id})
    prior=json.loads(Path('frozen_results/personal-m4-20261009/protocol/scenarios.json').read_text())['primary']
    for id in ['recovery-already_completed-01','recovery-already_completed-03']:
        s=copy.deepcopy(next(s for s in prior if s['scenario_id']==id));s['group']='already_completed';core.append(s)
    original=json.loads(Path('frozen_results/personal-m3-20261009/protocol/scenarios.json').read_text())['primary']
    loading=[copy.deepcopy(s) for s in original if s['task']=='loading'][:2]
    assert len(core)==8 and len(loading)==2
    dump(OUT/'evaluation_protocol.json',{'core':core,'conditional_loading':loading,'conditional_gate':'at least one A source strict success and time before rollout hard stop','methods':['F','C3000','A3000'],'source_seeds':[7317,7318,7319],'H':30,'denoise_steps':10,'domain':3,'rate_hz':20,'completion_guard':False,'wrist_projection':False,'expert_rescue':False,'validation_status':'development held-out configs previously inspected; not untouched confirmatory test'})

def train(name,mode,micro,updates):
    folder=OUT/'training'/name;folder.mkdir(parents=True,exist_ok=False)
    e,a,chosen,make_batch=dataset();probe=make_batch([0]);probe.pop('action')
    model=load_model();raw=inference(model,probe,mode) if name=='profile' else None
    model=wrap(model);params=dict(model.named_parameters())
    targets=[n for n,m in model.named_modules() if hasattr(m,'lora_A')]
    if raw is not None:
        wrapped=inference(model,probe,mode);error=float((raw-wrapped).abs().max());assert error<=1e-6,('initial_adapter_not_noop',error)
        dump(folder/'initial_noop.json',{'max_action_error':error,'passed':True})
    row_original={n:p.detach().cpu().clone() for n,p in params.items() if group(n) in ['prompt','heads']}
    frozen_before={n:tensor_hash(p) for n,p in params.items() if group(n)=='frozen'}
    before={n:tensor_hash(p) for n,p in params.items() if group(n)!='frozen'}
    groups={k:[] for k in ['prompt','heads','policy_lora','vlm_lora']}
    for n,p in params.items():
        g=group(n);p.requires_grad_(g!='frozen' and (name!='C3000' or g=='prompt'))
        if g in groups:groups[g].append(p)
        if g in ['prompt','heads'] and p.requires_grad:
            def mask(grad):
                out=grad.clone();out[:3]=0;out[4:]=0;return out
            p.register_hook(mask)
    assert all(groups.values())
    optimizer=torch.optim.AdamW([{'params':ps,'lr':{'prompt':1e-5,'heads':1e-4,'policy_lora':1e-4,'vlm_lora':1e-5}[g],'weight_decay':.01 if 'lora' in g else 0.,'label':g} for g,ps in groups.items()],betas=(.9,.95))
    dump(folder/'parameter_manifest.json',{'lora_config':LORA,'lora_targets':targets,'counts':{g:sum(p.numel() for p in ps) for g,ps in groups.items()},'frozen_count':sum(p.numel() for n,p in params.items() if group(n)=='frozen'),'checkpoint_storage':'all LoRA tensors plus full prompt/action-head domain tables; unchanged base referenced by pinned revision','head_strategy':'custom DomainAwareLinear embeddings kept unwrapped, saved explicitly; only domain3 gradients; wd0','native_libero':'unavailable in existing environments; no rebuild under bounded budget'})
    rng=np.random.default_rng(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
    model.train();torch.cuda.reset_peak_memory_stats();samplelog=[];losses=[];start=time.perf_counter();forward_calls={};handles=[]
    for n,m in model.named_modules():
        if hasattr(m,'lora_A'):
            def hook(mod,args,out,n=n):forward_calls[n]=forward_calls.get(n,0)+1
            handles.append(m.register_forward_hook(hook))
    def save(label,step):
        state={'schema':'m5-v1','base_revision':REV,'observation_version':'O2','lora_config':LORA,'numerical_mode':mode,'microbatch':micro,'effective_batch':4,'name':name,'step':step,'seed':17,'parameters':{n:p.detach().cpu().clone() for n,p in params.items() if group(n)!='frozen'}}
        torch.save(state,folder/(label+'.pt'))
        calls_before=dict(forward_calls)
        expected=inference(model,probe,mode)
        traversed=[n for n,v in forward_calls.items() if v>calls_before.get(n,0)]
        assert any('.vlm.' in n for n in traversed) and any('.transformer.' in n for n in traversed),traversed
        dump(folder/(label+'-inference-adapter-path.json'),{'traversed_lora_modules':traversed,'native_generate_actions':True})
        torch.save({'inputs':{k:v.cpu() for k,v in probe.items()},'expected':expected},folder/(label+'-reload-probe.pt'))
        return state
    for step in range(updates):
        if time.time()>=1791563030:raise RuntimeError('M5 training hard stop reached; partial artifacts preserved')
        active=name=='profile' or (name=='A3000' and step>=1000)
        for g in ['policy_lora','vlm_lora']:
            for p in groups[g]:p.requires_grad_(active)
        factor=1. if name=='profile' else min(1.,max(0.,(step-999)/1000))
        for pg in optimizer.param_groups:
            if 'lora' in pg['label']:pg['lr']=(1e-5 if pg['label']=='vlm_lora' else 1e-4)*factor
        samples=[int(rng.integers(48)) for _ in range(4)]
        if name=='profile':samples=[0]*4  # fully visible source frame for worst-case three-view memory
        samplelog.append(samples)
        optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter();sums={}
        for j in range(0,4,micro):
            inputs=make_batch(samples[j:j+micro])
            if name=='profile':assert inputs['image_mask'].all(),'profile requires all three views'
            # Training retains the declared RGB crop detector's masked/occluded cases.
            with amp(mode):ls=model(**inputs);loss=sum(ls.values())*(micro/4)
            assert torch.isfinite(loss),ls
            loss.backward()
            for k,v in ls.items():sums[k]=sums.get(k,0.)+float(v.detach())*(micro/4)
        grads={g:float(torch.sqrt(sum((p.grad.detach().float().square().sum() for p in ps if p.grad is not None),torch.tensor(0.,device='cuda')))) for g,ps in groups.items()}
        assert all(np.isfinite(x) for x in grads.values()) and grads['prompt']>0,grads
        if active:assert grads['policy_lora']>0 and grads['vlm_lora']>0,grads
        norm=float(torch.nn.utils.clip_grad_norm_([p for p in params.values() if p.requires_grad],1.))
        optimizer.step();torch.cuda.synchronize()
        row={'step':step+1,'seconds':time.perf_counter()-t,'gradient_norm_before_clip':norm,'gradient_groups':grads,'lora_active':active,'lora_ramp':factor,**sums};losses.append(row)
        with (folder/'losses.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if (step+1)%100==0 or name=='profile':print(name,row,flush=True)
        if name=='A3000' and step+1==1000:save('W1000',1000)
        if (step+1)%500==0:save('recovery',step+1)
    final=save(name,updates);elapsed=time.perf_counter()-start
    generation_calls=dict(forward_calls)
    for h in handles:h.remove()
    changed=[n for n,p in params.items() if group(n)!='frozen' and tensor_hash(p)!=before[n]]
    unchanged=all(tensor_hash(params[n])==v for n,v in frozen_before.items());assert unchanged
    inactive={}
    for n,original in row_original.items():
        delta=(params[n].detach().cpu()-original).reshape(30,-1).abs().max(dim=1).values.tolist();inactive[n]=delta
        assert all(v==0 for i,v in enumerate(delta) if i!=3),(n,delta)
    assert changed and all(group(n)=='prompt' for n in changed) if name=='C3000' else changed
    projected=float(np.mean([r['seconds'] for r in losses[-min(len(losses),5):]]))*2000
    report={'name':name,'updates':updates,'mode':mode,'microbatch':micro,'accumulation':4//micro,'effective_batch':4,'wall_s':elapsed,'mean_update_s':float(np.mean([r['seconds'] for r in losses])),'projected_stage2_s':projected,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'frozen_parameters_unchanged':unchanged,'changed_parameters':changed,'per_domain_max_delta':inactive,'lora_forward_calls':generation_calls,'checkpoint_sha256':sha(folder/(name+'.pt')),'passed':True,'source_npz_sha256':e['sha256']}
    dump(folder/'training.json',report);dump(folder/'sampling.json',samplelog)
    print(json.dumps({k:v for k,v in report.items() if k not in ['changed_parameters','per_domain_max_delta','lora_forward_calls']},indent=2),flush=True)
    if name=='profile':assert projected<=2400,('Stage2 projection exceeds40min',projected)

def verify(path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=restore(load_model(),state);probe=torch.load(path.with_name(path.stem+'-reload-probe.pt'),weights_only=False)
    actual=inference(model,{k:v.cuda() for k,v in probe['inputs'].items()},state['numerical_mode']);error=float((actual-probe['expected']).abs().max())
    dump(path.with_suffix('.reload.json'),{'fresh_process':True,'max_action_error':error,'passed':error==0,'checkpoint_sha256':sha(path)})
    assert error==0,error
    print('FRESH RELOAD VERIFIED',path,error,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['protocol','train','verify']);p.add_argument('--name');p.add_argument('--mode',choices=['fp32','bf16'],default='fp32');p.add_argument('--micro',type=int,choices=[1,4],default=4);p.add_argument('--updates',type=int,default=3000);p.add_argument('--checkpoint',type=Path);a=p.parse_args()
    if a.operation=='protocol':protocol()
    elif a.operation=='verify':verify(a.checkpoint)
    else:train(a.name,a.mode,a.micro,a.updates)
