"""Full-weight joint adaptation from the separate pinned X-VLA foundation checkpoint."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
import json,hashlib
from pathlib import Path
import torch
from transformers import AutoModel,AutoProcessor
BASE=Path('.cache-xvla/checkpoint')
REV='c1c4a64a7e03ac5b95c468bf1578f3d03651b53b'
OUT=Path('results/personal-m6-joint-20261009')

def dump(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')

def load_model(base=BASE):
 """Restore Florence's serialized shared encoder embedding alias explicitly."""
 torch.set_num_threads(4)
 model,info=AutoModel.from_pretrained(str(base),trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32,output_loading_info=True)
 allowed={'vlm.language_model.model.encoder.embed_tokens.weight'}
 assert set(info.get('missing_keys',[]))<=allowed,info
 assert not any(info.get(k) for k in ['unexpected_keys','mismatched_keys','error_msgs']),info
 lm=model.vlm.language_model.model
 lm.encoder.embed_tokens=lm.shared
 assert lm.encoder.embed_tokens.weight is lm.shared.weight
 from safetensors import safe_open
 with safe_open(str(base/'model.safetensors'),framework='pt') as f:
  assert torch.equal(lm.shared.weight.detach(),f.get_tensor('vlm.language_model.model.shared.weight'))
 return model.cuda().eval()

from .m6_contract import DOMAIN,reorder,self_check
from .m5 import dataset,sha,inference
from .train import tensor_hash
import numpy as np,time,argparse,shutil,math

def group(n):
    if n.startswith('vlm.'):return 'vlm'
    if 'soft_prompt_hub' in n:return 'prompt'
    if 'action_encoder' in n or 'action_decoder' in n:return 'heads'
    return 'core'

def custom_init(model):
    """Reset only domain29 with the pinned initializer's distribution and fan size."""
    records={}
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        with torch.no_grad():
            for n,p in model.named_parameters():
                if group(n) not in ['prompt','heads']:continue
                assert p.shape[0]==30
                before=tensor_hash(p[:DOMAIN])
                if group(n)=='prompt':torch.nn.init.normal_(p[DOMAIN],std=.02)
                elif '.bias.' in n:p[DOMAIN].zero_()
                else:
                    bound=math.sqrt(6./(p.shape[0]+p.shape[1]))
                    torch.nn.init.uniform_(p[DOMAIN],-bound,bound)
                assert tensor_hash(p[:DOMAIN])==before
                records[n]={'initial_active_sha256':tensor_hash(p[DOMAIN]),'inactive_sha256':before,'initializer':'normal std0.02' if group(n)=='prompt' else 'zero bias / Xavier uniform with original table fan dimensions'}
    return records

def make_data():
    e,a,chosen,old=dataset()
    def batch(indices):
        b=old(indices)
        for k in ['action','proprio']:b[k]=reorder(b[k])
        b['domain_id'].fill_(DOMAIN)
        return b
    return e,a,chosen,batch

def optimizer_for(model):
    groups={g:[] for g in ['prompt','heads','vlm','core']}
    for n,p in model.named_parameters():
        g=group(n);groups[g].append(p)
        if g in ['prompt','heads']:
            def mask(grad):
                grad=grad.clone();grad[:DOMAIN]=0;grad[DOMAIN+1:]=0;return grad
            p.register_hook(mask)
    rates={'prompt':1e-5,'heads':1e-4,'vlm':1e-5,'core':1e-4}
    opt=torch.optim.AdamW([{'params':ps,'lr':rates[g],'peak_lr':rates[g],'weight_decay':0. if g in ['prompt','heads'] else .01,'name':g} for g,ps in groups.items()],betas=(.9,.95),eps=1e-8,foreach=False)
    return groups,opt

def schedule(model,opt,step,profile=False):
    active=profile or step>1000
    factor=1. if profile or step<=1000 else min(1.,(step-1000)/1000.)
    for n,p in model.named_parameters():p.requires_grad_(active or group(n) in ['prompt','heads'])
    for pg in opt.param_groups:pg['lr']=pg['peak_lr']*factor if active or pg['name'] in ['prompt','heads'] else 0.
    return active,factor

def save_checkpoint(model,opt,folder,label,step,probe,rng):
    target=folder/(label+'.pt');tmp=target.with_suffix('.partial')
    state={'schema':'m6-full-v1','base_revision':REV,'step':step,'domain_id':DOMAIN,'seed':17,'observation_version':'O2','rotation_contract':'generic_interleaved_v1','precision':'bf16','model':model.state_dict(),'optimizer':opt.state_dict(),'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
    torch.save(state,tmp);tmp.replace(target)
    expected=inference(model,probe,'bf16')
    torch.save({'inputs':{k:v.cpu() for k,v in probe.items()},'expected':expected},folder/(label+'-probe.pt'))
    dump(folder/(label+'-checkpoint.json'),{'step':step,'sha256':sha(target),'bytes':target.stat().st_size,'full_model_and_optimizer':True})
    return target

def train(profile=False):
    folder=OUT/('qualification' if profile else 'training');folder.mkdir(parents=True,exist_ok=False)
    assert self_check()['passed']
    e,a,chosen,batch=make_data();model=load_model();assert model.action_space.XYZ_SCALE==500.
    records=custom_init(model);dump(folder/'initialization.json',records)
    params=dict(model.named_parameters());before={n:tensor_hash(p) for n,p in params.items()}
    if not profile:
        torch.save({n:p.detach().cpu() for n,p in params.items() if group(n) in ['prompt','heads']},folder/'custom-init-tables.pt')
    groups,opt=optimizer_for(model);probe=batch([0]);probe.pop('action')
    # Same tensors for training and native inference; no cached image embeddings.
    assert torch.equal(reorder(reorder(torch.from_numpy(a['targets'])),True),torch.from_numpy(a['targets']))
    dump(folder/'config.json',{'base':'2toINF/X-VLA-Pt','revision':REV,'domain':DOMAIN,'seed':17,'source':e,'windows':chosen.tolist(),'updates':6 if profile else 3000,'effective_batch':4,'microbatch':1,'accumulation':4,'warmup':1000,'joint_ramp':1000,'joint_hold':1000,'groups':{g:sum(p.numel() for p in ps) for g,ps in groups.items()},'optimizer':'AdamW betas0.9,0.95 eps1e-8 weight_decay0.01; domain-table decay active row only','peak_lr':{g['name']:g['peak_lr'] for g in opt.param_groups},'training_end_utc':'2026-10-09T16:23:50Z','claim':'bounded full-weight diagnostic; not benchmark reproduction','image_shapes':{k:list(v.shape) for k,v in probe.items()},'rotation_check':self_check()})
    rng=np.random.default_rng(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17);model.train();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();rows=[]
    steps=6 if profile else 3000
    for step in range(1,steps+1):
        if time.time()>=1791563030:raise RuntimeError('M6 training cutoff reached; partial evidence retained')
        active,factor=schedule(model,opt,step,profile)
        samples=[0]*4 if profile else [int(rng.integers(48)) for _ in range(4)]
        opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter();losses={}
        for index in samples:
            b=batch([index])
            if profile:assert b['image_mask'].all()
            with torch.autocast('cuda',dtype=torch.bfloat16):ls=model(**b);loss=sum(ls.values())/4
            assert torch.isfinite(loss);loss.backward()
            for k,v in ls.items():losses[k]=losses.get(k,0.)+float(v.detach())/4
        grads={g:float(torch.sqrt(sum((p.grad.detach().square().sum() for p in ps if p.grad is not None),torch.tensor(0.,device='cuda')))) for g,ps in groups.items()}
        assert all(np.isfinite(x) for x in grads.values()) and grads['prompt']>0 and grads['heads']>0,grads
        if active:assert grads['vlm']>0 and grads['core']>0,grads
        else:assert grads['vlm']==0 and grads['core']==0
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));opt.step()
        with torch.no_grad():
            for pg in opt.param_groups:
                if pg['name'] in ['prompt','heads']:
                    for p in pg['params']:p[DOMAIN].mul_(1.-pg['lr']*.01)
        torch.cuda.synchronize();row={'step':step,'joint':active,'ramp':factor,'seconds':time.perf_counter()-t,'samples':samples,'grads':grads,'clip_norm':norm,**losses};rows.append(row)
        with (folder/'losses.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if step<=3 or step%100==0 or step==1001:print(row,flush=True)
        if step==1000 and not profile:
            assert all(not p.requires_grad for n,p in params.items() if group(n) in ['vlm','core'])
            dump(folder/'warmup_gate.json',{'passed':all(tensor_hash(p)==before[n] for n,p in params.items() if group(n) in ['vlm','core']),'shared_optimizer_states':sum(p in opt.state for g in ['vlm','core'] for p in groups[g]),'seconds':time.perf_counter()-start})
            assert json.loads((folder/'warmup_gate.json').read_text())['passed']
            torch.save({n:p.detach().cpu() for n,p in params.items() if group(n) in ['prompt','heads']},folder/'W1000-tables.pt')
    ckpt=save_checkpoint(model,opt,folder,'final',steps,probe,rng)
    changed={g:[] for g in groups}
    for n,p in params.items():
        if tensor_hash(p)!=before[n]:changed[group(n)].append(n)
    assert all(changed.values())
    for n,r in records.items():assert tensor_hash(params[n][:DOMAIN])==r['inactive_sha256'],n
    report={'passed':True,'updates':steps,'wall_seconds':time.perf_counter()-start,'mean_step_seconds':float(np.mean([r['seconds'] for r in rows])),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'changed':changed,'inactive_domain_rows_unchanged':True,'full_joint':True,'lora':False,'checkpoint':str(ckpt),'checkpoint_sha256':sha(ckpt)}
    dump(folder/'training.json',report);print(json.dumps({k:v for k,v in report.items() if k!='changed'}),flush=True)

def verify(path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['schema']=='m6-full-v1' and state['base_revision']==REV
    model=load_model();model.load_state_dict(state['model'],strict=True)
    _,opt=optimizer_for(model);opt.load_state_dict(state['optimizer'])
    assert all(s['exp_avg'].dtype==torch.float32 and s['exp_avg_sq'].dtype==torch.float32 for s in opt.state.values())
    probe=torch.load(path.with_name(path.stem+'-probe.pt'),weights_only=False)
    actual=inference(model,{k:v.cuda() for k,v in probe['inputs'].items()},'bf16');error=float((actual-probe['expected']).abs().max())
    dump(path.with_suffix('.reload.json'),{'passed':error==0.,'fresh_process':True,'full_optimizer_restored':True,'optimizer_parameters':len(opt.state),'action_max_error':error,'sha256':sha(path)})
    assert error==0.,error
    print('M6 FULL CHECKPOINT RELOAD PASSED',path,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('op',choices=['profile','train','verify']);parser.add_argument('--checkpoint',type=Path);args=parser.parse_args()
    if args.op=='verify':verify(args.checkpoint)
    else:train(profile=args.op=='profile')
