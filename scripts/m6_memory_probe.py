"""Disposable six-step full joint-update feasibility probe; never a training initializer."""
import os
os.environ['HF_HUB_OFFLINE']='1'
import sys,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from supervisor.adaptation.m6_joint import load_model
from supervisor.adaptation.m5 import dataset
out=Path('results/personal-m6-joint-20261009/profile');out.mkdir(parents=True,exist_ok=False)
model=load_model();e,a,chosen,make_batch=dataset()
assert model.action_space.XYZ_SCALE==500.,model.action_space.XYZ_SCALE
params=dict(model.named_parameters())
def group(n):
 if n.startswith('vlm.'):return 'vlm'
 if 'soft_prompt_hub' in n:return 'prompt'
 if 'action_encoder' in n or 'action_decoder' in n:return 'heads'
 return 'core'
groups={g:[p for n,p in params.items() if group(n)==g] for g in ['vlm','prompt','heads','core']}
opt=torch.optim.AdamW([{'params':ps,'lr':1e-5 if g in ['vlm','prompt'] else 1e-4,'weight_decay':.01,'name':g} for g,ps in groups.items()],betas=(.9,.95),foreach=False)
model.train();torch.cuda.reset_peak_memory_stats();rows=[]
for step in range(6):
 opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter();values={}
 for micro in range(4):
  batch=make_batch([0]);assert batch['image_mask'].all()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   ls=model(**batch);loss=sum(ls.values())/4
  assert torch.isfinite(loss);loss.backward()
  for k,v in ls.items():values[k]=values.get(k,0.)+float(v.detach())/4
 grads={g:sum(float(p.grad.detach().float().square().sum()) for p in ps if p.grad is not None)**.5 for g,ps in groups.items()}
 assert all(v>0 for v in grads.values()),grads
 torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();torch.cuda.synchronize()
 row={'step':step+1,'seconds':time.perf_counter()-start,'grads':grads,**values};rows.append(row);print(row,flush=True)
report={'passed':True,'base':'2toINF/X-VLA-Pt','revision':'c1c4a64a7e03ac5b95c468bf1578f3d03651b53b','trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'group_parameters':{g:sum(p.numel() for p in ps) for g,ps in groups.items()},'microbatch':1,'accumulation':4,'effective_batch':4,'precision':'FP32 weights and AdamW states, BF16 autocast','full_joint_updates':True,'lora':False,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'rows':rows,'initialization_use':False}
(out/'profile.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
