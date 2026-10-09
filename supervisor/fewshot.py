"""Separate, explicitly labelled adaptation after zero-shot results are frozen.

Requires quarantined support data; never runs automatically with fabricated
phone demonstrations. Run identically for correctly matched and mismatched
support files, then evaluate the two checkpoints on unchanged test conditions.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from .model import TemporalActorCritic


def adapt(root,base,support,kind,clips):
    frozen=root/'runs/evaluation/frozen.json'
    if not frozen.exists():raise ValueError('Freeze zero-shot evaluation before target adaptation')
    base_sha=hashlib.sha256(base.read_bytes()).hexdigest()
    accepted={m['checkpoint_sha256'] for m in json.loads(frozen.read_text())['methods'].values() if m is not None}
    if base_sha not in accepted:raise ValueError('Base checkpoint was not part of the frozen zero-shot evaluation')
    declaration=json.loads((support.parent/'personal-manifest.json').read_text())
    allowed={r['id'] for r in declaration if r['split']=='target_support'}
    forbidden={r['id'] for r in declaration if r['split'] in ['test','train','validation']}
    d=np.load(support);ids=sorted(set(d['episode']))
    if not set(ids)<=allowed or set(ids)&forbidden:raise ValueError('Adaptation file contains non-support or held-out episodes')
    if len(ids)<clips:raise ValueError(f'Need {clips} distinct support clips')
    ids=ids[:clips];keep=np.isin(d['episode'],ids)
    ck=torch.load(base,map_location='cpu',weights_only=False);cfg=ck['config'];torch.manual_seed(cfg['seed']);torch.set_num_threads(2)
    model=TemporalActorCritic(**{k:cfg[k] for k in ['width','heads','layers','history']});model.load_state_dict(ck['state_dict'])
    x=torch.from_numpy(d['x'][keep]);m=torch.from_numpy(d['mask'][keep]);y=torch.from_numpy(d['y'][keep]).long()
    if ck['no_history']:x[:]=x[:,-1:]
    if not torch.all(m[torch.arange(len(y)),y]):raise ValueError('Infeasible support labels')
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    # Fixed adaptation budget, no tuning on target test outcomes.
    for _ in range(200):
        logits,_=model(x,m);loss=torch.nn.functional.cross_entropy(logits,y)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
    ck.update(state_dict=model.state_dict(),adaptation_kind=kind,support_ids=ids,support_sha256=hashlib.sha256(support.read_bytes()).hexdigest(),
              base_checkpoint_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),adaptation_steps=200,
              zero_shot_frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest())
    out=root/f'runs/fewshot-{kind}-{clips}';out.mkdir(parents=True,exist_ok=True);torch.save(ck,out/'policy.pt')
    (out/'manifest.json').write_text(json.dumps({k:v for k,v in ck.items() if k!='state_dict'},indent=2))
    print(out/'policy.pt')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--base',type=Path,required=True)
    p.add_argument('--support',type=Path,required=True);p.add_argument('--kind',choices=['matched','mismatched'],required=True)
    p.add_argument('--clips',choices=[1,4],type=int,required=True)
    a=p.parse_args();adapt(a.root.resolve(),a.base,a.support,a.kind,a.clips)
