"""BC initialization from public events, then clipped on-policy PPO on sources.

PPO rewards and critic targets may use evaluator truth. The actor sees only
camera/history/goals and common feasibility masks, never privileged poses.
"""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical
from .model import TemporalActorCritic
from .env import TaskEnv, MODES


def run(root, condition='clean', no_history=False, shuffle=False, seed=None, steps=None):
    cfg=json.loads((root/'configs/pilot.json').read_text())
    seed=cfg['seed'] if seed is None else seed
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    name=f'{condition}' + ('_no_history' if no_history else '') + ('_shuffled' if shuffle else '') + f'_s{seed}'
    out=root/'runs'/name;out.mkdir(parents=True,exist_ok=True)
    model=TemporalActorCritic(**{k:cfg[k] for k in ['width','heads','layers','history']}).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    dataset=np.load(root/f'data/processed/train_{condition}.npz')
    x,y,mask=dataset['x'].copy(),dataset['y'].copy(),dataset['mask'].copy()
    if no_history: x[:]=x[:,-1:, :, :]
    if shuffle:
        # Permute action labels only among examples sharing the feasibility
        # mask. This keeps the negative control feasible, not trivially invalid.
        for key in np.unique(mask,axis=0):
            idx=np.flatnonzero(np.all(mask==key,axis=1));y[idx]=np.random.permutation(y[idx])
    tensor=lambda a:torch.as_tensor(a,device=device)
    bc_start=time.perf_counter()
    for epoch in range(cfg['bc_epochs']):
        order=np.random.permutation(len(x));losses=[]
        for start in range(0,len(x),cfg['batch_size']):
            ids=order[start:start+cfg['batch_size']]
            logits,_=model(tensor(x[ids]),tensor(mask[ids]))
            loss=torch.nn.functional.cross_entropy(logits,tensor(y[ids]).long())
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
            losses.append(loss.item())
        if epoch%10==0:print(json.dumps(dict(stage='bc',run=name,epoch=epoch,loss=float(np.mean(losses)))),flush=True)
    torch.save({'state_dict':model.state_dict(),'config':cfg,'condition':condition,'no_history':no_history,'seed':seed},out/'bc.pt')
    val=np.load(root/f'data/processed/validation_{condition}.npz')
    vx=val['x'].copy()
    if no_history:vx[:]=vx[:,-1:]
    with torch.no_grad():
        pred=model(tensor(vx),tensor(val['mask']))[0].argmax(-1).cpu().numpy()
    bc_accuracy=float(np.mean(pred==val['y']))
    manifest=json.loads((root/'data/processed/manifest.json').read_text())
    source=[e for e in manifest['episodes'] if e['split']=='train']
    env=TaskEnv(seed,history=cfg['history'],max_decisions=cfg['max_decisions'])
    episode_idx=0
    def reset():
        nonlocal episode_idx
        e=source[episode_idx%len(source)];episode_idx+=1
        mode='clean' if condition=='clean' else MODES[(episode_idx-1)%len(MODES)]
        return env.reset(initial=e['initial'],goal_size=int(np.random.choice(cfg['training_goal_sizes'])),mode=mode)
    obs,action_mask=reset();budget=cfg['ppo_steps'] if steps is None else steps
    log=(out/'training.jsonl').open('w');total=0;ppo_start=time.perf_counter()
    while total < budget:
        histories=[];masks=[];actions=[];oldlogs=[];values=[];rewards=[];dones=[]
        for _ in range(min(cfg['rollout_steps'],budget-total)):
            h=obs.copy()
            if no_history:h[:]=h[-1]
            with torch.no_grad():
                logits,value=model(tensor(h[None]),tensor(action_mask[None]));dist=Categorical(logits=logits)
                action=dist.sample();lp=dist.log_prob(action)
            histories.append(h);masks.append(action_mask.copy());actions.append(action.item());oldlogs.append(lp.item());values.append(value.item())
            (obs,action_mask),reward,done,result=env.step(action.item());rewards.append(reward);dones.append(done);total+=1
            if done:
                log.write(json.dumps(dict(step=total,**result))+'\n');log.flush();obs,action_mask=reset()
        with torch.no_grad():
            bootstrap=model(tensor((np.repeat(obs[-1:],[cfg['history']],axis=0) if no_history else obs)[None]),tensor(action_mask[None]))[1].item()
        advantages=np.zeros(len(rewards),np.float32);gae=0
        for t in reversed(range(len(rewards))):
            nv=bootstrap if t==len(rewards)-1 else values[t+1]
            live=1-float(dones[t]);delta=rewards[t]+cfg['gamma']*nv*live-values[t]
            gae=delta+cfg['gamma']*cfg['gae_lambda']*live*gae;advantages[t]=gae
        returns=advantages+np.array(values,np.float32)
        advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        hx=tensor(np.array(histories));mm=tensor(np.array(masks));aa=tensor(np.array(actions));ll=tensor(np.array(oldlogs,dtype=np.float32));ad=tensor(advantages);rt=tensor(returns)
        for _ in range(cfg['ppo_epochs']):
            order=np.random.permutation(len(rewards))
            for start in range(0,len(order),cfg['batch_size']):
                ids=order[start:start+cfg['batch_size']]
                logits,value=model(hx[ids],mm[ids]);dist=Categorical(logits=logits)
                ratio=(dist.log_prob(aa[ids])-ll[ids]).exp()
                policy=-torch.minimum(ratio*ad[ids],ratio.clamp(1-cfg['clip'],1+cfg['clip'])*ad[ids]).mean()
                loss=policy+.5*(value-rt[ids]).square().mean()-cfg['entropy']*dist.entropy().mean()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
        if total%2048==0:print(json.dumps(dict(stage='ppo',run=name,steps=total,loss=loss.item())),flush=True)
    log.close()
    checkpoint=dict(state_dict=model.cpu().state_dict(),config=cfg,condition=condition,no_history=no_history,seed=seed,
                    source_ids=[e['id'] for e in source],dataset_sha256=manifest['sha256'],ppo_steps=total,
                    shuffled_labels=shuffle,actor_input='RGB-derived features',task_holdout=[3,4])
    torch.save(checkpoint,out/'policy.pt')
    summary=dict(run=name,device=device,parameters=sum(p.numel() for p in model.parameters()),bc_validation_accuracy=bc_accuracy,
                 bc_seconds=ppo_start-bc_start,ppo_seconds=time.perf_counter()-ppo_start,steps=total,config=cfg)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--condition',choices=['clean','disturbed'],default='clean')
    p.add_argument('--no-history',action='store_true');p.add_argument('--shuffle',action='store_true');p.add_argument('--seed',type=int);p.add_argument('--steps',type=int)
    a=p.parse_args();run(a.root.resolve(),a.condition,a.no_history,a.shuffle,a.seed,a.steps)
