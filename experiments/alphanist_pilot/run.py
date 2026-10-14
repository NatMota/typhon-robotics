"""Reproducible small PPO pilot; fresh outputs only and held-out seeds never fit."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from .bank import load_bank,build_bank
from .core import ResetSampler
from .environment import PandaJointEnv
from .network import ActorCritic

def dump(p,x): Path(p).write_text(json.dumps(x,indent=2))
def train(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed)
    meta,states=load_bank(args.bank)
    allowed=[r["index"] for r in meta["records"] if r["split"]=="train"]
    assert not set(allowed)&{r["index"] for r in meta["records"] if r["split"]=="development"}
    sampler=ResetSampler(len(allowed),args.seed,args.sampling)
    envs={t:PandaJointEnv(t,args.horizon) for t in ("feeding","loading")}
    model=ActorCritic().to(args.device);opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    config=vars(args).copy();config.update(bank_sha256=hashlib.sha256((Path(args.bank)/"states.npz").read_bytes()).hexdigest(),
       actor_observation="simulator surface geometry + proprioception + previous action; no reset phase or task ID",
       action="relative joint position 7 + gripper 1",task_reward="binary strict completion",
       control_cost=.0001,rollout_steps=args.rollout,ppo_epochs=5,minibatches=4,
       sim="MuJoCo CPU physics; GPU neural network; not massive Isaac training")
    dump(out/"config.json",config)
    starts=[]
    for idx in allowed:
        r=meta["records"][idx];starts.append(envs[r["task"]].reset(states[idx]))
    # Train-reset observations only. Frozen normalization makes PPO ratios consistent.
    model.update_normalization(torch.tensor(np.stack(starts),device=args.device))
    selected=sampler.sample();idx=allowed[selected];env=envs[meta["records"][idx]["task"]]
    obs=env.reset(states[idx]);started=time.perf_counter();total=0;episode_return=0.;episodes=0;successes=0
    try:
        while total<args.steps:
            batch=[];ep_rows=[]
            for _ in range(min(args.rollout,args.steps-total)):
                x=torch.tensor(obs[None],device=args.device)
                with torch.no_grad(): action,raw,logp,entropy,value=model.sample(x)
                nxt,reward,done,info=env.step(action[0].cpu().numpy())
                with torch.no_grad(): nxtv=model.distribution(torch.tensor(nxt[None],device=args.device))[1].item()
                terminated=info["success"] or info["limit_violation"]
                batch.append((obs,raw[0].cpu().numpy(),logp.item(),value.item(),reward,done,0. if terminated else nxtv))
                total+=1;episode_return+=reward;obs=nxt
                if done:
                    sampler.update(selected,info["success"]);episodes+=1;successes+=info["success"]
                    row=dict(info,episode=episodes,reset_index=idx,return_=episode_return,train_step=total)
                    ep_rows.append(row)
                    selected=sampler.sample();idx=allowed[selected];env=envs[meta["records"][idx]["task"]]
                    obs=env.reset(states[idx]);episode_return=0.
            advantage=np.zeros(len(batch),np.float32);tail=0.
            for t in reversed(range(len(batch))):
                _,_,_,v,r,done,nv=batch[t]
                delta=r+.99*nv-v
                tail=delta+.99*.95*(not done)*tail;advantage[t]=tail
            tensors=[torch.tensor(np.stack([b[k] for b in batch]),device=args.device,dtype=torch.float32) for k in range(4)]
            bx,br,bl,bv=tensors
            returns=torch.tensor(advantage,device=args.device)+bv
            adv=torch.tensor(advantage,device=args.device);adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
            losses=[]
            for epoch in range(5):
                for inds in torch.randperm(len(batch),device=args.device).chunk(4):
                    if not len(inds):continue
                    _,_,lp,ent,val=model.sample(bx[inds],br[inds])
                    ratio=(lp-bl[inds]).exp()
                    pg=-torch.minimum(ratio*adv[inds],ratio.clamp(.8,1.2)*adv[inds]).mean()
                    vf=.5*(val-returns[inds]).square().mean()
                    loss=pg+vf-.001*ent.mean()
                    if not torch.isfinite(loss):raise RuntimeError("Nonfinite PPO loss")
                    opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step()
                    losses.append(float(loss.detach()))
            with (out/"episodes.jsonl").open("a") as f:
                for row in ep_rows:f.write(json.dumps(row)+"\n")
            row={"step":total,"episodes":episodes,"successes":int(successes),"loss":float(np.mean(losses)),
                 "steps_per_second":total/(time.perf_counter()-started),"elapsed_s":time.perf_counter()-started}
            with (out/"training.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
            checkpoint={"model":model.state_dict(),"optimizer":opt.state_dict(),"step":total,"config":config,
                        "sampler":sampler.state(),"torch_rng":torch.get_rng_state()}
            torch.save(checkpoint,out/"latest.pt.tmp");(out/"latest.pt.tmp").replace(out/"latest.pt")
        dump(out/"complete.json",dict(row,finished_unix=time.time(),training_completed=True,
             conclusion="Software pilot only; inspect independent evaluation before any learning claim."))
    finally:
        for e in envs.values():e.close()

def evaluate(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    meta,states=load_bank(args.bank)
    model=ActorCritic().to(args.device)
    if args.checkpoint:
        ckpt=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
        model.load_state_dict(ckpt["model"])
    else:
        # Fixed normalization from train reset observations, like the training route.
        pass
    model.eval();envs={t:PandaJointEnv(t,args.horizon) for t in ("feeding","loading")}
    rows=[]
    try:
        if not args.checkpoint:
            norm=[envs[r["task"]].reset(states[r["index"]]) for r in meta["records"] if r["split"]=="train"]
            model.update_normalization(torch.tensor(np.stack(norm),device=args.device))
        candidates=[r for r in meta["records"] if r["split"]=="development"]
        for r in candidates[:args.episodes]:
            e=envs[r["task"]];obs=e.reset(states[r["index"]]);ret=0.;actions=[];frames=[];truths=[]
            for step in range(args.horizon):
                with torch.no_grad():action,_,_,_,_=model.sample(torch.tensor(obs[None],device=args.device),deterministic=True)
                a=action[0].cpu().numpy();actions.append(a.tolist())
                if args.video and step%4==0:frames.append(e.scene.frame())
                obs,reward,done,info=e.step(a);ret+=reward;truths.append(e.scene.truth())
                if done:break
            row=dict(info,reset_index=r["index"],seed=r["seed"],source_control_step=r["source_control_step"],return_=ret)
            rows.append(row);dump(out/"summary.json",{"episodes":rows,"successes":sum(x["success"] for x in rows),"count":len(rows),
              "evaluation_scope":"Held-out expert-derived reset states, NOT fresh-start whole-task success",
              "checkpoint":args.checkpoint,"checkpoint_sha256":hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None})
            dump(out/("episode-"+str(r["index"])+".json"),{"summary":row,"actions":actions,"truths":truths})
            if frames:
                import imageio.v2 as imageio
                imageio.mimwrite(out/("episode-"+str(r["index"])+".mp4"),frames,fps=25/4)
            print(json.dumps(row),flush=True)
    finally:
        for e in envs.values():e.close()

def main():
    p=argparse.ArgumentParser();p.add_argument("op",choices=["bank","train","evaluate"])
    p.add_argument("--output",required=True);p.add_argument("--bank")
    p.add_argument("--steps",type=int,default=4096);p.add_argument("--rollout",type=int,default=256)
    p.add_argument("--horizon",type=int,default=100);p.add_argument("--seed",type=int,default=314)
    p.add_argument("--device",default="cuda");p.add_argument("--sampling",choices=["adaptive","uniform"],default="adaptive")
    p.add_argument("--checkpoint");p.add_argument("--episodes",type=int,default=32);p.add_argument("--video",action="store_true")
    a=p.parse_args()
    if a.op=="bank":build_bank(a.output)
    elif a.op=="train":train(a)
    else:evaluate(a)
if __name__=="__main__":main()
