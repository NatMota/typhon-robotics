"""Focused regression checks for the new controller/observation/reward interface."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .bank import load_bank
from .core import ResetSampler,objective
from .environment import PandaJointEnv
from .network import ActorCritic

def verify(bank,output):
    meta,states=load_bank(bank)
    train={(r["task"],r["seed"]) for r in meta["records"] if r["split"]=="train"}
    dev={(r["task"],r["seed"]) for r in meta["records"] if r["split"]=="development"}
    assert train.isdisjoint(dev)
    sampler=ResetSampler(3)
    assert np.allclose(sampler.probabilities(),[1/3]*3)
    for outcome in [0,1]*10:sampler.update(1,outcome)
    for _ in range(20):sampler.update(2,True)
    assert sampler.probabilities()[1]>sampler.probabilities()[0]
    assert np.isclose(sampler.probabilities()[0],sampler.probabilities()[2])
    assert objective(False,np.zeros(8),np.zeros(7),np.zeros(7))[0]==0
    assert objective(True,np.zeros(8),np.zeros(7),np.zeros(7))[0]==1
    rows=[]
    for task in ("feeding","loading"):
        e=PandaJointEnv(task,horizon=8)
        idx=next(r["index"] for r in meta["records"] if r["task"]==task and r["split"]=="train")
        try:
            obs=e.reset(states[idx])
            assert obs.shape==(888,) and np.isfinite(obs).all()
            assert np.allclose(e.scene.env.sim.data.qpos,states[idx]["qpos"],atol=1e-12)
            assert e.scene.env.sim.model.body_name2id("m1_fixture")>=0
            network=ActorCritic()
            x=torch.tensor(obs[None]);_,_,lp,_,v=network.sample(x)
            (lp+v).sum().backward()
            assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
            deltas=[]
            for sign in (-1,1):
                e.reset(states[idx]);robot=e.scene.env.robots[0]
                before=e.scene.env.sim.data.qpos[robot._ref_joint_pos_indexes[0]]
                action=np.zeros(8);action[0]=sign*.5
                for _ in range(8):
                    _,_,done,info=e.step(action)
                    if done:break
                after=e.scene.env.sim.data.qpos[robot._ref_joint_pos_indexes[0]]
                deltas.append(float(after-before))
                assert done
                try:e.step(action)
                except RuntimeError:pass
                else:raise AssertionError("Terminal transition allowed another reward")
            assert deltas[0]<0 and deltas[1]>0
            rows.append({"task":task,"actual_joint0_delta_rad_for_negative_positive":deltas,
                         "fixture_present":True,"observation_shape":[888],"action_shape":[8]})
        finally:e.close()
    out={"passed":True,"split_disjoint":True,"adaptive_sampler_test":True,"binary_reward_test":True,
         "reset_state_test":True,"physical_action_sign_test":True,"terminal_guard_test":True,
         "finite_gradient_test":True,"rows":rows}
    Path(output).write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--bank",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();verify(a.bank,a.output)
