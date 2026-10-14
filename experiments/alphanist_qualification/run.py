"""Frozen ALPHA02 qualification: no training, new outputs only."""
import argparse, collections, csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from experiments.alphanist_pilot.bank import load_bank
from experiments.alphanist_pilot.environment import PandaJointEnv
from experiments.alphanist_pilot.network import ActorCritic
from supervisor.feeding_loading.evaluation import score_episode

def dump(p, x): Path(p).write_text(json.dumps(x, indent=2))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def physical_endpoint(score):
    return all(v for k,v in score["criteria"].items() if k != "stable")
def stage(t, score):
    if not t["grasp_contact"] and all(v for k,v in score["criteria"].items() if k not in ("stable","released")): return "unheld_goal_like"
    if t["bilateral_grasp_contact"]:
        return "held_near" if score["position_error_m"] <= .03 and score["orientation_error_rad"] <= np.deg2rad(15) else "held_away"
    return "ungrasped"

def audit(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    old=Path(args.source);meta,states=load_bank(old/"bank-v1")
    envs={t:PandaJointEnv(t,375) for t in ("feeding","loading")}
    train={mode:rows(old/f"train-{mode}-32k-v1/episodes.jsonl") for mode in ("adaptive","uniform")}
    ledger=[]
    for r in meta["records"]:
        e=envs[r["task"]];e.reset(states[r["index"]]);truth=e.scene.truth()
        score=score_episode(truth,truth,[],e.scene.config,[])
        assert not score["success"] and not e.history and not e.terminated
        parent=next(p for p in meta["episodes"] if p["task"]==r["task"] and p["seed"]==r["seed"])
        starts=[z["source_control_step"] for z in meta["records"] if z["task"]==r["task"] and z["seed"]==r["seed"]]
        # Original score reports elapsed steps relative to the settled full start.
        final_source_step=min(starts)+parent["score"]["control_steps"]
        row=dict(r,initial_truth=truth,initial_score=score,stage=stage(truth,score),
            initial_physical_endpoint_without_stability=physical_endpoint(score),
            source_verified_end_step=final_source_step,
            remaining_to_verified_expert_end_s=(final_source_step-r["source_control_step"])/25 if parent["expert_success"] else None,
            remaining_time_caveat="Time until saved final verified expert score, not first success or a lower bound for this controller.")
        for mode,episodes in train.items():
            subset=[x for x in episodes if x["reset_index"]==r["index"]]
            row[mode+"_visits"]=len(subset);row[mode+"_successes"]=sum(x["success"] for x in subset)
            row[mode+"_control_cost_total"]=float(sum(float(x["success"])-x["return_"] for x in subset))
        ledger.append(row)
    dump(out/"reset-ledger.json",ledger)
    selected=[];slots=[]
    for task in ("feeding","loading"):
        for seed in (1701,1702):
            candidates=[r for r in ledger if r["task"]==task and r["seed"]==seed and r["split"]=="development"]
            first=min(candidates,key=lambda r:(r["source_control_step"],r["index"]))
            held=[r for r in candidates if r["bilateral_contact"] and r["initial_truth"]["bilateral_grasp_contact"] and not r["initial_physical_endpoint_without_stability"]]
            selected.append(dict(reset_index=first["index"],role="full_start",task=task,seed=seed))
            if held:
                last=max(held,key=lambda r:(r["source_control_step"],r["index"]))
                selected.append(dict(reset_index=last["index"],role="latest_held",task=task,seed=seed))
            else:slots.append(dict(task=task,seed=seed,unavailable="No eligible held reset"))
    dump(out/"panel.json",dict(selected=selected,unavailable=slots,selection="Timestamp-only latest bilateral held reset and original full start; independent of policy outcomes."))

    # Same physical state, different selected goal geometry. Goal enters actor explicitly as cloud group 1.
    obs={}
    for task,e in envs.items():obs[task]=e.reset(states[16])
    goal_delta=float(np.max(np.abs(obs["feeding"][288:576]-obs["loading"][288:576])))
    assert goal_delta>0
    e=envs["feeding"];original_horizon=e.horizon
    e.horizon=100;o100=e.reset(states[16]);e.horizon=375;o375=e.reset(states[16])
    assert np.array_equal(o100,o375)
    old_ep=json.loads((old/"eval-untrained-v1/episode-16.json").read_text())
    traces=[]
    for horizon in (100,375):
        e.horizon=horizon;e.reset(states[16]);trace=[]
        for a in old_ep["actions"][:10]:
            o,_,done,_=e.step(a);trace.append(o);assert not done
        traces.append(np.stack(trace))
    assert np.array_equal(*traces)
    e.history.append(e.scene.truth());e.terminated=True;e.reset(states[23])
    assert len(e.history)==0 and not e.terminated and e.scene.steps==0
    contracts=dict(goal_cloud_max_difference=goal_delta,goal_feature="selected stage/fixture surface point cloud; no task ID needed",
        residual_meaning="Neural network residual blocks only; no expert action is added",
        time_feature_present=False,horizon_initial_observation_identical=True,first_10_action_observations_identical=True,
        reset_clears_history_and_terminal_latch=True,all_resets_score_false_without_new_stability=True,
        reward="one strict success minus 0.0001*(mean(action^2)+mean(normalized_velocity^2)+mean(normalized_torque^2))",
        truncation="4s time limit stored as done but next-state value bootstrapped unless success/limit violation; GAE recursion cut at reset.",
        bootstrap_qualification="Timeout treated as external truncation; success and safety violation treated terminal. Audit only.",
        stability="New post-release samples at distinct times, >=0.4s; no inherited success state.",
        open_command=-1,open_control="Zero seven joint deltas; gripper -1. Joint servo still applies gravity compensation/impedance.",
        source_full_starts="After original 40-step settling. Preserved original ALPHA01 full-start controls.",
        fixed_seed=314,policy_evaluation="Deterministic tanh of actor mean, original evaluation convention",
        gamma=.99,discount_at_4s=.99**99,discount_at_15s=.99**374)
    dump(out/"contract-audit.json",contracts)
    costs={}
    for mode,episodes in train.items():
        values=np.array([float(x["success"])-x["return_"] for x in episodes])
        costs[mode]=dict(episodes=len(values),sum_control_cost=float(values.sum()),median_episode_control_cost=float(np.median(values)),
            max_episode_control_cost=float(values.max()),successes=sum(x["success"] for x in episodes),
            discounted_training_return="Not reconstructable exactly: old training traces retain aggregate episodic reward, not per-step rewards.")
    dump(out/"training-reward-audit.json",costs)
    # Existing eval truth+actions support velocity; torque was not retained per step.
    eval_rewards=[]
    for folder in ["eval-untrained-v1","eval-adaptive-32k-v1","eval-uniform-32k-v1"]:
        for p in (old/folder).glob("episode-*.json"):
            x=json.loads(p.read_text());s=x["summary"];n=len(x["actions"])
            eval_rewards.append(dict(condition=folder,reset_index=s["reset_index"],steps=n,success=s["success"],
                control_cost_total=float(s["success"])-s["return_"],terminal_reward_discount=.99**(n-1) if s["success"] else 0,
                discounted_return_exact=None,reason="Per-step old torque/control penalty was not saved; ALPHA02 logs it."))
    dump(out/"prior-evaluation-reward-audit.json",eval_rewards)
    for e in envs.values():e.close()
    dump(out/"audit-complete.json",dict(reset_count=len(ledger),panel_count=len(selected),contracts_passed=True,
        counts=dict(collections.Counter(r["stage"] for r in ledger)),failed_source_resets=[r["index"] for r in ledger if not r["expert_episode_success"]]))
    print(json.dumps(json.loads((out/"audit-complete.json").read_text())),flush=True)

def evaluate(args):
    out=Path(args.output);source=Path(args.source);auditdir=Path(args.audit)
    assert json.loads((auditdir/"audit-complete.json").read_text())["contracts_passed"]
    destination=out/args.condition;destination.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(314);np.random.seed(314)
    meta,states=load_bank(source/"bank-v1");panel=json.loads((auditdir/"panel.json").read_text())["selected"]
    envs={t:PandaJointEnv(t,375) for t in ("feeding","loading")}
    model=None
    if args.condition!="OPEN":
        model=ActorCritic().to("cuda")
        if args.condition=="U0":
            norm=[envs[r["task"]].reset(states[r["index"]]) for r in meta["records"] if r["split"]=="train"]
            model.update_normalization(torch.tensor(np.stack(norm),device="cuda"))
        else:
            mode={"A32":"adaptive","U32":"uniform"}[args.condition]
            ckpt=source/f"train-{mode}-32k-v1/latest.pt"
            model.load_state_dict(torch.load(ckpt,map_location="cuda",weights_only=False)["model"])
        model.eval()
    rows_out=[];started=time.time()
    try:
        for slot in panel:
            idx=slot["reset_index"];e=envs[slot["task"]];obs=e.reset(states[idx]);initial=e.scene.truth()
            actions=[];truths=[];rewards=[];infos=[];scores=[];obs_digest=hashlib.sha256()
            for i in range(375):
                obs_digest.update(obs.tobytes())
                if model is None:a=np.r_[np.zeros(7),-1].astype(np.float32)
                else:
                    with torch.no_grad():a=model.sample(torch.tensor(obs[None],device="cuda"),deterministic=True)[0][0].cpu().numpy()
                obs,reward,done,info=e.step(a);truth=e.scene.truth()
                score=score_episode(initial,truth,list(e.history),e.scene.config,[])
                actions.append(a.tolist());truths.append(truth);rewards.append(reward);infos.append(info);scores.append(score)
                if done:break
            success=bool(infos[-1]["success"]);steps=len(actions)
            supported=0.;supported_xy=0.;prev=initial
            inherited=bool(initial["bilateral_grasp_contact"]);new_contact=False;new_lift=False
            for t in truths:
                if t["bilateral_grasp_contact"]:
                    if not inherited:new_contact=True
                    if not inherited and t["object_position_m"][2]-initial["object_position_m"][2]>=.02:new_lift=True
                if t["bilateral_grasp_contact"] and prev["bilateral_grasp_contact"]:
                    d=np.array(t["object_position_m"])-prev["object_position_m"]
                    supported+=float(np.linalg.norm(d));supported_xy+=float(np.linalg.norm(d[:2]))
                prev=t
            initial_score=score_episode(initial,initial,[],e.scene.config,[])
            row=dict(slot,condition=args.condition,steps=steps,success_4s=success and steps<=100,success_15s=success,
                first_success_s=steps/25 if success else None,initial_grasp_inherited=inherited,
                new_bilateral_contact=new_contact,new_grasp_and_lift=new_lift,
                supported_path_m=supported,supported_xy_path_m=supported_xy,
                inherited_physical_endpoint_without_stability=physical_endpoint(initial_score),
                preservation_only=physical_endpoint(initial_score),
                termination="strict_success" if success else ("limit_violation" if infos[-1]["limit_violation"] else "time_limit"),
                reward_success=int(success),control_cost=sum(info["effort"]*.0001 for info in infos),
                discounted_return=sum((.99**j)*r for j,r in enumerate(rewards)),
                undiscounted_return=sum(rewards),final_score=scores[-1],at_4s_score=scores[min(99,steps-1)],
                observation_sha256=obs_digest.hexdigest())
            oldfolder={"U0":"eval-untrained-v1","A32":"eval-adaptive-32k-v1","U32":"eval-uniform-32k-v1"}.get(args.condition)
            if oldfolder:
                old=json.loads((source/oldfolder/f"episode-{idx}.json").read_text())
                n=min(100,len(old["actions"]),len(actions))
                row["prior_prefix_action_max_abs"]=float(np.max(np.abs(np.array(old["actions"][:n])-np.array(actions[:n]))))
                row["prior_prefix_object_pose_max_abs"]=float(max(np.max(np.abs(np.array(old["truths"][j]["object_position_m"])-np.array(truths[j]["object_position_m"]))) for j in range(n)))
                row["prior_4s_success_agrees"]=bool(old["summary"]["success"])==row["success_4s"]
            dump(destination/f"episode-{idx}.json",dict(summary=row,initial_truth=initial,actions=actions,truths=truths,rewards=rewards,infos=infos,scores=scores))
            rows_out.append(row);dump(destination/"summary.json",dict(condition=args.condition,episodes=rows_out,elapsed_s=time.time()-started))
            print(json.dumps({k:v for k,v in row.items() if k not in ("final_score","at_4s_score")}),flush=True)
        dump(destination/"complete.json",dict(episodes=len(rows_out),wall_s=time.time()-started,no_training=True))
    finally:
        for e in envs.values():e.close()

def main():
    p=argparse.ArgumentParser();p.add_argument("operation",choices=["audit","evaluate"])
    p.add_argument("--source",required=True);p.add_argument("--output",required=True)
    p.add_argument("--audit");p.add_argument("--condition",choices=["U0","A32","U32","OPEN"])
    a=p.parse_args();audit(a) if a.operation=="audit" else evaluate(a)
if __name__=="__main__":main()
