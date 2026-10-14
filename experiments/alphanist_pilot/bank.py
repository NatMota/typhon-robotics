"""Small controller-derived reset bank; this is NOT OmniReset or imitation data."""
import json, time
from pathlib import Path
import numpy as np
from .environment import make_scene, snapshot
from supervisor.feeding_loading.runner import scripted_attempt
from supervisor.feeding_loading.evaluation import score_episode

def build_bank(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=[];states=[];episodes=[]
    for task in ("feeding","loading"):
        env=make_scene(task)
        try:
            for split,seeds in (("train",(601,602)),("development",(1701,1702))):
                for seed in seeds:
                    env.reset(seed)
                    initial=env.truth(); captured=[(snapshot(env),initial)]
                    def capture(e): captured.append((snapshot(e),e.truth()))
                    env.capture_callback=capture
                    attempt,retreat,yaw=scripted_attempt(env,env.episode_config)
                    stable=[]
                    for _ in range(12):
                        env.tick(retreat,yaw,-1,2);stable.append(env.truth())
                    env.capture_callback=None
                    score=score_episode(initial,env.truth(),stable,env.episode_config,[attempt])
                    episodes.append({"task":task,"seed":seed,"split":split,"expert_success":bool(score["success"]),
                                     "expert_verified_lift":bool(attempt["verified_lift"]),"score":score})
                    eligible=[]
                    goal=env.config["task"]
                    for i,(state,truth) in enumerate(captured):
                        pos=np.array(truth["object_position_m"])[:2]
                        yawerr=abs(np.arctan2(np.sin(truth["object_yaw_rad"]-goal["target_yaw_rad"]),
                                             np.cos(truth["object_yaw_rad"]-goal["target_yaw_rad"])))
                        # Do not reward a reset that is already completed; strict reward is checked after physical settling.
                        completed=np.linalg.norm(pos-goal["target_position_xy_m"])<=.015 and yawerr<=np.deg2rad(10) and truth["gripper_open"] and not truth["grasp_contact"]
                        if not completed: eligible.append(i)
                    if not eligible: raise RuntimeError("No unsolved reset states")
                    selected=sorted(set([eligible[j] for j in np.linspace(0,len(eligible)-1,8).astype(int)]))
                    for i in selected:
                        state,truth=captured[i];row={"index":len(states),"task":task,"split":split,"seed":seed,
                          "source":"geometric_controller_rollout_state_only","source_control_step":truth["control_steps"],
                          "bilateral_contact":truth["bilateral_grasp_contact"],"gripper_open":truth["gripper_open"],
                          "object_position_m":truth["object_position_m"],"object_yaw_rad":truth["object_yaw_rad"],
                          "reset_label_supplied_to_actor":False,"expert_episode_success":bool(score["success"])}
                        rows.append(row);states.append(state)
                    print(json.dumps(episodes[-1]),flush=True)
        finally: env.close()
    arrays={k:np.stack([s[k] for s in states]) for k in states[0]}
    np.savez_compressed(output/"states.npz",**arrays)
    (output/"manifest.json").write_text(json.dumps({"schema":"alphanist-inspired-reset-bank-v1",
       "created_unix":time.time(),"records":rows,"episodes":episodes,"observation":"simulator surface point clouds and proprioception",
       "not_human_demonstrations":True,"no_imitation_action_targets":True,
       "difference_from_paper":"Small controller-derived bank replaces OmniReset; two DUPLO tasks replace 20 NIST assemblies."},indent=2))
    return len(rows)

def load_bank(path):
    path=Path(path);meta=json.loads((path/"manifest.json").read_text())
    with np.load(path/"states.npz",allow_pickle=False) as z:
        states=[{k:np.array(z[k][i],copy=True) for k in z.files} for i in range(len(meta["records"]))]
    return meta,states
