"""Post-hoc held-state description from the frozen FL source-0001 trace; no execution."""
from pathlib import Path
import json,gzip,time
import numpy as np
OUT=Path("/var/tmp/typhon-load01-20261013")
d=OUT/"evaluation/FL/source/load01-source-demo_0001"
rows=[json.loads(x)for x in gzip.open(d/"controller_trace.jsonl.gz","rt")]
late=[r for r in rows if r["step"]>=400];targets=np.array([r["target"]for r in late])
s=next(x for x in json.loads((d.parent/"summary.json").read_text())if x["id"]==d.name)
v={"schema":"load01-posthoc-held-state-diagnostic-v1","created":time.time(),"source":str(d),"policy_changed":False,"new_rollouts":0,
"supported_pickup":s["supported_lift_10mm_3steps"],"maximum_lift_mm":s["maximum_lift_above_initial_mm"],"longest_supported_lift_s":s["longest_supported_lift_s"],"max_horizontal_transport_while_supported_mm":s["horizontal_transport_while_supported_mm"],
"late_step_range":[late[0]["step"],late[-1]["step"]],"late_command_xyz_min":targets[:,:3].min(0).tolist(),"late_command_xyz_max":targets[:,:3].max(0).tolist(),"late_command_xyz_range_mm":(np.ptp(targets[:,:3],axis=0)*1000).tolist(),"late_gripper_sigmoid_min_max":[float(targets[:,9].min()),float(targets[:,9].max())],
"late_tracking_error_mm_p50_p95_max":np.percentile([r["position_error_mm"]for r in late],[50,95,100]).tolist(),"minimum_joint_limit_margin_rad":min(r["nearest_joint_limit_rad"]for r in rows),
"observed_failure":"Object grasped and lifted, then held near source. Late policy XYZ requests stay near raised pickup; final gripper remains closed. Not an early-release failure.",
"causal_limit":"A near-limit joint is observed, but this trace does not establish whether it causes the policy to stall. Controller follows the locally requested targets closely; no long transport request is visible in the late window."}
(OUT/"analysis/held-state-diagnostic.json").write_text(json.dumps(v,indent=2)+"\n")
print(json.dumps(v,indent=2))
