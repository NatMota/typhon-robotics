"""Render two fixed diagnostic states; replay must match saved physics."""
from pathlib import Path
import json
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from experiments.alphanist_pilot.bank import load_bank
from experiments.alphanist_pilot.environment import PandaJointEnv

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 source=Path(a.source);out=Path(a.output);media=out/"media";media.mkdir(exist_ok=False)
 meta,states=load_bank(source/"bank-v1")
 sheet=Image.new("RGB",(1024,660),(18,22,28));draw=ImageDraw.Draw(sheet)
 draw.text((12,8),"ALPHA02: initial / untrained / adaptive32k / OPEN-only. Frozen policies, maximum 15 seconds.",fill="white")
 proof=[]
 for row,idx in enumerate((16,54)):
  record=meta["records"][idx]
  for col,condition in enumerate(("U0","A32","OPEN"),start=1):
   log=json.loads((out/"evaluation"/condition/f"episode-{idx}.json").read_text())
   env=PandaJointEnv(record["task"],375)
   try:
    env.reset(states[idx]);initial=env.scene.frame()
    if col==1:
     sheet.paste(Image.fromarray(initial),(0,65+row*300))
     draw.text((4,42+row*300),record["task"]+f" reset {idx}: initial",fill="white")
    writer=imageio.get_writer(media/f"{condition}-reset-{idx}.mp4",fps=12.5)
    writer.append_data(initial)
    for i,action in enumerate(log["actions"]):
     _,_,done,info=env.step(action)
     if i%2==1:writer.append_data(env.scene.frame())
    final=env.scene.frame();writer.append_data(final);writer.close()
    truth=env.scene.truth();expected=log["truths"][-1]
    error=float(np.max(np.abs(np.array(truth["object_position_m"])-expected["object_position_m"])))
    qe=float(np.max(np.abs(np.array(truth["object_quat_wxyz"])-expected["object_quat_wxyz"])))
    assert error<1e-8 and qe<1e-8 and info["success"]==log["summary"]["success_15s"]
    sheet.paste(Image.fromarray(final),(col*256,65+row*300))
    label=f'{condition}: '+("success" if info["success"] else "not completed")+f' ({len(log["actions"])/25:.2f}s)'
    draw.text((col*256+4,42+row*300),label,fill="white")
    proof.append(dict(condition=condition,reset_index=idx,position_error_m=error,quaternion_error=qe,success=info["success"]))
   finally:env.close()
 draw.text((10,640),"No new grasps. Reset54 begins held above the loading goal. Blue-2 image recognition is not tested.",fill="white")
 sheet.save(media/"comparison.png");(media/"replay-validation.json").write_text(json.dumps(proof,indent=2))
 print(json.dumps(proof))
if __name__=="__main__":main()
