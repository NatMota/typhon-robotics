"""Render saved actions, asserting the physical replay matches logged outcomes."""
from pathlib import Path
import json
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from .bank import load_bank
from .environment import PandaJointEnv

def render(output):
    out=Path(output);media=out/"media";media.mkdir(exist_ok=False)
    meta,states=load_bank(out/"bank-v1")
    sheet=Image.new("RGB",(768,940),(18,22,28));draw=ImageDraw.Draw(sheet)
    draw.text((12,8),"Held-out Panda reset-state continuations: initial / untrained / adaptive32k",fill="white")
    proof=[]
    for row,idx in enumerate((16,54,22)):
        record=meta["records"][idx]
        for col,label in enumerate(("untrained-v1","adaptive-32k-v1"),start=1):
            log=json.loads((out/("eval-"+label)/("episode-"+str(idx)+".json")).read_text())
            env=PandaJointEnv(record["task"],100)
            try:
                env.reset(states[idx]);initial=env.scene.frame()
                if col==1:
                    sheet.paste(Image.fromarray(initial),(0,52+row*296));draw.text((4,32+row*296),record["task"]+" reset "+str(idx)+" initial",fill="white")
                writer=imageio.get_writer(media/(label+"-reset-"+str(idx)+".mp4"),fps=12.5)
                writer.append_data(initial)
                for i,action in enumerate(log["actions"]):
                    _,_,done,info=env.step(action)
                    if i%2==1:writer.append_data(env.scene.frame())
                writer.append_data(env.scene.frame());writer.close()
                end=env.scene.truth();expected=log["truths"][-1]
                error=float(np.max(np.abs(np.array(end["object_position_m"])-expected["object_position_m"])))
                quaterror=float(np.max(np.abs(np.array(end["object_quat_wxyz"])-expected["object_quat_wxyz"])))
                assert error<1e-8 and quaterror<1e-8 and info["success"]==log["summary"]["success"]
                sheet.paste(Image.fromarray(env.scene.frame()),(col*256,52+row*296))
                draw.text((4+col*256,32+row*296),label+" "+("success" if info["success"] else "failed"),fill="white")
                proof.append({"reset_index":idx,"model":label,"position_replay_max_error_m":error,
                              "quaternion_replay_max_error":quaterror,"success":info["success"]})
            finally:env.close()
    sheet.save(media/"comparison.png")
    (media/"replay-validation.json").write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof))
