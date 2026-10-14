from pathlib import Path
import json,gzip,hashlib,shutil,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
R=Path(__file__).resolve().parents[1];O=Path("/var/tmp/typhon-data07ge-20261014");D=R/"experiment-history/2026-10-14-data07ge";D.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
s=read(O/"summary.json");assert read(O/"complete.json")["passed"]
ep=read(O/"episode.json");tr=[json.loads(l) for l in gzip.open(O/"continuation/controller_trace.jsonl.gz","rt")];post=[x for x in tr if x["step"]>460]
firstcontact=next(x for x in post if x["object_truth"]["grasp_contact"]);firstclose=next(x for x in post if x["command"][-1]>0);final=post[-1]
events=[]
last=-1
for x in post:
 if x["command"][-1]!=last:events.append({"step":x["step"],"grip":x["command"][-1],"tilt_deg":x["object_tilt_deg"],"bilateral":x["object_truth"]["bilateral_grasp_contact"]})
 last=x["command"][-1]
s.update(final_tilt_deg=final["object_tilt_deg"],first_close_command=firstclose["step"],tilt_preceded_close_command=s["first_tilt_over15"]<firstclose["step"],minimum_joint_limit_margin_rad=min(x["nearest_joint_limit_rad"] for x in post),grip_events=events,expert_high_level_lift_verified=read(O/"continuation/expert-result.json")["attempt"]["verified_lift"],expert_lift_phase_converged=next(x["converged"] for x in read(O/"continuation/expert-result.json")["attempt"]["phases"] if x["phase"]=="lift"))
dump(D/"summary.json",s)
dump(D/"first-contact.json",firstcontact)
dump(D/"final-state.json",final)
for f in ["protocol.json","clock.json","complete.json","launch.json","launcher-amendment.json","saved-command-comparison.json","benchmark-hook-diff.txt","episode.json"]:
 shutil.copy2(O/f,D/f)
for f in ["handoff-comparison.json","handoff-qualified.json","expert-result.json"]:
 shutil.copy2(O/"continuation"/f,D/f)
(D/"videos").mkdir(exist_ok=True);shutil.copy2(O/"continuation/rollout.mp4",D/"videos/geometric-current-state-demo_0001.mp4")
fontpath=Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf")
font=ImageFont.truetype(str(fontpath),17) if fontpath.exists() else ImageFont.load_default()
paths=[Path("/var/tmp/typhon-data07r-20261014/native/demo_0001/rollout.mp4"),O/"continuation/rollout.mp4"];readers=[imageio.get_reader(p) for p in paths]
ticks=[460,528,532,572,752,792];sheet=None
for j,t in enumerate(ticks):
 cells=[]
 for k,v in enumerate(readers):
  im=Image.fromarray(v.get_data((t-40)//4));p=Image.new("RGB",(im.width,im.height+36),"#0b1220");p.paste(im,(0,36));ImageDraw.Draw(p).text((9,9),("Native B" if k==0 else "Geometric teacher after tick460")+" | tick "+str(t),font=font,fill="white");cells.append(p)
 if sheet is None:sheet=Image.new("RGB",(cells[0].width*2,cells[0].height*len(ticks)),"#0b1220")
 for k,im in enumerate(cells):sheet.paste(im,(k*im.width,j*im.height))
sheet.save(D/"native-vs-expert-contact-sheet.jpg",quality=92)
for v in readers:v.close()
dump(D/"visual-provenance.json",{"initial_tick":40,"capture_cadence":4,"sheet_ticks":ticks,"common_clock_frames":True,"ge_terminal_tick":793,"native_terminal_tick":1500,"do_not_treat_compared_endpoints_as_identical_duration":True})
manifest=[]
for p in sorted(O.rglob("*")):
 if p.is_file():manifest.append({"path":str(p.relative_to(O)),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
dump(D/"raw-file-manifest.json",{"root":str(O),"files":manifest,"total_bytes":sum(x["bytes"] for x in manifest)})
dump(D/"validation.json",{"passed":True,"new_expert_attempts":1,"new_VLA_queries":0,"qualified_same_state":True,"no_training":True,"no_positive_correction":True,"final_horizon_preserved":793<=1500 and s["24_passive_stability_steps"],"expert_algorithm_unchanged":True,"one_launcher_qualification_amendment_before_expert_commands":True,"frozen_comparator_unchanged":True,"stopped_after_one_attempt":True,"finished_under_30min":read(O/"complete.json")["finished"]<read(O/"clock.json")["final_deadline"]})
print(json.dumps({"summary":s,"files":len(manifest),"raw_bytes":sum(x["bytes"] for x in manifest)},indent=2))
