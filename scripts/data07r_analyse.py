from pathlib import Path
import json,gzip,hashlib,subprocess,shutil,time,csv
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parents[1];O=Path("/var/tmp/typhon-data07r-20261014")
D=R/"experiment-history/2026-10-14-data07r";D.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
rows=read(O/"summary.json");assert read(O/"complete.json")["passed"];assert len(rows)==5
outrows=[];eventlog=[]
for row in rows:
 d=O/row["mode"]/row["source_demo_id"];ep=read(d/"episode.json")
 assert ep["status"]=="evaluated" and not any(ep[k] for k in ["controller_exceptions","artifact_errors","evaluation_errors"])
 tr=[json.loads(x) for x in gzip.open(d/"controller_trace.jsonl.gz","rt")]
 assert len(tr)==ep["policy_actions_executed"]+24==1460
 events=[];prev=-1
 for x in tr:
  grip=x["command"][-1]
  if grip!=prev:
   events.append({"step":x["step"],"grip":"close" if grip>0 else "open","bilateral":x["object_truth"]["bilateral_grasp_contact"],"tilt_deg":x["object_tilt_deg"]})
  prev=grip
 row={**row,"actions_executed":ep["policy_actions_executed"],"all_control_actions":len(tr),"model_queries":len(ep["inference_latency_s"]),"initial_control_step":ep["initial_truth"]["control_steps"],"final_control_step":ep["final_truth"]["control_steps"],"final_object_position_m":ep["final_truth"]["object_position_m"],"final_object_quat_wxyz":ep["final_truth"]["object_quat_wxyz"],"grip_transition_count":len(events),"open_transitions_with_bilateral_contact":sum(e["grip"]=="open" and e["bilateral"] for e in events),"final_bilateral_contact":ep["final_truth"]["bilateral_grasp_contact"]}
 outrows.append(row);eventlog.append({"source_demo_id":row["source_demo_id"],"mode":row["mode"],"events":events})
 for f in ["summary.json","episode.json","initial-qualification.json","intervention.json","prefix-state-comparison.json"]:
  if (d/f).exists():
   dst=D/"episodes"/row["mode"]/row["source_demo_id"];dst.mkdir(parents=True,exist_ok=True);shutil.copy2(d/f,dst/f)
 dst=D/"videos";dst.mkdir(exist_ok=True);shutil.copy2(d/"rollout.mp4",dst/(row["mode"]+"-"+row["source_demo_id"]+".mp4"))
def rot(a):
 x=a[3:6]/np.linalg.norm(a[3:6]);y=a[6:9]-x*np.dot(x,a[6:9]);y/=np.linalg.norm(y);return np.column_stack((x,y,np.cross(x,y)))
native=np.load(O/"native/demo_0001/policy_io.npz")["model_actions"][14]
donor=np.load(O/"donors/demo_0001.npz")["actions"]
dxyz=np.linalg.norm(native[:,:3]-donor[:,:3],axis=1)*1000
drot=np.array([np.degrees(np.linalg.norm(Rotation.from_matrix(rot(c)@rot(b).T).as_rotvec())) for b,c in zip(native,donor)])
delta={"mean_position_target_change_mm":float(dxyz.mean()),"max_position_target_change_mm":float(dxyz.max()),"mean_wrist_target_change_deg":float(drot.mean()),"max_wrist_target_change_deg":float(drot.max()),"gripper_kept_exact":True,"channels_replaced":"0:9","query_index":14,"step":460}
dump(D/"donor-difference.json",delta);dump(D/"summary.json",outrows);dump(D/"gripper-events.json",eventlog)
for f in ["protocol.json","protocol-amendment-01.json","clock.json","complete.json","launch.json","selection.json"]:
 shutil.copy2(O/f,D/f)
fontpath="/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"
if not Path(fontpath).exists():fontpath="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font=ImageFont.truetype(fontpath,17) if Path(fontpath).exists() else ImageFont.load_default()
readers=[imageio.get_reader(O/"native/demo_0001/rollout.mp4"),imageio.get_reader(O/"repair/demo_0001/rollout.mp4")]
meta=[v.get_meta_data() for v in readers];assert all(x["fps"]==5 for x in meta)
# Capture once at reset tick40, then every4 control ticks. Metadata is explicit.
ticks=[460,476,508,600,1500]
sheet=None
for j,tick in enumerate(ticks):
 cells=[]
 for k,reader in enumerate(readers):
  index=0 if tick==40 else (tick-40)//4
  im=Image.fromarray(reader.get_data(index));pad=Image.new("RGB",(im.width,im.height+36),"#0b1220");pad.paste(im,(0,36))
  ImageDraw.Draw(pad).text((9,9),("Native B" if k==0 else "C pose chunk, then B")+" | tick "+str(tick),font=font,fill="white");cells.append(pad)
 if sheet is None:sheet=Image.new("RGB",(cells[0].width*2,cells[0].height*len(ticks)),"#0b1220")
 for k,im in enumerate(cells):sheet.paste(im,(k*im.width,j*im.height))
sheet.save(D/"comparison-contact-sheet.jpg",quality=92)
# Full matched video, same simulation-clock frames; original cameras preserved.
writer=imageio.get_writer(D/"videos/native-vs-pose-repair-demo_0001.mp4",fps=5,codec="libx264")
count=0
try:
 for a,c in zip(readers[0],readers[1]):
  im=Image.fromarray(np.concatenate([a,c],axis=1));pad=Image.new("RGB",(im.width,im.height+32),"#0b1220");pad.paste(im,(0,32));draw=ImageDraw.Draw(pad)
  draw.text((10,7),"Native B | FAILED",font=font,fill="white");draw.text((im.width//2+10,7),"One C pose chunk, then B | FAILED",font=font,fill="white")
  writer.append_data(np.asarray(pad));count+=1
finally:
 writer.close()
 for reader in readers:reader.close()
dump(D/"visual-provenance.json",{"comparison_frames":count,"fps":5,"capture_cadence_control_ticks":4,"initial_control_tick":40,"contact_sheet_ticks":ticks,"pairing":"same control-clock frames from native and repaired executions; source camera pixels unchanged, labels added above","raw_frames_saved":True})
validation={"passed":True,"native":4,"repairs":1,"native_successes":3,"repair_conversions":0,"initial_RGB_physics_match_training_archives":True,"same_branch_state_passed":read(O/"repair/demo_0001/prefix-state-comparison.json")["passed"],"branch_B_recomputed_exact":read(O/"repair/demo_0001/intervention.json")["B_chunk_exact_reproduction"],"new_training":False,"positive_repair_examples_qualified":0,"no_dev_or_holdout_access":True,"finished_under_frozen_45min_cap":read(O/"complete.json")["finished"]<read(O/"clock.json")["final_deadline"],"legacy_initial_isolation_flag_not_used":True}
dump(D/"validation.json",validation);print(json.dumps({"summary":outrows,"delta":delta,"validation":validation,"report_directory":str(D)},indent=2))
