"""OBS06 passive aperture/release diagnostic; no control authority."""
import argparse,gzip,json,math,sys,time,hashlib
from pathlib import Path
import numpy as np,cv2
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
from obs05_feature_motion import Motion,Latch,FeedingLoadingEnv,default
MODES=("legacy","aperture","opening","both")
class Observer:
 def __init__(self):self.latches={m:Latch(False) for m in MODES};self.previous=None
 def update(self,evidence,step):
  span=evidence["span_m"];rate=0. if self.previous is None else (span-self.previous[1])/((step-self.previous[0])/20)
  self.previous=(step,span);positive,near=evidence["contour"];result={}
  for mode,latch in self.latches.items():
   closed=evidence["closed"]
   if mode in ("aperture","both"):closed=closed and span>=.024
   if mode in ("opening","both"):closed=closed and rate<=.025
   result[mode]=latch.update(positive and closed,near,closed,step)
  return result,rate
def save(path,x):path.write_text(json.dumps(x,indent=2,default=default,allow_nan=False)+"\n")
def run(protocol,out,part,parts):
 cfg=json.loads(protocol.read_text());begin=time.monotonic();status=[]
 for k,c in enumerate(cfg["cases"]):
  if k%parts!=part:continue
  dest=out/c["id"];dest.mkdir(parents=True,exist_ok=False)
  folder=Path(c["path"]);ep=json.loads((folder/"episode.json").read_text())
  src=folder/"trajectory.jsonl.gz";assert hashlib.sha256(src.read_bytes()).hexdigest()==c["action_sha256"]
  original=[json.loads(l) for l in gzip.open(src,"rt")];actions=[r["requested_action"] for r in original]
  expected={r["step"]:r["truth"] for r in original}
  env=FeedingLoadingEnv(c["config"]);observer=Observer();motion=Motion(c["config"],cfg["feature_tracking"])
  rows=[];video=None;maxdiff=0.
  try:
   env.reset(c["seed"]);initial=env.truth();z=initial["object_position_m"][2]
   def sample(index):
    nonlocal video,maxdiff
    obs=env.observe();e=motion.update(obs,env.steps);pred,rate=observer.update(e,env.steps)
    # Scoring begins after all predictions.
    truth=env.truth();target=ep["initial_truth"] if index==0 else expected[env.steps]
    diff=float(np.linalg.norm(np.asarray(truth["object_position_m"])-target["object_position_m"]));maxdiff=max(maxdiff,diff)
    contact=bool(truth["bilateral_grasp_contact"]);lift=truth["object_position_m"][2]-z
    row={"step":env.steps,"index":index,"predictions":pred,"aperture_rate_m_s":rate,"evidence":{"closed":e["closed"],"span_m":e["span_m"],"contour":e["contour"],"views":[{key:v[key] for key in ["view","body_visible","body_center_px","eef_px","contour_positive","contour_delta_px","eef_delta_px"] if key in v} for v in e["views"]]},"scorer_only":{"bilateral_contact":contact,"supported_lift":bool(contact and lift>=.01),"lift_m":lift,"truth":truth,"position_diff_m":diff}}
    rows.append(row)
    im=cv2.cvtColor(np.concatenate([obs["rgb"]["primary"],obs["rgb"]["secondary"]],1),cv2.COLOR_RGB2BGR)
    im=cv2.copyMakeBorder(im,70,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
    cv2.putText(im,f'{c["id"]} step={env.steps} contact={contact}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
    cv2.putText(im," | ".join(m+":"+str(pred[m]["held"]) for m in MODES),(8,49),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    if video is None:video=cv2.VideoWriter(str(dest/"preview.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),5.,(im.shape[1],im.shape[0]));assert video.isOpened()
    video.write(im)
   sample(0)
   for i,a in enumerate(actions,1):
    env.step(a)
    if i%4==0 or i==len(actions):sample(i)
   with gzip.open(dest/"observations.jsonl.gz","wt") as f:
    for r in rows:f.write(json.dumps(r,default=default,allow_nan=False)+"\n")
   result={"case":c["id"],"model":c["model"],"group":c["group"],"layout":c["layout"],"seed":c["seed"],"samples":len(rows),"actions":len(actions),"maximum_sample_position_difference_m":maxdiff,"final_yaw_difference_rad":abs(math.atan2(math.sin(env.truth()["object_yaw_rad"]-ep["final_truth"]["object_yaw_rad"]),math.cos(env.truth()["object_yaw_rad"]-ep["final_truth"]["object_yaw_rad"]))),"actions_changed":0}
   save(dest/"summary.json",result);status.append(result);print(json.dumps(result),flush=True)
   save(out/f"worker-{part}-status.json",{"complete":False,"cases":status})
  finally:
   if video is not None:video.release()
   env.close()
 save(out/f"worker-{part}-status.json",{"complete":True,"cases":status,"seconds":time.monotonic()-begin})
def score(rows):
 result={}
 for mode in MODES:
  result[mode]={}
  for label in ("bilateral_contact","supported_lift"):
   tp=sum(r["predictions"][mode]["held"] and r["scorer_only"][label] for r in rows)
   fp=sum(r["predictions"][mode]["held"] and not r["scorer_only"][label] for r in rows)
   fn=sum(not r["predictions"][mode]["held"] and r["scorer_only"][label] for r in rows)
   result[mode][label]={"tp":tp,"fp":fp,"fn":fn,"precision":tp/(tp+fp) if tp+fp else None,"recall":tp/(tp+fn) if tp+fn else None}
 return result
def analyze(protocol,out):
 cfg=json.loads(protocol.read_text());rows=[];cases=[];frames=0
 for c in cfg["cases"]:
  dest=out/c["id"];s=json.loads((dest/"summary.json").read_text());cases.append(s)
  rs=[json.loads(l) for l in gzip.open(dest/"observations.jsonl.gz","rt")];rows+=rs
  cap=cv2.VideoCapture(str(dest/"preview.mp4"));n=0
  while cap.read()[0]:n+=1
  cap.release();assert n==len(rs);frames+=n
 summary={"cases":cases,"samples":len(rows),"metrics":score(rows),"maximum_position_difference_m":max(s["maximum_sample_position_difference_m"] for s in cases),"maximum_final_yaw_difference_rad":max(s["final_yaw_difference_rad"] for s in cases),"videos_decoded":len(cases),"frames_decoded":frames,"control_authority":False,"new_training":False}
 assert summary["maximum_position_difference_m"]<1e-6
 assert summary["maximum_final_yaw_difference_rad"]<1e-6
 save(out/"summary.json",summary)
 old=[]
 for p in sorted(Path(cfg["development_root"]).glob("*/observations.jsonl.gz")):
  o=Observer()
  for line in gzip.open(p,"rt"):
   r=json.loads(line);pred,rate=o.update(r["evidence"],r["step"]);old.append({"predictions":pred,"scorer_only":r["scorer_only"]})
 save(out/"development_diagnostics.json",{"samples":len(old),"metrics":score(old),"scope":"Previously inspected OBS05 development data; not qualification evidence."})
 print(json.dumps(summary,indent=2))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("command",choices=["run","analyze"]);p.add_argument("--protocol",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--part",type=int,default=0);p.add_argument("--parts",type=int,default=1);a=p.parse_args()
 if a.command=="run":run(a.protocol,a.out,a.part,a.parts)
 else:analyze(a.protocol,a.out)
