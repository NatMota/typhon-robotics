"""Passive 2x2 audit of motion representation and held-evidence lifetime."""
import os
os.environ.setdefault("MUJOCO_GL","egl")
os.environ.setdefault("PYOPENGL_PLATFORM","egl")
import argparse,copy,gzip,hashlib,json,math,sys,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
from exp20261011_hierarchy import color_measure
from exp20261011_hierarchy_visual import project_points
from supervisor.feeding_loading.environment import FeedingLoadingEnv
import video06_motion as flow
cv2.setNumThreads(1)
MODES=["contour_legacy","contour_fresh","glyph_legacy","glyph_fresh"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def default(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 raise TypeError(type(v).__name__)
def save(p,v):
 Path(p).write_text(json.dumps(v,indent=2,default=default,allow_nan=False)+"\n")
class Latch:
 def __init__(self,fresh):
  self.fresh=fresh;self.count=0;self.held=False;self.last_proximity=None;self.last_confirm=None;self.last_positive=None
 def update(self,positive,near,closed,step):
  if positive:self.last_positive=step
  if self.fresh:
   self.count=self.count+1 if positive and closed else 0
   if self.count>=2:self.last_confirm=step
   if not closed:self.last_confirm=None
   self.held=bool(closed and self.last_confirm is not None and step-self.last_confirm<=10)
  else:
   self.count=self.count+1 if positive else (0 if not near else self.count)
   if self.count>=2 and closed:self.held=True
   if near and closed:self.last_proximity=step
   if not closed or self.last_proximity is None or step-self.last_proximity>10:
    self.held=False
    if not closed:self.count=0
  # Closed without support is unknown, not an assertion of no physical grasp.
  state=True if self.held else (None if closed else False)
  return {"held":self.held,"state":state,"counter":self.count,
          "positive_age_s":None if self.last_positive is None else (step-self.last_positive)/20,
          "last_confirmation_step":self.last_confirm}
class Motion:
 def __init__(self,config,tracking):
  self.config=config;self.tracking=tracking;self.previous=None
 def update(self,obs,step):
  if set(obs)-{"rgb","proprio","task","cameras"}:raise ValueError("unexpected_observation_channel")
  eef=np.asarray(obs["proprio"]["eef_position_m"],float)
  span=float(np.abs(obs["proprio"]["gripper_joint_position_m"]).sum())
  closed=.008<span<.060
  contour_positive=False;contour_near=False;glyph_positive=False;glyph_near=False
  current=[];diagnostics=[]
  for i,(view,cam) in enumerate([("primary","overhead"),("secondary","oblique")]):
   rgb=obs["rgb"][view]
   _,body=color_measure(rgb)
   ee=np.asarray(project_points([eef],self.config["cameras"][cam])[0],float)
   bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
   gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
   mask,area=flow.blue_mask(bgr,view,self.tracking)
   raw_pts=cv2.findNonZero(mask)
   gc=None if raw_pts is None else raw_pts.reshape(-1,2).mean(0)
   bn=bool(body["visible"] and np.linalg.norm(np.asarray(body["center_px"])-ee)<40)
   gn=bool(gc is not None and np.linalg.norm(gc-ee)<40)
   contour_near|=bn;glyph_near|=gn
   d={"view":view,"body_visible":bool(body["visible"]),"blue_area":area,"contour_near":bn,"glyph_near":gn,
      "eef_px":ee.tolist(),"body_center_px":body.get("center_px"),"glyph_center_px":None if gc is None else gc.tolist(),
      "contour_positive":False,"glyph_positive":False,"flow":None,"eef_delta_px":None}
   if self.previous is not None:
    old=self.previous[i];de=ee-old["eef"];dn=float(np.linalg.norm(de));d["eef_delta_px"]=de.tolist()
    if body["visible"] and old["body"]["visible"] and closed:
     do=np.asarray(body["center_px"])-np.asarray(old["body"]["center_px"])
     ok=bool(bn and dn>.3 and np.linalg.norm(do)>.2 and np.linalg.norm(do-de)<max(3.,dn*.7))
     contour_positive|=ok;d["contour_positive"]=ok;d["contour_delta_px"]=do.tolist()
    f=flow.estimate_pair(old["gray"],gray,old["mask"],mask,self.tracking);d["flow"]=f
    if f["qualified"] and closed:
     do=np.asarray(f["median_delta_px"])
     ok=bool(gn and dn>.3 and np.linalg.norm(do)>.2 and np.linalg.norm(do-de)<max(3.,dn*.7))
     glyph_positive|=ok;d["glyph_positive"]=ok
   current.append({"body":body,"eef":ee,"gray":gray,"mask":mask})
   diagnostics.append(d)
  self.previous=current
  return {"closed":closed,"span_m":span,"contour":(contour_positive,contour_near),
          "glyph":(glyph_positive,glyph_near),"views":diagnostics}
def self_check(tracking):
 a=Latch(False);b=Latch(True)
 # Two isolated positives accumulate historically but cannot establish consecutive fresh support.
 for step,positive in [(0,True),(4,False),(8,True)]:
  x=a.update(positive,True,True,step);y=b.update(positive,True,True,step)
 assert x["held"] and not y["held"]
 b=Latch(True)
 for step,positive in [(0,True),(4,True),(8,False),(12,False),(16,False)]:y=b.update(positive,True,True,step)
 assert not y["held"] and y["state"] is None
 assert b.update(False,False,False,20)["state"] is False
 im=np.full((120,120,3),200,np.uint8)
 cv2.putText(im,"2",(35,90),cv2.FONT_HERSHEY_SIMPLEX,2.5,(220,45,30),7)
 mask,_=flow.blue_mask(im,"primary",tracking)
 gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);checks=[]
 for dx,dy in [(0,0),(3,-2),(-4,3)]:
  new=cv2.warpAffine(im,np.float32([[1,0,dx],[0,1,dy]]),(120,120),borderValue=(200,200,200))
  m,_=flow.blue_mask(new,"primary",tracking)
  r=flow.estimate_pair(gray,cv2.cvtColor(new,cv2.COLOR_BGR2GRAY),mask,m,tracking)
  assert r["qualified"],r
  error=float(np.linalg.norm(np.asarray(r["median_delta_px"])-[dx,dy]));assert error<.35,error
  checks.append({"delta":[dx,dy],"error_px":error})
 r=flow.estimate_pair(gray,np.zeros_like(gray),mask,np.zeros_like(mask),tracking)
 assert not r["qualified"]
 return {"latch_gap_expiry_open_checks":3,"known_image_translation_checks":checks,"full_occlusion_rejected":True}
def execute(protocol,out,part,parts):
 cfg=json.loads(protocol.read_text())
 out.mkdir(parents=True,exist_ok=True)
 status=[];begin=time.monotonic()
 for index,c in enumerate(cfg["cases"]):
  if index%parts!=part:continue
  dest=out/c["id"];dest.mkdir(exist_ok=False)
  folder=Path(c["path"]);episode=json.loads((folder/"episode.json").read_text())
  if c["expert"]:
   source=folder/"robot_actions.jsonl";archive=[json.loads(l) for l in source.read_text().splitlines()]
   actions=[r["normalized_command"] for r in archive];expected=archive[-1]["post_action_truth"]
  else:
   source=folder/"trajectory.jsonl.gz";archive=[json.loads(l) for l in gzip.open(source,"rt")]
   actions=[r["requested_action"] for r in archive if r.get("requested_action") is not None];expected=episode["final_truth"]
  assert sha(source)==c["action_sha256"]
  original=Path(cfg["earlier_observer_root"])/c["id"]/"observations.jsonl.gz"
  originals={r["step"]:r for r in (json.loads(l) for l in gzip.open(original,"rt"))}
  env=FeedingLoadingEnv(c["config"]);rows=[];video=None
  detectors={k:Motion(c["config"],cfg["feature_tracking"]) for k in ["real","frozen_rgb","fixed_eef"]}
  states={k:{m:Latch(m.endswith("fresh")) for m in MODES} for k in detectors}
  first_obs=None;first_fp_saved=False;first_tp_saved=False
  try:
   env.reset(episode["seed"]);initial_z=env.truth()["object_position_m"][2]
   def inspect(i):
    nonlocal first_obs,video,first_fp_saved,first_tp_saved
    obs=env.observe()
    if first_obs is None:first_obs=copy.deepcopy(obs)
    variants={"real":obs,
      "frozen_rgb":{**obs,"rgb":first_obs["rgb"]},
      "fixed_eef":{**obs,"proprio":{**obs["proprio"],"eef_position_m":first_obs["proprio"]["eef_position_m"]}}}
    predictions={};all_evidence={}
    for key,ob in variants.items():
     evidence=detectors[key].update(ob,env.steps);all_evidence[key]=evidence
     predictions[key]={m:states[key][m].update(*evidence[m.split("_")[0]],evidence["closed"],env.steps) for m in MODES}
    # Independent scorer starts only after every observer and control prediction.
    truth=env.truth();contact=bool(truth["bilateral_grasp_contact"])
    lifted=float(truth["object_position_m"][2]-initial_z)
    supported=bool(contact and lifted>=.01)
    historical=originals[env.steps]
    sample_hashes={view:hashlib.sha256(im.tobytes()).hexdigest() for view,im in obs["rgb"].items()}
    row={"action_index":i,"step":env.steps,"input_rgb_sha256":sample_hashes,"predictions":predictions,
         "evidence":all_evidence["real"],
         "historical_held":historical["prediction"]["held"],
         "scorer_only":{"bilateral_contact":contact,"supported_lift":supported,"lift_m":lifted,"truth":truth,
           "historical_supported_lift":historical["scorer_only"]["supported_held"],
           "historical_object_position_difference_m":float(np.linalg.norm(np.asarray(truth["object_position_m"])-historical["scorer_only"]["truth"]["object_position_m"]))}}
    rows.append(row)
    panel=cv2.cvtColor(np.concatenate([obs["rgb"]["primary"],obs["rgb"]["secondary"]],1),cv2.COLOR_RGB2BGR)
    panel=cv2.copyMakeBorder(panel,70,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
    cv2.putText(panel,f'{c["id"]} step={env.steps} scorer supported={supported}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.53,(240,240,240),1)
    cv2.putText(panel," | ".join(m+":"+str(predictions["real"][m]["state"]) for m in MODES),(8,49),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1)
    if video is None:
     video=cv2.VideoWriter(str(dest/"preview.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),5.,(panel.shape[1],panel.shape[0]));assert video.isOpened()
    video.write(panel)
    fp=bool(predictions["real"]["contour_legacy"]["held"] and not supported)
    if i==0 or (fp and not first_fp_saved) or (supported and not first_tp_saved):
     name="initial" if i==0 else ("first-false-supported" if fp and not first_fp_saved else "first-supported")
     cv2.imwrite(str(dest/(name+".jpg")),panel,[cv2.IMWRITE_JPEG_QUALITY,92])
     if fp:first_fp_saved=True
     if supported:first_tp_saved=True
   inspect(0)
   for i,a in enumerate(actions,1):
    env.step(a)
    if i%4==0 or i==len(actions):inspect(i)
   final=env.truth()
   position_diff=float(np.linalg.norm(np.asarray(final["object_position_m"])-expected["object_position_m"]))
   yaw_diff=float(abs(math.atan2(math.sin(final["object_yaw_rad"]-expected["object_yaw_rad"]),math.cos(final["object_yaw_rad"]-expected["object_yaw_rad"]))))
   with gzip.open(dest/"observations.jsonl.gz","wt") as f:
    for row in rows:f.write(json.dumps(row,default=default,allow_nan=False)+"\n")
   summary={"case":c["id"],"samples":len(rows),"actions":len(actions),"source_action_sha256":c["action_sha256"],
     "final_position_difference_m":position_diff,"final_yaw_difference_rad":yaw_diff,
     "legacy_held_mismatches":sum(r["predictions"]["real"]["contour_legacy"]["held"]!=r["historical_held"] for r in rows),
     "truth_label_mismatches":sum(r["scorer_only"]["supported_lift"]!=r["scorer_only"]["historical_supported_lift"] for r in rows),
     "maximum_sample_position_difference_m":max(r["scorer_only"]["historical_object_position_difference_m"] for r in rows),
     "actions_changed":0}
   save(dest/"summary.json",summary);status.append(summary);save(out/f"worker-{part}-status.json",{"complete":False,"cases":status})
   print(json.dumps(summary),flush=True)
  finally:
   if video is not None:video.release()
   env.close()
 save(out/f"worker-{part}-status.json",{"complete":True,"cases":status,"seconds":time.monotonic()-begin})
def analyze(protocol,out):
 cfg=json.loads(protocol.read_text());rows=[];cases=[]
 for c in cfg["cases"]:
  d=out/c["id"];s=json.loads((d/"summary.json").read_text());cases.append(s)
  rows.extend([{**json.loads(l),"case":c["id"]} for l in gzip.open(d/"observations.jsonl.gz","rt")])
 assert len(cases)==8 and len(rows)==1219
 metrics={}
 for mode in MODES:
  metrics[mode]={}
  for label in ["bilateral_contact","supported_lift"]:
   y=np.array([r["scorer_only"][label] for r in rows],bool)
   pred=np.array([r["predictions"]["real"][mode]["held"] for r in rows],bool)
   metrics[mode][label]={"positive_labels":int(y.sum()),"predicted_positive":int(pred.sum()),
       "true_positive":int((y&pred).sum()),"false_positive":int((~y&pred).sum()),
       "precision":float((y&pred).sum()/pred.sum()) if pred.sum() else None,
       "recall":float((y&pred).sum()/y.sum()) if y.sum() else None,
       "false_positive_cases":sorted({r["case"] for r in rows if r["predictions"]["real"][mode]["held"] and not r["scorer_only"][label]})}
  metrics[mode]["unknown_samples"]=sum(r["predictions"]["real"][mode]["state"] is None for r in rows)
  metrics[mode]["held_samples_without_a_positive_pair_for_over_0_5_s"]=sum(r["predictions"]["real"][mode]["held"] and r["predictions"]["real"][mode]["positive_age_s"] is not None and r["predictions"]["real"][mode]["positive_age_s"]>.5 for r in rows)
 controls={key:{mode:sum(r["predictions"][key][mode]["held"] for r in rows) for mode in MODES} for key in ["frozen_rgb","fixed_eef"]}
 result={"cases":cases,"samples":len(rows),"metrics":metrics,"negative_control_held_counts":controls,
   "actions_changed":0,"controller_authority":False,
   "held_gate":{"minimum_precision":.95,"minimum_recall":.90},
   "historical_parity":all(s["legacy_held_mismatches"]==0 and s["truth_label_mismatches"]==0 for s in cases),
   "exact_replay_positions":all(s["maximum_sample_position_difference_m"]<=1e-6 and s["final_position_difference_m"]<=1e-6 for s in cases),
   "new_pose_estimates":0,"scope":"Repeated development observation panel; no new independent scenes or policy outcomes."}
 for mode in MODES:
  m=metrics[mode]["supported_lift"]
  metrics[mode]["held_component_gate_passed"]=bool(m["precision"] is not None and m["precision"]>=.95 and m["recall"]>=.90 and result["historical_parity"] and result["exact_replay_positions"] and all(controls[k][mode]==0 for k in controls))
 validation={"decoded_previews":0,"preview_frames":0}
 for c in cfg["cases"]:
  cap=cv2.VideoCapture(str(out/c["id"]/"preview.mp4"));n=0
  while True:
   ok,frame=cap.read()
   if not ok:break
   n+=1
  cap.release();expected=next(s["samples"] for s in cases if s["case"]==c["id"]);assert n==expected
  validation["decoded_previews"]+=1;validation["preview_frames"]+=n
 save(out/"summary.json",result);save(out/"validation.json",validation)
 print(json.dumps(result,indent=2))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("command",choices=["check","run","analyze"]);p.add_argument("--protocol",type=Path,required=True);p.add_argument("--out",type=Path);p.add_argument("--part",type=int,default=0);p.add_argument("--parts",type=int,default=1)
 a=p.parse_args()
 if a.command=="check":print(json.dumps(self_check(json.loads(a.protocol.read_text())["feature_tracking"])))
 elif a.command=="run":execute(a.protocol,a.out,a.part,a.parts)
 else:analyze(a.protocol,a.out)
