"""LIVE01-R fixed-state capture diagnostic. No model loading or generated action."""
import os,sys,time,json,gzip,pickle,random,hashlib,subprocess,traceback
from pathlib import Path
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYOPENGL_PLATFORM","egl")
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/"scripts")];os.chdir(R)
OUT=Path("/var/tmp/typhon-live01-r-20261015/attempt-02")
OLD=Path("/var/tmp/typhon-live01-p-20261015")
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def limit(seconds=2400):assert time.time()-read(OUT/"clock.json")["start_epoch"]<seconds,"LIVE01-R wall-clock limit"
def worker():
 import numpy as np
 import torch
 from supervisor.adaptation.m6_contract import reorder,DOMAIN
 from transformers import AutoProcessor
 from supervisor.adaptation.observations import views,object_crop
 from supervisor.planning.rgb_lineage import PreprocessLineage,digest
 wire=sys.stdout.buffer;sys.stdout=sys.stderr
 proc=AutoProcessor.from_pretrained(str(R/".cache-xvla/checkpoint"),trust_remote_code=True,local_files_only=True)
 proc.language_max_length=128
 def reply(x):pickle.dump(x,wire,protocol=4);wire.flush()
 reply({"ready":True,"model_loaded":False})
 while True:
  q=pickle.load(sys.stdin.buffer)
  if q["op"]=="close":break
  try:
   rgb=dict(zip(["primary","secondary"],q["images"]))
   if "_rgb_provenance" in q:
    with PreprocessLineage(q,proc) as lineage:
     images=views(rgb,"O2");inputs=proc(images=images,language_instruction=q["instruction"])
    meta=lineage.finish(images,inputs,OUT/("lineage-%02d.json"%q["audit_index"]))
   else:
    images=views(rgb,"O2");inputs=proc(images=images,language_instruction=q["instruction"]);meta=None
   inputs["proprio"]=reorder(torch.as_tensor(q["proprio"],dtype=torch.float32).unsqueeze(0));inputs["domain_id"]=torch.tensor([DOMAIN],dtype=torch.long)
   reply({"arrays":{k:v.detach().cpu().numpy().copy() for k,v in inputs.items()},"crop":object_crop(rgb["primary"])[1],"lineage":meta})
  except BaseException:reply({"error":traceback.format_exc()})
def different(a,b,path=""):
 import numpy as np
 if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
  return [] if isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b,equal_nan=True) else [path]
 if isinstance(a,dict) and isinstance(b,dict):
  if set(a)!=set(b):return [path+"/keys"]
  return [p for k in a for p in different(a[k],b[k],path+"/"+str(k))]
 if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
  if len(a)!=len(b):return [path+"/len"]
  return [p for i,(x,y) in enumerate(zip(a,b)) for p in different(x,y,path+"/"+str(i))]
 return [] if a==b else [path]
def jsonable(x):
 import numpy as np
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
 return x
def graphics(env):
 import numpy as np
 ctx=env.env.sim._render_context_offscreen;scn=ctx.scn;sim=env.env.sim
 def fields(obj,keys):
  out={}
  for k in keys:
   if not hasattr(obj,k):continue
   v=getattr(obj,k)
   if isinstance(v,np.ndarray):out[k]=v.copy().tolist()
   elif isinstance(v,(str,float,int,bool,np.generic)):out[k]=jsonable(v)
   else:
    try:out[k]=int(v)
    except (ValueError,TypeError):pass
  return out
 gfields=["type","dataid","objtype","objid","category","texid","texuniform","texcoord","segid","size","pos","mat","rgba","emission","specular","shininess","reflectance","label"]
 lfields=["pos","dir","attenuation","cutoff","exponent","ambient","diffuse","specular","headlight","directional","castshadow"]
 cfields=["pos","forward","up","frustum_center","frustum_bottom","frustum_top","frustum_near","frustum_far"]
 model=sim.model._model
 result={"camera":fields(ctx.cam,["type","fixedcamid","trackbodyid","lookat","distance","azimuth","elevation"]),
  "viewport":{"left":0,"bottom":0,"width":512,"height":512},
  "context":fields(ctx.con,["offWidth","offHeight","offSamples","currentBuffer","windowAvailable","windowSamples","windowStereo","windowDoublebuffer","shadowSize","glInitialized"]),
  "options":fields(ctx.vopt,["label","frame","geomgroup","sitegroup","jointgroup","tendongroup","actuatorgroup","flags"]),
  "scene":fields(scn,["ngeom","nlight","flags","stereo","enabletransform","translate","rotate","scale"]),
  "active_geometries":[fields(scn.geoms[i],gfields) for i in range(scn.ngeom)],
  "active_lights":[fields(scn.lights[i],lfields) for i in range(scn.nlight)],
  "active_cameras":[fields(c,cfields) for c in scn.camera],
  "model_inputs":{k:hashlib.sha256(np.asarray(getattr(model,k)).tobytes()).hexdigest() for k in ["cam_pos","cam_quat","cam_fovy","light_pos","light_dir","light_ambient","light_diffuse","light_specular","geom_rgba","mat_rgba","tex_rgb"] if hasattr(model,k)},
  "unavailable":["historical GPU context contents","driver scheduling and GPU internal precision/state","complete opaque framebuffer contents before render"],
  "capacity_and_pointer_addresses_serialized":False}
 return result
def capture():
 import numpy as np
 import live01_p_run as prior
 from supervisor.feeding_loading.environment import FeedingLoadingEnv
 from supervisor.planning.rgb_lineage import RGBLineage,digest
 from supervisor.planning.m7_telemetry import LiveTelemetry
 import supervisor.planning.m7_telemetry as telemetry_module
 from obs06_aperture import Motion,Observer
 from load03_visibility import install
 protocol=read(OLD/"protocol.json");case=protocol["cases"][0]
 native=OLD/"monitoring/demo_0002"
 rows=[]
 with gzip.open(native/"states.pkl.gz","rb") as f:
  while True:
   try:r=pickle.load(f)
   except EOFError:break
   if r["control_step"]>190:break
   rows.append(r)
 assert [r["control_step"] for r in rows]==list(range(41,191))
 oldio=np.load(native/"policy_io.npz")
 fixed_history=np.load(native/"predictions.npz")["histories"][5].copy()
 log=open(OUT/"processor.log","w")
 proc=subprocess.Popen([str(R/".venv-xvla/bin/python"),str(Path(__file__).resolve()),"processor"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,env={**os.environ,"HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"})
 def request(q):pickle.dump(q,proc.stdin,protocol=4);proc.stdin.flush();reply=pickle.load(proc.stdout);assert "error" not in reply,reply;return reply
 assert pickle.load(proc.stdout)["ready"]
 random.seed(case["simulation_seed"])
 env=FeedingLoadingEnv(case["config"]);env.reset(case["simulation_seed"]);install(env,False)
 n=np.load(case["initial_archive"])
 assert env.truth()==case["initial_truth"]
 for k in ["qpos","qvel","mocap_pos","mocap_quat"]:np.testing.assert_array_equal(getattr(env.env.sim.data,k),n[k])
 # Original benchmark: video-capture bundle, initial-PNG bundle, then query/action observations.
 for _ in range(2):
  env.frame("primary");env.frame("secondary")
 def video_capture(e):
  e.frame("primary");e.frame("secondary")
 env.capture_callback=video_capture
 motion=Motion(case["config"],protocol["feature_tracking"]);observer=Observer()
 prefix=[]
 for i,row in enumerate(rows):
  limit(900)
  if i%30==0:
   obs=env.observe()
   for cam in ["primary","secondary"]:np.testing.assert_array_equal(obs["rgb"][cam],oldio[cam][i//30])
   evidence=motion.update(obs,env.steps);observer.update(evidence,env.steps)
  env.observe() # Existing per-action observation; requested command comes from archive.
  env.step(row["requested_action"])
  diffs=different(prior.state(env),row["state"])
  prefix.append({"tick":env.steps,"exact_available_state":not diffs,"differences":diffs})
  if diffs:
   dump(OUT/"qualification.json",{"passed":False,"tick":env.steps,"differences":diffs,"prefix":prefix})
   raise AssertionError("Archived available state mismatch")
 assert env.steps==190
 limit(900)
 baseline=prior.state(env)
 with gzip.open(OUT/"qualified-state.pkl.gz","wb") as f:pickle.dump(baseline,f,protocol=4)
 qualification={"passed":True,"label":"replay-reconstructed query5 state","control_tick":190,"query_index":5,"archived_actions_replayed":150,"all_150_available_state_snapshots_exact":True,"prefix_rgb_queries_compared":5,"prefix":prefix,"full_historical_graphics_context_reconstructed":False,"qualified_epoch":time.time(),"gate_seconds":time.time()-read(OUT/"clock.json")["start_epoch"]}
 dump(OUT/"qualification.json",qualification);print(json.dumps({k:v for k,v in qualification.items() if k!="prefix"}),flush=True)
 # From here onward no forward(), step(), reset(), or extra render.
 events=[];lineage=RGBLineage("live01-r-fixed",events.append)
 raw_frame,raw_observe=env.frame,env.observe
 lineage.install(env);p_frame,p_observe=env.frame,env.observe
 env.frame,env.observe=raw_frame,raw_observe
 bundles=[];state_pairs=[];deferred=[];published=[];cached_observer=None;capture_sequence=0
 publisher=LiveTelemetry(OUT/"deferred-publication","LIVE01-R","LOAD04-B","demo_0002")
 publisher._atomic=lambda name,data:published.append((name,data))
 original_imwrite=telemetry_module.imageio.imwrite
 def deferred_encode(uri,frame,**kw):
  assert uri=="<bytes>"
  deferred.append((np.asarray(frame).copy(),kw))
  return b"DEFERRED_IDENTICAL_FOR_N_AND_P"
 telemetry_module.imageio.imwrite=deferred_encode
 order=["N","N","P","P","P","P","N","N"]*3
 try:
  for i,mode in enumerate(order):
   limit()
   env.frame,env.observe=(raw_frame,raw_observe) if mode=="N" else (p_frame,p_observe)
   base_frame=env.frame;render_info=[]
   def measured_frame(camera="primary"):
    nonlocal capture_sequence
    start=time.monotonic_ns();img=base_frame(camera);end=time.monotonic_ns();capture_sequence+=1
    render_info.append({"camera":camera,"audit_capture_id":f"live01-r-attempt02/capture-{capture_sequence:04d}/tick-190","capture_sequence":capture_sequence,"capture_start_ns":start,"capture_end_ns":end,"graphics":graphics(env)});return img
   env.frame=measured_frame
   before=prior.state(env)
   obs=env.observe()
   if cached_observer is None:
    evidence=motion.update(obs,env.steps);pred,rate=observer.update(evidence,env.steps)
    cached_observer={"cached":True,"estimate_control_tick":190,"estimate_bundle":0,"estimate_monotonic_ns":time.monotonic_ns(),"prediction":pred,"aperture_rate":rate,"input_frame_references":{c:{"audit_capture_id":next(x["audit_capture_id"] for x in render_info if x["camera"]==c),"content_sha256":digest(a)} for c,a in obs["rgb"].items()},"temporal_updates_at_tick190":1}
   publisher.metadata_provider=(lambda e,o,p:lineage.publication(o)) if mode=="P" else None
   publisher.publish(env,obs,phase="fixed-query",force=True);assert publisher.last_error is None,publisher.last_error
   req={"op":"prepare","images":[obs["rgb"]["primary"],obs["rgb"]["secondary"]],"instruction":case["instruction"],"audit_index":i,"proprio":fixed_history}
   if mode=="P":req["_rgb_provenance"]=lineage.references(obs)
   prepared=request(req)
   after=prior.state(env)
   diffs=different(before,after)+different(baseline,after)
   state_pairs.append({"bundle":i,"mode":mode,"before":before,"after":after})
   bundle={"bundle":i,"mode":mode,"rgb":{k:v.copy() for k,v in obs["rgb"].items()},"prepared":prepared,"graphics":render_info,"read_only":not diffs,"state_differences":diffs,"lineage":lineage.references(obs) if mode=="P" else None,"observer":cached_observer}
   bundles.append(bundle)
   print(json.dumps({"bundle":i,"mode":mode,"tick":env.steps,"read_only":not diffs,"secondary_pixel_24_417":obs["rgb"]["secondary"][417,24].tolist()}),flush=True)
   if diffs:raise AssertionError("Fixed capture changed physical/controller state")
 finally:
  telemetry_module.imageio.imwrite=original_imwrite
  env.frame,env.observe=raw_frame,raw_observe
  with gzip.open(OUT/"captures.pkl.gz","wb",compresslevel=1) as f:pickle.dump(bundles,f,protocol=4)
  with gzip.open(OUT/"state-pairs.pkl.gz","wb",compresslevel=1) as f:pickle.dump(state_pairs,f,protocol=4)
  dump(OUT/"lineage-events.json",jsonable(events))
  for i,(frame,kw) in enumerate(deferred):original_imwrite(OUT/("publication-%02d.jpg"%i),frame,**{k:v for k,v in kw.items() if k!="format"})
  # No post-capture image/physics generation; publication encoding now uses saved bytes only.
  dump(OUT/"publications.json",[json.loads(data) for name,data in published if name=="latest.json"])
  pickle.dump({"op":"close"},proc.stdin,protocol=4);proc.stdin.flush();proc.wait(timeout=20);log.close()
  env.close()
 dump(OUT/"complete.json",{"capture_bundles":len(bundles),"native_images":2*len(bundles),"physics_steps_after_boundary":0,"model_forward_calls":0,"training":False,"finished_epoch":time.time(),"deferred_image_encoding_common_to_modes":True})
if __name__=="__main__":
 if len(sys.argv)>1 and sys.argv[1]=="processor":worker()
 else:
  try:capture()
  except BaseException:
   dump(OUT/"STOPPED.json",{"epoch":time.time(),"error":traceback.format_exc(),"retry":False});raise
