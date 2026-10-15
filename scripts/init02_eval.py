"""INIT02 fixed matched native evaluations; no action intervention or outcome retry."""
import os,sys,time,json,copy,random,pickle,gzip,hashlib,traceback,subprocess
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
import numpy as np
import imageio.v2 as imageio
import hier01_task as h
import hier01_chain_amend as chain
import init01 as initial
from init02_common import OUT,INIT,read,dump,sha
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.baseline.benchmark import Recorder
from supervisor.feeding_loading.evaluation import score_episode
from load04_common import array_sha,scores,endpoint
state_blob=chain.state_blob
request_case=chain.request_case
h.O=OUT;chain.O=OUT
def segment(env,w,source,context,out,first,previous_stability,model_name):
 out.mkdir(parents=True);started=time.perf_counter();before=state_blob(env)
 case=request_case(env,source,context,out,first)
 assert state_blob(env)==before
 policy=h.Policy(w,case,out,source["policy_seed"],context);initial=env.truth();start_step=env.steps
 original=(env.step,env.env.step,env.move_to,env.capture_callback);controller=id(env.env.robots[0].controller);sim=id(env.env.sim)
 writer=None;rec=None;inputs=[];chunks=[];stability=[];latencies=[];exceptions=[];n=0
 try:
  policy.instrument(env,out);rec=Recorder(env,out/'trajectory.jsonl.gz');policy.reset(case['simulation_seed'])
  assert state_blob(env)==before and policy.proprio is None
  (out/'boundary-before.pkl').write_bytes(before);(out/'boundary-after-request.pkl').write_bytes(state_blob(env))
  writer=imageio.get_writer(out/'rollout.mp4',fps=5)
  def capture(scene):writer.append_data(np.concatenate([scene.frame('primary'),scene.frame('secondary')],axis=1))
  capture(env);env.capture_callback=capture
  imageio.imwrite(out/'initial.png',np.concatenate([env.frame(),env.frame('secondary')],axis=1))
  deadline=start_step+1436
  while env.steps<deadline:
   h.limit();obs=env.observe();assert set(obs)<= {'rgb','proprio','task','cameras'}
   inputs.append({'primary':obs['rgb']['primary'].copy(),'secondary':obs['rgb']['secondary'].copy(),'eef_position_m':np.asarray(obs['proprio']['eef_position_m']).copy(),'eef_quat_xyzw':np.asarray(obs['proprio']['eef_quat_xyzw']).copy()})
   t=time.perf_counter();chunk=np.asarray(policy.predict(obs));latencies.append(time.perf_counter()-t);chunks.append(chunk.copy())
   assert chunk.ndim==2 and len(chunk) and np.all(np.isfinite(chunk))
   for target in chunk:
    if env.steps>=deadline:break
    action=policy.to_action(target,env.observe(),source['config']);env.step(action);n+=1
  for _ in range(12):
   for _ in range(2):env.step(np.r_[np.zeros(6),env.last_grip])
   stability.append(env.truth())
  assert n==1436 and env.steps-start_step==1460 and len(chunks)==48
 except Exception:
  exceptions.append(traceback.format_exc());raise
 finally:
  if rec:rec.finish()
  final=env.truth()
  if writer:
   rgb=np.concatenate([env.frame(),env.frame('secondary')],axis=1);imageio.imwrite(out/'final.png',rgb);writer.append_data(rgb);writer.close()
  if hasattr(policy,'telemetry') and not policy.telemetry.closed:policy.telemetry.close()
  physical=state_blob(env)
  env.step,env.env.step,env.move_to,env.capture_callback=original
  assert state_blob(env)==physical and id(env.env.robots[0].controller)==controller and id(env.env.sim)==sim
 if exceptions:raise RuntimeError(exceptions)
 np.savez_compressed(out/'policy_io.npz',model_actions=np.asarray(chunks),**{k:np.asarray([x[k] for x in inputs]) for k in inputs[0]})
 requested=source['requested_task'];ec=endpoint(source['config'],requested)
 result=score_episode(initial,final,stability,ec,rec.observed_attempts)
 initial_assessment=score_episode(initial,initial,previous_stability,ec,[])
 initial_pose=all(v for k,v in initial_assessment['criteria'].items() if k!='stable')
 episode={'scenario_id':context,'group':'fixed-continuous-workflow-development','task':requested,'seed':source['policy_seed'],'method':'pretrained','uses_simulator_truth_for_control':False,'initial_truth':initial,'final_truth':final,'stability_truths':stability,'attempts':rec.observed_attempts,'evaluation':result,'success':result['success'],'controller_exceptions':[],'artifact_errors':[],'evaluation_errors':[],'policy_actions_executed':n,'wall_duration_s':time.perf_counter()-started,'inference_latency_s':latencies,'native_config':source['config'],'initial_endpoint_assessment':initial_assessment,'initial_pose_satisfied':initial_pose,'interpretation':'preservation' if initial_pose else 'new manipulation requested','control_step_start':start_step,'control_step_end':env.steps}
 episode.update(experiment='INIT02',panel=source['panel'],model=model_name,model_sha256=model_entry(model_name)['sha256'],policy_seed=source['policy_seed'],simulation_seed=source['simulation_seed'],harness=False,TTC=False)
 dump(out/'episode.json',episode)
 dump(out/'scenario.json',{'scenario_id':context,'task':source['config']['task']['type'],'requested_task':requested,'config':source['config'],'seed':source['policy_seed'],'continuous_state':not first})
 both=scores(episode,source['config']) if source['config']['task']['type']=='loading' else {requested:result}
 row={'context':context,'requested_task':requested,'model':model_name,'requested_success':result['success'],'opposite_success':both.get('loading' if requested=='feeding' else 'feeding',{}).get('success'),'position_error_mm':result['position_error_m']*1000,'yaw_error_deg':result['orientation_error_deg'],'tilt_deg':float(np.degrees(result['upright_error_rad'])),'failed_criteria':[k for k,v in result['criteria'].items() if not v],'scores':both,'initial_pose_satisfied':initial_pose,'initial_stability_evidence':bool(initial_assessment['criteria']['stable']),'interpretation':episode['interpretation'],'first_stage_native_exact':first,'state_boundary_unchanged':True,'initial_step':start_step,'final_step':env.steps,'policy_actions':n,'queries':len(chunks),'directory':str(out),'wall_seconds':episode['wall_duration_s'],**h.supplemental(episode,out)}
 dump(out/'segment-summary.json',row)
 return row,stability,physical

def model_entry(name):
 if name=="P":return read(OUT/"protocol.json")["model"]
 d=OUT/("train-"+name);c=read(d/"complete.json")
 return {"checkpoint":str(d/"step3000.pt"),"probe":str(d/"step3000-probe.pt"),"sha256":c["checkpoint_sha256"],"probe_sha256":c["probe_sha256"]}
def initialise(source,folder):
 if source["kind"]=="prefix":
  p=next(x for x in read(INIT/"protocol.json")["parents"] if x["id"]==source["parent"])
  env,receipt,previous=initial.replay(p,folder,False)
 else:
  folder.mkdir(parents=True,exist_ok=False);random.seed(source["simulation_seed"])
  env=FeedingLoadingEnv(copy.deepcopy(source["config"]));env.reset(source["simulation_seed"]);previous=[]
  assert initial.numeric(env.truth(),source["initial_truth"])<1e-8
  assert state_blob(env)==(Path(source["initial_archive"]).parent/"initial-state.pkl").read_bytes()
 assert initial.numeric(env.truth(),source["initial_truth"])<1e-8
 with np.load(source["initial_archive"]) as a:
  obs=env.observe()
  for k in ["qpos","qvel","mocap_pos","mocap_quat"]:np.testing.assert_array_equal(getattr(env.env.sim.data,k),a[k])
  for k in ["primary","secondary"]:np.testing.assert_array_equal(obs["rgb"][k],a[k])
 before=state_blob(env)
 env.config=copy.deepcopy(endpoint(source["config"],source["requested_task"]))
 env.config["control"]["max_steps"]=env.steps+1460
 env.episode_config=copy.deepcopy(env.config)
 assert state_blob(env)==before
 env.reset=initial.forbidden;env.env.reset=initial.forbidden
 return env,previous
def native_reference(row,source,name,out):
 if name!="P" or source["kind"]!="prefix" or source["requested_task"]!="loading" or source["policy_seed"]!=912171:return
 ref=INIT/"continuations"/source["parent"]/"L-policy/episode";old=read(ref/"episode.json");ep=read(out/"episode.json")
 assert ep["initial_truth"]==old["initial_truth"] and ep["final_truth"]==old["final_truth"] and ep["stability_truths"]==old["stability_truths"]
 for fn in ["predictions.npz","input-audit/initial-prepared.npz"]:
  with np.load(out/fn) as a,np.load(ref/fn) as b:
   assert a.files==b.files
   for k in a.files:np.testing.assert_array_equal(a[k],b[k])
 for f in sorted((out/"input-audit").glob("input-*.json")):
  a=read(f);b=read(ref/"input-audit"/f.name)
  assert a["input_hashes"]==b["input_hashes"] and a["noise_hashes"]==b["noise_hashes"]
 for fn in ["controller_trace.jsonl.gz","trajectory.jsonl.gz"]:
  with gzip.open(out/fn,"rt") as a,gzip.open(ref/fn,"rt") as b:assert [json.loads(x) for x in a]==[json.loads(x) for x in b],fn
 dump(out/"INIT01-native-reference.json",{"passed":True,"source":str(ref),"all_48_actor_inputs_predictions_commands_and_truth_exact":True})
def extra_summary(row,source,name,out):
 ep=read(out/"episode.json")
 with gzip.open(out/"trajectory.jsonl.gz","rt") as f:states=[json.loads(x)["truth"] for x in f]
 obj=np.asarray([s["object_position_m"] for s in states]);first=np.asarray(ep["initial_truth"]["object_position_m"])
 maxdisp=float(np.linalg.norm(obj-first,axis=1).max());contact=any(s["grasp_contact"] for s in states)
 with gzip.open(out/"controller_trace.jsonl.gz","rt") as f:traces=[json.loads(x) for x in f]
 any_geometry_contact=any(any("cube" in str(c.get("geom1",""))+str(c.get("geom2","")) and "gripper" in str(c.get("geom1",""))+str(c.get("geom2","")) for c in t["contacts"]) for t in traces)
 no_contact=not(contact or any_geometry_contact)
 preserved=bool(row["requested_success"] and no_contact and maxdisp<=.001)
 row.update(id=source["id"],panel=source["panel"],model=name,policy_seed=source["policy_seed"],model_sha256=model_entry(name)["sha256"],
  maximum_object_displacement_mm=maxdisp*1000,any_gripper_object_contact=not no_contact,preservation_qualified=preserved if source["panel"]=="preservation" else None,
  close_target_count=sum(t["target"][9]>.5 for t in traces),bilateral_contact_steps=sum(s["bilateral_grasp_contact"] for s in states),
  wrist_max_displacement_mm=float(np.linalg.norm(np.asarray([s["eef_position_m"] for s in states])-ep["initial_truth"]["eef_position_m"],axis=1).max()*1000),
  minimum_joint_limit_margin_rad=min(t["nearest_joint_limit_rad"] for t in traces))
 npz=np.load(out/"input-audit/initial-prepared.npz");initial_hash={k:array_sha(npz[k]) for k in npz.files}
 if name!="P":
  ref=OUT/"episodes"/"P"/source["id"]
  old=np.load(ref/"input-audit/initial-prepared.npz");assert {k:array_sha(old[k]) for k in old.files}==initial_hash
  h.base.noise_gate(out,source["policy_seed"],ref)
 native_reference(row,source,name,out)
 dump(out/"summary.json",row);return row
def run(name,lane):
 assert name in ["P","R","T"] and lane in [0,1]
 seal=read(OUT/"execution-freeze.json");assert seal["private_push_verified"]
 for f,digest in seal["source_sha256"].items():assert sha(R/f)==digest,f
 assert sha(OUT/"evaluation-cases.json")==read(OUT/"data/launch-gate.json")["evaluation_sha256"]
 cases=read(OUT/"evaluation-cases.json")["core"];subset=cases[lane::2]
 p=read(OUT/"protocol.json");mo=OUT/"workers"/(name+"-lane"+str(lane));mo.mkdir(parents=True,exist_ok=False)
 h.base.OUT=mo;h.base.native.OUT=OUT;h.O=OUT;chain.O=OUT
 model=model_entry(name);assert sha(model["checkpoint"])==model["sha256"] and sha(model["probe"])==model["probe_sha256"]
 w=h.base.Worker({"model":model,"expected_initial_nontext":{}});rows=[];env=None
 try:
  for source in subset:
   h.limit();context=source["id"];dest=OUT/"episodes"/name/context
   assert not dest.exists(),"No outcome retry"
   env,previous=initialise(source,OUT/"replayed-prefixes"/name/context)
   w.protocol["expected_initial_nontext"]=source["expected_initial_nontext"]
   row,_,_=segment(env,w,source,context,dest,False,previous,name)
   row=extra_summary(row,source,name,dest);rows.append(row);dump(mo/"summary.json",rows)
   print(json.dumps({k:row[k] for k in ["id","model","requested_success","preservation_qualified","position_error_mm","supported_lift_10mm_3steps"]}),flush=True)
   env.close();env=None
 finally:
  if env:env.close()
  w.close()
 dump(mo/"complete.json",{"completed":True,"episodes":len(rows),"epoch":time.time()})
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--model",choices=["P","R","T"],required=True);p.add_argument("--lane",type=int,choices=[0,1],required=True);a=p.parse_args()
 try:run(a.model,a.lane)
 except BaseException:dump(OUT/("STOPPED-eval-"+a.model+"-"+str(a.lane)+".json"),{"epoch":time.time(),"error":traceback.format_exc()});raise
