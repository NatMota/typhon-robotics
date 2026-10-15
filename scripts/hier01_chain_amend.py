"""Two explicit task requests in one retained simulator; no learned sequencing."""
import os,sys,time,json,copy,random,pickle,gzip,hashlib,traceback,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
import numpy as np
from scipy.spatial.transform import Rotation
import imageio.v2 as imageio
import hier01_task as h
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.baseline.benchmark import Recorder
from supervisor.feeding_loading.evaluation import score_episode
from load04_common import array_sha,scores,endpoint
O=Path('/var/tmp/typhon-hier01-chain-amend1-20261015');D=Path('/var/tmp/typhon-hier01-dev-20261015');h.O=O
read,dump,sha=h.read,h.dump,h.sha
def state_blob(env):
 s=h.prior.state(env);d=env.env.sim.data
 s['mocap_pos']=d.mocap_pos.copy();s['mocap_quat']=d.mocap_quat.copy()
 s['contacts']=[{'geom1':int(c.geom1),'geom2':int(c.geom2),'dist':float(c.dist),'pos':np.asarray(c.pos).copy(),'frame':np.asarray(c.frame).copy()} for c in list(d.contact)[:d.ncon]]
 return pickle.dumps(s,protocol=4)
def set_lifetime_budget(env):
 before=state_blob(env);old=copy.deepcopy(env.config)
 env.config['control']['max_steps']=40+2*(1436+24)
 assert state_blob(env)==before
 changed=copy.deepcopy(old);changed['control']['max_steps']=2960;assert env.config==changed
 return {'original_max_steps':old['control']['max_steps'],'workflow_max_steps':2960,'physics_unchanged':True,'per_segment_policy_actions':1436,'per_segment_settle_steps':24}
def request_case(env,source,context,out,first):
 before=state_blob(env);controller=id(env.env.robots[0].controller);sim=id(env.env.sim)
 obs=env.observe();rot=Rotation.from_quat(obs['proprio']['eef_quat_xyzw']).as_matrix()
 prop=np.r_[obs['proprio']['eef_position_m'],rot[:,0],rot[:,1],0.,np.zeros(10)].astype(np.float32)
 dest=out/'initial-live.npz';d=env.env.sim.data
 np.savez_compressed(dest,primary=obs['rgb']['primary'],secondary=obs['rgb']['secondary'],proprio=prop,qpos=d.qpos.copy(),qvel=d.qvel.copy(),mocap_pos=d.mocap_pos.copy(),mocap_quat=d.mocap_quat.copy())
 case=copy.deepcopy(source);case.update(initial_archive=str(dest),initial_archive_sha256=sha(dest),initial_truth=env.truth(),expected_initial_nontext=source['expected_initial_nontext'] if first else {})
 if first:
  with np.load(source['initial_archive']) as a,np.load(dest) as b:
   assert a.files==b.files
   for k in a.files:np.testing.assert_array_equal(a[k],b[k])
  assert case['initial_truth']==source['initial_truth']
 assert state_blob(env)==before and controller==id(env.env.robots[0].controller) and sim==id(env.env.sim)
 dump(out/'request-initialization.json',{'context':context,'current_measured_proprio':True,'first_query_gripper_and_dummy_channels_zero':True,'previous_request_command_history_not_carried':True,'physical_state_unchanged':True,'controller_identity_unchanged':True,'simulator_identity_unchanged':True,'initial_steps':env.steps,'initial_sim_time_s':float(d.time),'initial_state_sha256':hashlib.sha256(before).hexdigest(),'first_stage_matches_DEV_archive':first})
 return case
def forbidden_reset(*args,**kwargs):raise AssertionError('World reset/home motion forbidden after workflow initialization')
def qualify():
 old=read(D/'protocol.json');assert read(D/'next-branch.json')['branch']=='HIER01-CHAIN';assert read(D/'validation_summary.json')['passed']
 clock=read(O/'clock.json');assert time.time()<clock['preparation_deadline']
 hashes={}
 for f,digest in old['source_sha256'].items():
  assert sha(R/f)==digest,f
  hashes[f]=digest
 sources={task:copy.deepcopy(old['cases'][f'demo_0012-{task}-912171']) for task in ['feeding','loading']}
 assert sources['feeding']['config']==sources['loading']['config'] and sources['feeding']['simulation_seed']==sources['loading']['simulation_seed']==912171
 for k in ['GOAL','FL']:
  e=old['models'][k];assert sha(e['checkpoint'])==e['sha256'] and sha(e['probe'])==e['probe_sha256']
 random.seed(912171);env=FeedingLoadingEnv(sources['feeding']['config']);env.reset(912171)
 class NoInferenceWorker:
  def _request(self,q):assert q['op']=='reset';return {'reset':True}
 receipts=[]
 try:
  budget=set_lifetime_budget(env);env.reset=forbidden_reset;env.env.reset=forbidden_reset;env.tick=forbidden_reset
  for i,task in enumerate(['feeding','loading']):
   out=O/'qualification'/task;out.mkdir(parents=True);before=state_blob(env);raw=env.step
   case=request_case(env,sources[task],'qualification-'+task,out,True)
   policy=h.Policy(NoInferenceWorker(),case,out,912171,'qualification-'+task)
   policy.instrument(env,out);policy.reset(912171)
   assert policy.proprio is None and policy.index==0
   policy.telemetry.close();env.step=raw
   assert state_blob(env)==before and env.steps==40
   receipts.append({'task':task,'physical_state_and_controller_unchanged':True,'no_inference':True,'no_physics_step':True})
 finally:env.close()
 hashes['scripts/hier01_chain_amend.py']=sha(R/'scripts/hier01_chain_amend.py')
 p={'experiment':'HIER01-CHAIN-AMEND1','ticket':11069,'frozen_epoch':time.time(),'clock':clock,'parent_commit':'47089a52b59460a8dcb61112cb3aed3c986743a7','models':{k:old['models'][k] for k in ['GOAL','FL']},'routing':old['routing'],'sources':sources,'workflows':[{'id':'L-F','tasks':['loading','feeding']}],'policy_seed':912171,'simulation_seed':912171,'runtime':old['runtime'],'budget_bookkeeping':budget,'source_sha256':hashes,'first_stage_validation':'Exact all48 prepared actor tensor hashes, predictions/history/EEF and applied-command/controller traces, noise and initial/final/stability truth. Raw RGB differences logged separately under explicit amendment.','boundary_rule':'Keep simulator/controller alive; no reset, home, object injection or physics ticks. Fresh policy request starts from measured current EEF pose and clears command history only.','switch_rule':'1436 policy-action steps plus24settle; no truth-dependent switch','second_stage_after_failure':True,'training':False,'observer':False,'TTC':False,'automatic_completion_detector':False,'scope':'Two fixed workflows in one inspected demo0012 context; not broad task composition'}
 assert time.time()<clock['preparation_deadline']
 dump(O/'protocol.json',p);dump(O/'qualification.json',{'passed':True,'receipts':receipts,'budget':budget,'new_model_calls':0,'new_physics_rollouts':0,'initial_reset_for_qualification_only':1,'epoch':time.time()})
 print('CHAIN_BOUNDARY_QUALIFIED',flush=True)
def segment(env,w,source,context,out,first,previous_stability):
 out.mkdir(parents=True);started=time.perf_counter();before=state_blob(env)
 case=request_case(env,source,context,out,first)
 assert state_blob(env)==before
 policy=h.Policy(w,case,out,912171,context);initial=env.truth();start_step=env.steps
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
 episode={'scenario_id':context,'group':'fixed-continuous-workflow-development','task':requested,'seed':912171,'method':'pretrained','uses_simulator_truth_for_control':False,'initial_truth':initial,'final_truth':final,'stability_truths':stability,'attempts':rec.observed_attempts,'evaluation':result,'success':result['success'],'controller_exceptions':[],'artifact_errors':[],'evaluation_errors':[],'policy_actions_executed':n,'wall_duration_s':time.perf_counter()-started,'inference_latency_s':latencies,'native_config':source['config'],'initial_endpoint_assessment':initial_assessment,'initial_pose_satisfied':initial_pose,'interpretation':'preservation' if initial_pose else 'new manipulation requested','control_step_start':start_step,'control_step_end':env.steps}
 dump(out/'episode.json',episode)
 dump(out/'scenario.json',{'scenario_id':context,'task':'loading','requested_task':requested,'config':source['config'],'seed':912171,'continuous_state':not first})
 if first:
  ref=D/'workers'/f'demo_0012-{requested}-912171-router'/'episode';old=read(ref/'episode.json')
  assert initial==old['initial_truth'] and final==old['final_truth'] and stability==old['stability_truths']
  raw_differences={}
  for fn in ['predictions.npz','policy_io.npz','input-audit/initial-prepared.npz']:
   with np.load(out/fn) as a,np.load(ref/fn) as b:
    assert a.files==b.files
    for k in a.files:
     if fn=='policy_io.npz' and k in ['primary','secondary']:
      neq=a[k]!=b[k];count=int(neq.sum())
      raw_differences[k]={'differing_channel_values':count,'total_channel_values':int(a[k].size),'max_abs':float(np.abs(a[k].astype(float)-b[k].astype(float)).max()) if count else 0.,'indices':np.argwhere(neq).tolist()}
     else:np.testing.assert_array_equal(a[k],b[k])
  for f in sorted((out/'input-audit').glob('input-*.json')):
   current=read(f);archived=read(ref/'input-audit'/f.name)
   assert current['input_hashes']==archived['input_hashes']
   assert current['noise_hashes']==archived['noise_hashes'] and current['instruction']==archived['instruction']
  assert len(list((out/'input-audit').glob('input-*.json')))==48
  for fn in ['controller_trace.jsonl.gz','trajectory.jsonl.gz']:
   with gzip.open(out/fn,'rt') as a,gzip.open(ref/fn,'rt') as b:
    assert [json.loads(x) for x in a]==[json.loads(x) for x in b],fn
  assert (out/'initial-state.pkl').read_bytes()==(ref/'initial-state.pkl').read_bytes()
  dump(out/'raw-render-differences.json',{'raw_differences':raw_differences,'all_actual_prepared_actor_hashes_exact':True,'all_applied_commands_and_physical_controller_traces_exact':True,'no_raw_pixel_control_consumer_outside_qualified_actor_preprocessing':True})
  for a in sorted((out/'input-audit').glob('noise-*.npz')):
   with np.load(a) as x,np.load(ref/'input-audit'/a.name) as y:
    for k in x.files:np.testing.assert_array_equal(x[k],y[k])
  dump(out/'native-first-stage-equivalence.json',{'passed':True,'reference':str(ref),'all_effective_predictions_inputs_noise_and_terminal_truth_exact':True,'raw_RGB_comparison_is_separate':True})
 both=scores(episode,source['config'])
 row={'context':context,'requested_task':requested,'model':h.select_checkpoint(requested),'requested_success':result['success'],'opposite_success':both['loading' if requested=='feeding' else 'feeding']['success'],'position_error_mm':result['position_error_m']*1000,'yaw_error_deg':result['orientation_error_deg'],'tilt_deg':float(np.degrees(result['upright_error_rad'])),'failed_criteria':[k for k,v in result['criteria'].items() if not v],'scores':both,'initial_pose_satisfied':initial_pose,'initial_stability_evidence':bool(initial_assessment['criteria']['stable']),'interpretation':episode['interpretation'],'first_stage_native_exact':first,'state_boundary_unchanged':True,'initial_step':start_step,'final_step':env.steps,'policy_actions':n,'queries':len(chunks),'directory':str(out),'wall_seconds':episode['wall_duration_s'],**h.supplemental(episode,out)}
 dump(out/'segment-summary.json',row)
 return row,stability,physical
def run():
 p=read(O/'protocol.json');assert read(O/'qualification.json')['passed']
 for f,digest in p['source_sha256'].items():assert sha(R/f)==digest,f
 rows=[];workflows=[]
 for workflow in p['workflows']:
  env=None;w=None;previous_stability=[];previous_boundary=None;wr=[]
  try:
   for index,task in enumerate(workflow['tasks']):
    h.limit();context=workflow['id']+'-stage'+str(index+1)+'-'+task;source=p['sources'][task]
    mo=O/'workers'/context;mo.mkdir(parents=True);h.base.OUT=mo
    w=h.base.Worker({'model':p['models'][h.select_checkpoint(task)],'expected_initial_nontext':source['expected_initial_nontext'] if index==0 else {}})
    if index==0:
     random.seed(p['simulation_seed']);env=FeedingLoadingEnv(source['config']);env.reset(p['simulation_seed']);set_lifetime_budget(env)
     env.reset=forbidden_reset;env.env.reset=forbidden_reset;env.tick=forbidden_reset
    else:assert state_blob(env)==previous_boundary,'Physics/controller changed while switching workers'
    row,previous_stability,previous_boundary=segment(env,w,source,context,mo/'episode',index==0,previous_stability)
    row.update(workflow=workflow['id'],stage=index+1)
    rows.append(row);wr.append(row);dump(O/'main-summary.json',rows)
    print(json.dumps({k:row[k] for k in ['context','requested_success','position_error_mm','tilt_deg','failed_criteria','supported_lift_10mm_3steps']}),flush=True)
    w.close();w=None
   workflows.append({'workflow':workflow['id'],'success':all(r['requested_success'] for r in wr),'segments':[r['context'] for r in wr],'same_simulator_retained':True})
  finally:
   if w:w.close()
   if env:env.close()
 assert len(rows)==2
 dump(O/'workflows.json',workflows);dump(O/'main-complete.json',{'completed':True,'workflows':1,'segments':2,'epoch':time.time()})
 print('AMENDED_L_F_COMPLETE',flush=True)
if __name__=='__main__':
 try:
  if sys.argv[1]=='qualify':qualify()
  else:run()
 except BaseException:dump(O/('STOPPED-'+sys.argv[1]+'.json'),{'epoch':time.time(),'error':traceback.format_exc()});raise
