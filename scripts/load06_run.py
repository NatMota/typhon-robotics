"""LOAD06: one same-state, first-chunk pose/gripper component intervention."""
import pathlib,sys,os,json,time,gzip,traceback
import numpy as np
R=pathlib.Path('/cs/student/msc_rai/2026/ndossant/Projects/typhon-experiments/load05r-prefix');sys.path.insert(0,str(R))
import load05_crossover as b
import load05r_state as semantic
OUT=pathlib.Path('/var/tmp/typhon-load06-20261014');b.OUT=OUT;b.R=R
def limit():assert time.time()<b.read(OUT/'clock.json')['execution_deadline'],'execution cap'
def source():
 rows=b.read(b.SRC/'analysis/episodes.json');r=next(x for x in rows if x['model']=='B' and x['visibility']=='visible' and x['scene']=='demo_0012' and x['requested_task']=='loading' and x['seed']==912171)
 p=b.SRC/'evaluation'/'B'/'visible'/'demo_0012'/r['id'];tr=[json.loads(x) for x in gzip.open(p/'controller_trace.jsonl.gz','rt')]
 return {'path':p,'trace':{x['step']:x for x in tr},'io':np.load(p/'policy_io.npz'),'history':np.load(p/'raw_predictions.npz')['histories'],'scenario':b.read(p/'scenario.json'),'episode':b.read(p/'episode.json')}
def check_trace(actual,expected,path):
 b.dump(path,{'actual':actual,'expected':expected});b.compare(actual,expected)
def main():
 limit();assert b.read(OUT/'frozen.json')['passed'];s=source();donors=np.load(OUT/'donor-chunks.npz')
 old_state=b.read(pathlib.Path('/var/tmp/typhon-load05r-20261014/continuations/B-to-B/branch-state/state.json'))
 cc={x['step']:x for x in [json.loads(l) for l in gzip.open('/var/tmp/typhon-load05r2-20261014/continuations/B-to-C/trace.jsonl.gz','rt')]}
 w=None;rows=[]
 try:
  for condition in ['BB','BC','CB','CC']:
   limit();dest=OUT/'continuations'/condition;dest.mkdir(parents=True,exist_ok=False);env=b.FeedingLoadingEnv(s['scenario']['config']);frames=[];branch_started=False
   def capture(e):
    a=e.frame('primary');c=e.frame('secondary')
    if branch_started:frames.append(np.concatenate([a,c],axis=1).copy())
   try:
    env.reset(s['scenario']['seed']);b.install(env,False);assert env.truth()==s['episode']['initial_truth']
    env.capture_callback=capture
    env.frame('primary');env.frame('secondary');env.frame('primary');env.frame('secondary')
    prefix=gzip.open(dest/'prefix-trace.jsonl.gz','wt')
    try:
     for i in range(60):
      if i%30==0:b.eqobs(env.observe(),s['io'],i//30)
      target=s['io']['model_actions'].reshape(-1,20)[i];obs=env.observe();cmd=b.to_action(target,obs,env.episode_config);env.step(cmd)
      d=b.details(env,target);prefix.write(json.dumps(d)+'\n');prefix.flush();check_trace(d,s['trace'][env.steps],dest/'prefix-last-comparison.json')
    finally:prefix.close()
    assert env.steps==100;obs=env.observe();b.eqobs(obs,s['io'],2);truth=env.truth();initial=s['episode']['initial_truth']
    assert not truth['grasp_contact'] and b.tilt(truth['object_quat_wxyz'])<10
    assert np.linalg.norm(np.asarray(truth['object_position_m'])-initial['object_position_m'])<=.002
    assert np.linalg.norm(truth['object_linear_velocity_m_s'])<=.01 and np.linalg.norm(truth['object_angular_velocity_rad_s'])<=.1
    history=s['history'][2].copy();np.savez_compressed(dest/'branch-input.npz',primary=obs['rgb']['primary'],secondary=obs['rgb']['secondary'],history=history)
    state=semantic.record(env,dest/'branch-state',history,'B');semantic.assert_equivalent(old_state,state,dest/'independent-prefix-comparison.json')
    if w is None:w=b.Worker('B')
    b.request(w,obs,history,s,2,dest/'start-prepared',prepare=True,expected=True)
    branch_started=True;records=[];chunks=[];histories=[];stream=gzip.open(dest/'trace.jsonl.gz','wt')
    try:
     for j in range(3):
      limit();obs=env.observe();histories.append(history.copy())
      if condition=='BB':b.eqobs(obs,s['io'],2+j)
      if j==0:chunk=donors[condition].copy()
      else:chunk=b.request(w,obs,history,s,2+j,dest/'input-audit',expected=condition=='BB')['actions']
      chunks.append(chunk.copy());np.savez_compressed(dest/'predictions.npz',chunks=np.asarray(chunks),histories=np.asarray(histories))
      if condition=='BB':
       old=s['io']['model_actions'][2+j];b.dump(dest/('prediction-comparison-%d.json'%j),{'exact':bool(np.array_equal(chunk,old)),'max_abs':float(np.max(np.abs(chunk-old)))});np.testing.assert_array_equal(chunk,old)
      history[:9]=chunk[-1,:9]
      for target in chunk:
       obs=env.observe();cmd=b.to_action(target,obs,env.episode_config);env.step(cmd);d=b.rich(env,target,cmd)
       stream.write(json.dumps(d)+'\n');stream.flush();records.append(d)
       if condition=='BB':check_trace(d,s['trace'][env.steps],dest/'continuation-last-comparison.json')
       elif condition=='CC' and j==0:check_trace(d,cc[env.steps],dest/'donor-first-chunk-comparison.json')
      b.dump(dest/'progress.json',{'chunks':j+1,'steps':len(records),'last_control_tick':env.steps})
    finally:stream.close()
    supported=[x for x in records if x['object_truth']['bilateral_grasp_contact'] and x['object_truth']['object_position_m'][2]-initial['object_position_m'][2]>=.01]
    first=records[:30]
    row={'condition':condition,'first_chunk_pose_donor':condition[0],'first_chunk_grip_donor':condition[1],'subsequent_policy':'B','steps':len(records),'first30_max_tilt_deg':max(x['object_tilt_deg'] for x in first),'first30_final_tilt_deg':first[-1]['object_tilt_deg'],'max_object_tilt_deg':max(x['object_tilt_deg'] for x in records),'final_object_tilt_deg':records[-1]['object_tilt_deg'],'first_tilt_over15':next((x['step'] for x in records if x['object_tilt_deg']>15),None),'first_contact':next((x['step'] for x in records if x['object_truth']['grasp_contact']),None),'first_close_command':next((x['step'] for x in records if x['command'][-1]>0),None),'first_bilateral':next((x['step'] for x in records if x['object_truth']['bilateral_grasp_contact']),None),'first_supported_lift':supported[0]['step'] if supported else None,'supported_lift_tilt_deg':supported[0]['object_tilt_deg'] if supported else None,'upright_supported_lift_under10deg':bool(supported and supported[0]['object_tilt_deg']<10),'final_placement_evaluation':False,'training_eligible':False,'qualification':'BB full90 archive parity' if condition=='BB' else ('CC first30 donor parity' if condition=='CC' else 'qualified independent B prefix')}
    b.dump(dest/'summary.json',row);rows.append(row);b.dump(OUT/'summary.json',rows);print(json.dumps(row),flush=True)
   finally:
    env.close()
    if frames:
     import imageio.v2 as imageio
     imageio.mimwrite(str(dest/'rollout.mp4'),frames,fps=5,codec='libx264')
   limit()
  b.dump(OUT/'complete.json',{'passed':True,'continuations':len(rows),'new_queries':8,'first_chunks_generated':0,'finished':time.time(),'training_eligible':False})
 finally:
  if w is not None:w.close()
if __name__=='__main__':
 try:main()
 except BaseException:b.dump(OUT/'STOPPED.json',{'time':time.time(),'error':traceback.format_exc(),'outcome_retries':0});raise
