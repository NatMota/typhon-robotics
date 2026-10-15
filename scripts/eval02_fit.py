"""Twelve same-observation action proposals; no simulator execution or fitting."""
import os,sys,json,time,hashlib,subprocess,traceback,copy,csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
O=Path(os.environ.get('TYPHON_FIT_ROOT','/var/tmp/typhon-eval02-fit-20261015'))
RET=Path('/var/tmp/typhon-eval02-retention-20261015')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def limit():assert time.time()<read(O/'clock.json')['execution_deadline'],'Fixed proposal block deadline'
def matrices(a):
 b1=a[:,3:6].astype(float);n=np.linalg.norm(b1,axis=1,keepdims=True);assert (n>1e-8).all();b1=b1/n
 b2=a[:,6:9]-np.sum(b1*a[:,6:9],axis=1,keepdims=True)*b1;n=np.linalg.norm(b2,axis=1,keepdims=True);assert (n>1e-8).all();b2=b2/n
 return np.stack([b1,b2,np.cross(b1,b2)],axis=2)
def metrics(pred,target):
 assert pred.shape==target.shape==(30,20) and np.isfinite(pred).all()
 e=np.linalg.norm(pred[:,:3]-target[:,:3],axis=1)*1000
 r=np.degrees(Rotation.from_matrix(matrices(pred)@matrices(target).transpose(0,2,1)).magnitude())
 a=pred[:,9]>.5;b=target[:,9]>.5
 changes=lambda x:(np.flatnonzero(x[1:]!=x[:-1])+1).tolist()
 return {'xyz_mean_l2_mm':float(e.mean()),'xyz_rms_l2_mm':float(np.sqrt(np.mean(e*e))),'xyz_final_error_mm':float(e[-1]),'rotation_mean_geodesic_deg':float(r.mean()),'rotation_final_geodesic_deg':float(r[-1]),'gripper_correct_steps':int(np.sum(a==b)),'steps':30,'expert_gripper_changes':changes(b),'predicted_gripper_changes':changes(a),'expert_closed_steps':int(b.sum()),'predicted_closed_steps':int(a.sum()),'predicted_final_eef_xyz_m':pred[-1,:3].tolist(),'expert_final_eef_xyz_m':target[-1,:3].tolist(),'xyz_error_per_step_mm':e.tolist(),'rotation_error_per_step_deg':r.tolist(),'predicted_close_probability':pred[:,9].tolist(),'expert_close_state':b.astype(int).tolist()}
def prepare():
 from PIL import Image,ImageDraw
 p=read(RET/'protocol.json');c=p['cases']['G1'];root=Path('/var/tmp/typhon-ost-sweep-20261012/goal-grounding/data')
 e=next(x for x in read(root/'dataset-v2.json')['episodes'] if x['id']==c['source_episode_id']);assert e['split']=='train'
 archive=root/e['path'];assert sha(archive)==e['sha256'];src=Path(e['source_path'])
 ep=read(src/'episode.json');assert sha(src/'episode.json')==e['source_episode_sha256'] and ep['success']
 actions=[json.loads(x) for x in (src/'robot_actions.jsonl').read_text().splitlines()];assert sha(src/'robot_actions.jsonl')==e['source_action_sha256']
 verified=next(a['lift_truth'] for a in ep['attempts'] if a['verified_lift'] and a['lift_truth']['bilateral_grasp_contact'])
 assert verified['object_position_m'][2]-ep['initial_truth']['object_position_m'][2]>.07
 z=np.load(archive);indices=z['frame_action_indices'];reset=ep['reset_warmup_steps'];assert reset==40
 assert np.array_equal(indices,np.arange(0,len(actions),4))
 ix=next(i for i,a in enumerate(indices) if reset+int(a)>=verified['control_steps'] and int(a)+30<=len(actions))
 packetframes={'approach':0,'postlift':ix};packets={}
 canvas=Image.new('RGB',(1024,2*544),(16,18,23));draw=ImageDraw.Draw(canvas)
 for j,(name,frame) in enumerate(packetframes.items()):
  start=int(indices[frame]);assert start+30<=len(actions)
  assert actions[start]['control_step']==reset+start+1
  pre=ep['initial_truth'] if start==0 else actions[start-1]['post_action_truth']
  if name=='postlift':
   assert pre['control_steps']>=verified['control_steps'] and pre['bilateral_grasp_contact']
   assert pre['object_position_m'][2]-ep['initial_truth']['object_position_m'][2]>.07
  target=z['targets'][start:start+30].copy();history=z['proprio'][frame].copy()
  if start:np.testing.assert_array_equal(history[:9],z['targets'][start-1,:9])
  assert (history[9:]==0).all()
  f=O/(name+'.npz');np.savez_compressed(f,primary=z['primary'][frame],secondary=z['secondary'][frame],proprio=history,targets=target)
  packets[name]={'archive':str(f),'sha256':sha(f),'frame_index':frame,'action_index':start,'pre_action_sim_step':reset+start,'complete_targets':30,'instruction':e['instruction'],'source_episode_id':e['id'],'source_archive':str(archive),'source_archive_sha256':e['sha256'],'pre_action_truth_evaluator_only':pre,'verified_lift_step':verified['control_steps'],'history_contract':'first9 last commanded target except initial measured robot pose; gripper and dummy channels remain zero, native training/client contract'}
  im=np.concatenate([z['primary'][frame],z['secondary'][frame]],axis=1);canvas.paste(Image.fromarray(im),(0,j*544+32));draw.text((5,j*544+8),f'{name} | frame {frame} | action {start} | pre-action tick {reset+start}',fill='white')
 canvas.save(O/'teacher-packets.png')
 targets=z['targets'][0:30];zero=metrics(targets,targets);assert zero['xyz_rms_l2_mm']==0 and zero['rotation_mean_geodesic_deg']<1e-5 and zero['gripper_correct_steps']==30
 shifted=targets.copy();shifted[:,0]+=.01;check=metrics(shifted,targets);assert abs(check['xyz_mean_l2_mm']-10)<.001
 flipped=targets.copy();flipped[:,9]=1-targets[:,9];assert metrics(flipped,targets)['gripper_correct_steps']==0
 script_hash=sha(Path(__file__))
 protocol={'experiment':'EVAL02-FIT','ticket':11063,'frozen_epoch':time.time(),'clock':read(O/'clock.json'),'models':p['models'],'model_order':p['model_order'],'packets':packets,'packet_order':['approach','postlift'],'policy_seeds':[912171,912172],'script_sha256':script_hash,'worker_sha256':sha(R/'scripts/load04_worker.py'),'rotation_contract':'native contiguous first-two columns after worker reorder inverse','action_horizon':30,'control_hz':20,'experimental_proposals':12,'new_physical_rollouts':0,'new_training':False,'holdouts':False,'source_provenance':{'training_manifest_sha256':sha(root/'dataset-v2.json'),'source_episode_sha256':sha(src/'episode.json'),'source_actions_sha256':sha(src/'robot_actions.jsonl'),'retention_protocol_sha256':sha(RET/'protocol.json')},'metric_checks':{'identity_pass':True,'10mm_translation_pass':True,'gripper_flip_pass':True},'interpretation':'Conditional action agreement on two archived teacher states. Not task success or a general fitting evaluation.'}
 dump(O/'protocol.json',protocol);print(json.dumps({'prepared':True,'packets':{k:{x:v[x] for x in ['frame_index','action_index','pre_action_sim_step','verified_lift_step']} for k,v in packets.items()}}),flush=True)
def run():
 import eval02_goal_run as base
 p=read(O/'protocol.json');assert sha(Path(__file__))==p['script_sha256'];assert sha(R/'scripts/load04_worker.py')==p['worker_sha256']
 rows=[];inputs={};noises={}
 for model in p['model_order']:
  limit();d=O/model;d.mkdir();base.OUT=d;w=base.Worker({'model':p['models'][model],'expected_initial_nontext':{}})
  try:
   for packet in p['packet_order']:
    meta=p['packets'][packet];assert sha(meta['archive'])==meta['sha256'];z=np.load(meta['archive'])
    for seed in p['policy_seeds']:
     limit();audit=d/(packet+'-'+str(seed));assert not audit.exists()
     w._request({'op':'reset','seed':seed})
     result=w._request({'op':'predict','images':[z['primary'],z['secondary']],'proprio':z['proprio'],'instruction':meta['instruction'],'audit_index':0,'audit_directory':str(audit)})
     pred=result.pop('actions');np.savez_compressed(audit/'proposal.npz',predicted=pred,expert=z['targets'])
     with np.load(audit/'initial-prepared.npz') as a:
      if packet in inputs:
       assert set(a.files)==set(inputs[packet])
       for k in a.files:np.testing.assert_array_equal(a[k],inputs[packet][k])
      else:inputs[packet]={k:a[k].copy() for k in a.files}
     with np.load(audit/'noise-000.npz') as a:
      if seed in noises:
       assert set(a.files)==set(noises[seed])
       for k in a.files:np.testing.assert_array_equal(a[k],noises[seed][k])
      else:noises[seed]={k:a[k].copy() for k in a.files}
     met=metrics(pred,z['targets']);row={'model':model,'packet':packet,'seed':seed,'directory':str(audit),'inference':result,**met};rows.append(row);dump(audit/'metrics.json',row);dump(O/'summary.json',rows)
     print(json.dumps({k:row[k] for k in ['model','packet','seed','xyz_mean_l2_mm','rotation_mean_geodesic_deg','gripper_correct_steps']}),flush=True)
  finally:w.close()
 assert len(rows)==12
 keys=['model','packet','seed','xyz_mean_l2_mm','xyz_rms_l2_mm','xyz_final_error_mm','rotation_mean_geodesic_deg','rotation_final_geodesic_deg','gripper_correct_steps','expert_closed_steps','predicted_closed_steps']
 with (O/'comparisons.csv').open('w',newline='') as f:
  out=csv.DictWriter(f,fieldnames=keys);out.writeheader();out.writerows({k:r[k] for k in keys} for r in rows)
 dump(O/'validation_summary.json',{'passed':True,'experimental_proposals':12,'fresh_checkpoint_probes':3,'physical_rollouts':0,'all_packet_inputs_including_text_exact_across_models_and_seeds':True,'paired_actual_noise_arrays_exact':True,'teacher_target_windows_complete_no_padding':True,'all_metrics_finite':True,'metric_self_checks':p['metric_checks'],'finished_epoch':time.time()})
 dump(O/'complete.json',{'completed':True,'proposals':12,'physical_rollouts':0,'epoch':time.time(),'elapsed_seconds':time.time()-p['clock']['start_epoch']})
if __name__=='__main__':
 try:{'prepare':prepare,'run':run}[sys.argv[1]]()
 except BaseException:dump(O/('STOPPED-'+sys.argv[1]+'.json'),{'error':traceback.format_exc(),'epoch':time.time()});raise
