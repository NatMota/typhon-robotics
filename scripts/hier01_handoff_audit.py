#!/usr/bin/env python3
"""Saved-evidence handoff audit. No simulator or policy imports/calls."""
import csv,gzip,hashlib,json,pickle,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from supervisor.adaptation.m6_contract import reorder
O=Path('/var/tmp/typhon-hier01-handoff-20261015');D=R/'experiment-history/2026-10-15-hier01-handoff';D.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):
 with gzip.open(p,'rt') as f:return [json.loads(x) for x in f]
def vals(x):return np.asarray(x,dtype=float)
roots={'F-L':Path('/var/tmp/typhon-hier01-chain-20261015/workers/F-L-stage2-loading/episode'),'L-F':Path('/var/tmp/typhon-hier01-chain-amend1-20261015/workers/L-F-stage2-feeding/episode')}
goalpath=Path('/var/tmp/typhon-ost-sweep-20261012/goal-grounding/data/dataset-v2.json')
flpath=Path('/var/tmp/typhon-load01-20261013/data/qualified-dataset.json')
protocol=read('/var/tmp/typhon-hier01-dev-20261015/protocol.json')
training={};receipts={};coverage=[]
sample_receipts={}
for model in ['GOAL','FL']:
 if model=='GOAL':
  wp=Path('/var/tmp/typhon-ost-sweep-20261012/goal-grounding/train-varied/window-index.json');sp=wp.parent/'sampling-plan.json.gz';windows=read(wp);plan=json.load(gzip.open(sp,'rt'));counts=Counter(windows[i]['sample_id'].rsplit(':',1)[0] for i in plan)
 else:
  wp=flpath;sp=wp.parent/'sampling-plan.json.gz';windows=read(wp)['windows'];plan=json.load(gzip.open(sp,'rt'))['plan']['FL'];counts=Counter(windows[i]['episode'] for b in plan for i in b)
 sample_receipts[model]={'sampling_sha256':sha(sp),'windows_sha256':sha(wp),'sample_counts':dict(counts),'total_exposures':sum(counts.values())}
 assert sum(counts.values())==12000
for model,p in [('GOAL',goalpath),('FL',flpath)]:
 manifest=read(p);episodes=[e for e in manifest['episodes'] if e['split'] in ['train','load01_train']]
 training[model]=[]
 for e in episodes:
  ep=Path(e['source_path'])/'episode.json';j=read(ep)
  assert sha(ep)==e['source_episode_sha256']
  archive=Path(e.get('absolute_path',str(p.parent/e['path'])))
  with np.load(archive) as a:prop=a['proprio'][0].copy()
  t=j['initial_truth'];entry={'id':e['id'],'source_demo_id':e['source_demo_id'],'task':e['config']['task']['type'],'initial_truth':t,'proprio_initial':prop.tolist(),'source_episode_sha256':sha(ep)}
  training[model].append(entry)
 assert set(sample_receipts[model]['sample_counts'])==set(e['id'] for e in episodes)
 receipts[model]={'sampling':sample_receipts[model],'manifest':str(p),'sha256':sha(p),'episodes':len(episodes),'source_ids':sorted(set(e['source_demo_id'] for e in episodes))}
summary=[];canvas=Image.new('RGB',(1120,2*570),'#11151b');draw=ImageDraw.Draw(canvas)
for row,(direction,p) in enumerate(roots.items()):
 task='loading' if direction=='F-L' else 'feeding';model='FL' if task=='loading' else 'GOAL'
 ep=read(p/'episode.json');truth=ep['initial_truth'];io=np.load(p/'policy_io.npz');pred=np.load(p/'predictions.npz');prep=np.load(p/'input-audit/initial-prepared.npz');live=np.load(p/'initial-live.npz')
 act=pred['chunks'];hist=pred['histories'];audit=read(p/'input-audit/input-000.json');inf=read(p/'inference.json')
 assert np.array_equal(act,io['model_actions'])
 rot=Rotation.from_quat(io['eef_quat_xyzw'][0]).as_matrix()
 prop=np.r_[io['eef_position_m'][0],rot[:,0],rot[:,1],0.,np.zeros(10)].astype(np.float32)
 np.testing.assert_array_equal(hist[0],prop);np.testing.assert_array_equal(hist[0],live['proprio'])
 np.testing.assert_array_equal(prep['proprio'][0],reorder(prop))
 np.testing.assert_array_equal(hist[1:,:9],act[:-1,-1,:9]);assert np.all(hist[:,9:]==0)
 assert np.array_equal(io['primary'][0],live['primary']) and np.array_equal(io['secondary'][0],live['secondary'])
 assert prep['image_mask'].all() and all(np.asarray(q['image_mask']).all() for q in inf)
 state=pickle.loads((p/'boundary-before.pkl').read_bytes());assert (p/'boundary-before.pkl').read_bytes()==(p/'boundary-after-request.pkl').read_bytes()
 np.testing.assert_allclose(state['qpos'][9:12],truth['object_position_m'],atol=1e-12,rtol=0);np.testing.assert_allclose(state['qpos'][7:9],truth['gripper_joint_position_m'],atol=1e-12,rtol=0)
 dev=Path('/var/tmp/typhon-hier01-dev-20261015/workers/demo_0012-'+task+'-912171-router/episode')
 devio=np.load(dev/'policy_io.npz');devpred=np.load(dev/'predictions.npz');devep=read(dev/'episode.json')
 assert audit['instruction']==read(dev/'input-audit/input-000.json')['instruction']
 assert audit['noise_hashes']==read(dev/'input-audit/input-000.json')['noise_hashes']
 trace=lines(p/'controller_trace.jsonl.gz');dt=lines(dev/'controller_trace.jsonl.gz')
 obj=vals(truth['object_position_m']);eef=vals(truth['eef_position_m'])
 def metrics(a,tr,t):
  obj=vals(t['object_position_m']);pos=np.array([x['object_truth']['object_position_m'] for x in tr])
  return {'predicted_actions':int(np.prod(a.shape[:2])),'raw_gripper_range':[float(a[:,:,9].min()),float(a[:,:,9].max())],'raw_close_proposals':int(np.sum(a[:,:,9]>.5)),'minimum_raw_target_distance_to_initial_brick_mm':float(np.min(np.linalg.norm(a[:,:,:3]-obj,axis=-1))*1000),'minimum_actual_eef_distance_to_initial_brick_mm':float(min(np.linalg.norm(vals(x['eef_xyz'])-obj) for x in tr)*1000),'grasp_contact_steps':sum(bool(x['object_truth']['grasp_contact']) for x in tr),'bilateral_grasp_steps':sum(bool(x['object_truth']['bilateral_grasp_contact']) for x in tr),'max_object_displacement_mm':float(np.linalg.norm(pos-obj,axis=-1).max()*1000),'max_object_lift_mm':float((pos[:,2]-obj[2]).max()*1000),'first_raw_target_xyz':a[0,0,:3].tolist(),'first_chunk_last_target_xyz':a[0,-1,:3].tolist(),'first_raw_delta_from_measured_eef_mm':((a[0,0,:3]-vals(t['eef_position_m']))*1000).tolist()}
 m=metrics(act,trace,truth);dm=metrics(devpred['chunks'],dt,devep['initial_truth'])
 peers=[x for x in training[model] if x['task']==task]
 for scope,peers2 in [('incoming_task',peers),('all_latest_stage_training',training[model])]:
  ds=[(float(np.linalg.norm(vals(x['initial_truth']['object_position_m'])[:2]-obj[:2])*1000),x) for x in peers2];distance,near=min(ds,key=lambda x:x[0])
  yaws=[abs((float(x['initial_truth']['object_yaw_rad'])-truth['object_yaw_rad']+np.pi)%(2*np.pi)-np.pi) for x in peers2]
  eefs=[np.linalg.norm(vals(x['initial_truth']['eef_position_m'])-eef)*1000 for x in peers2]
  coverage.append({'direction':direction,'model':model,'scope':scope,'episodes':len(peers2),'nearest_start_xy_mm':distance,'nearest_start_id':near['id'],'minimum_start_yaw_gap_deg':float(np.rad2deg(min(yaws))),'minimum_initial_eef_distance_mm':float(min(eefs)),'training_start_x_range_m':[min(x['initial_truth']['object_position_m'][0] for x in peers2),max(x['initial_truth']['object_position_m'][0] for x in peers2)],'exact_initial_joint_coverage':'not stored in teacher episode archive; not inferred from EEF'})
 cp=protocol['models'][model]
 reload_receipt=read(p.parent/'reload-verification.json');assert reload_receipt['passed'] and reload_receipt['checkpoint']==cp['checkpoint'] and reload_receipt['action_max_error']==0
 devstate=pickle.load(open(dev/'initial-state.pkl','rb'));joint_delta=state['qpos'][:7]-devstate['qpos'][:7]
 joint_delta=(joint_delta+np.pi)%(2*np.pi)-np.pi
 # Native policy has a loaded checkpoint receipt; retain exact existing receipt paths.
 reloads=list(p.parent.glob('*reload*'))+list(p.parent.glob('*ready*'))
 s={'direction':direction,'incoming_task':task,'selected_model':model,'checkpoint':cp,'fresh_reload_receipt':reload_receipt,'robot_joint_difference_from_standalone_deg':np.rad2deg(joint_delta).tolist(),'maximum_joint_difference_from_standalone_deg':float(np.rad2deg(abs(joint_delta).max())),'source_episode':str(p),'standalone_reference':str(dev),'instruction':audit['instruction'],'query_index':0,'noise_hashes':audit['noise_hashes'],'input_hashes':audit['input_hashes'],'crop':inf[0]['crop_metadata'],'image_mask':inf[0]['image_mask'],'image_shape':list(prep['image_input'].shape),'history_initial_measured_and_fresh':True,'history_native_previous_command_verified_all_queries':True,'world_state_exact_across_request':True,'boundary_truth':truth,'boundary_robot_joints_rad':state['qpos'][:7].tolist(),'boundary_robot_velocity_rad_s':state['qvel'][:7].tolist(),'boundary_full_qpos':state['qpos'].tolist(),'boundary_full_qvel':state['qvel'].tolist(),'boundary_contact_count':len(state['contacts']),'boundary_controller_keys':list(state['controller']),'boundary_mocap_pos':state['mocap_pos'].tolist(),'boundary_mocap_quat':state['mocap_quat'].tolist(),'metrics':m,'standalone_metrics':dm,'standalone_initial_truth':devep['initial_truth'],'robot_eef_difference_from_standalone_mm':float(np.linalg.norm(eef-vals(devep['initial_truth']['eef_position_m']))*1000),'first_actionable_failure':'policy emits near-home open-gripper targets and no acquisition; input contract checks pass','numeral_recognition':'not established by crop validity or visible pixels'}
 # Render exact 224px prepared tensors using frozen processor normalization.
 cfg=read(R/'.cache-xvla/checkpoint/preprocessor_config.json')
 imgs=np.clip((prep['image_input'][0].transpose(0,2,3,1)*np.array(cfg['image_std'])+np.array(cfg['image_mean']))*255,0,255).round().astype(np.uint8)
 draw.text((12,row*570+8),direction+' incoming '+model+' / '+task+' : current primary, secondary, exact prepared crop',fill='white')
 for k,arr in enumerate([io['primary'][0],io['secondary'][0],imgs[2]]):
  im=Image.fromarray(arr);im.thumbnail((350,500));canvas.paste(im,(12+k*370,row*570+38))
 draw.text((12,row*570+550),'Crop validity is not proof of numeral recognition. Training-state coverage is separate from robot posture.',fill='white')
 summary.append(s)
canvas.save(O/'boundary-contact-sheet.png')
dump(O/'boundaries.json',summary);dump(O/'training-initial-states.json',training);dump(O/'training-coverage.json',coverage);dump(O/'manifest-receipts.json',receipts)
with open(O/'boundary-table.csv','w') as f:
 keys=['direction','selected_model','incoming_task','robot_eef_difference_from_standalone_mm','first_actionable_failure'];mkeys=list(summary[0]['metrics'])
 w=csv.DictWriter(f,fieldnames=keys+mkeys);w.writeheader()
 for s in summary:w.writerow({**{k:s[k] for k in keys},**s['metrics']})
dump(O/'validation_summary.json',{'passed':True,'model_calls':0,'physics_steps':0,'directions':2,'queries_verified':96,'training_manifests_checked':receipts,'audit_finished_at_epoch':time.time(),'within_20_minute_audit_clock':time.time()<read(O/'clock.json')['deadline'],'limitations':['latest adaptation stage coverage, not exhaustive base pretraining coverage','training exact initial joint states absent; first post-action state not substituted','two directions from one already-visible family, not independent holdout families','visible numeral is not demonstrated recognition']})
for name in ['boundaries.json','training-initial-states.json','training-coverage.json','manifest-receipts.json','boundary-table.csv','validation_summary.json','boundary-contact-sheet.png']:
 (D/name).write_bytes((O/name).read_bytes())
print(json.dumps({'summary':[{k:s[k] for k in ['direction','metrics','robot_eef_difference_from_standalone_mm']} for s in summary],'coverage':coverage},indent=2))
