"""Qualify train-only recovery labels and convert measured-proprio records to native command history."""
import os,sys,json,gzip,copy
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));os.chdir(R)
import numpy as np
from scipy.spatial.transform import Rotation
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.baseline.xvla import to_action,task_prompt
from exp20261012_local_robustness import dump,sha
ROOT=Path('/var/tmp/typhon-ost-sweep-20261012');SRC=ROOT/'recovery-v2';OUT=ROOT/'recovery-learning';OUT.mkdir(exist_ok=False);(OUT/'data').mkdir()
protocol=json.loads((SRC/'protocol.json').read_text());goal=json.loads((ROOT/'goal-grounding/data/dataset-v2.json').read_text())
ids={e['source_demo_id']:e['source_index']for e in goal['episodes']};seen=set();entries=[];rejections=[];duplicates=[]
for case in protocol['cases']:
 d=SRC/case['id']/'grasp_transform_expert';s=json.loads((d/'summary.json').read_text());sc=json.loads((Path(case['source'])/'scenario.json').read_text());cfg=sc['config']
 state=s['same_start_state_sha256']
 if state in seen:duplicates.append(case['id']);continue
 seen.add(state);assert case['source_demo_id']in ids
 if not s['evaluation']['success']or s['exception']:rejections.append({'id':case['id'],'reason':'teacher_failed'});continue
 a0=np.load(d/'correction.npz');a={k:a0[k]for k in a0.files};a0.close();trace=[json.loads(l)for l in gzip.open(d/'privileged-trace.jsonl.gz','rt')]
 env=FeedingLoadingEnv(cfg)
 try:
  env.reset(sc['seed']);sim=env.env.sim;robot=env.env.robots[0];jointids=[sim.model.joint_name2id(x)for x in robot.robot_model.joints];limits=sim.model.jnt_range[jointids].copy()
 finally:env.close()
 q=np.array([x['pre_action_joint_position_rad']for x in trace]);margin=float(np.minimum(q-limits[:,0],limits[:,1]-q).min());roundtrip=[]
 for i,x in enumerate(trace):
  obs={'proprio':{'eef_position_m':x['pre_action_truth']['eef_position_m'],'eef_quat_xyzw':Rotation.from_matrix(x['pre_action_eef_rotation_matrix']).as_quat()}}
  roundtrip.append(float(abs(to_action(a['targets'][i],obs,cfg)-a['actions'][i]).max()))
 if margin<0 or max(roundtrip)>1e-4:rejections.append({'id':case['id'],'reason':'joint_or_action_contract_failed','margin_rad':margin,'max_roundtrip':max(roundtrip)});continue
 indices=a['frame_action_indices'];prop=a['proprio'][indices].copy();prop[:,9]=0 # native M3 history leaves gripper conditioning unset
 for fi,ai in enumerate(indices):
  if ai>0:prop[fi,:9]=a['targets'][ai-1,:9]
 assert len(prop)==len(a['primary'])and np.isfinite(prop).all()
 path=OUT/'data'/(case['id']+'.npz');np.savez_compressed(path,primary=a['primary'],secondary=a['secondary'],frame_action_indices=indices,targets=a['targets'],proprio=prop)
 entries.append({'id':'correction-'+case['id'],'path':str(path),'sha256':sha(path),'source_demo_id':case['source_demo_id'],'source_index':ids[case['source_demo_id']],'goal_index':0,'split':'P_train_recovery','config':cfg,'instruction':task_prompt(cfg['task']),'source':str(d),'source_npz_sha256':sha(d/'correction.npz'),'source_state_sha256':state,'original_prefix_steps':case['prefix_steps'],'joint_margin_rad':margin,'action_roundtrip_max':max(roundtrip),'qualification':'strict-success teacher; exact physical prefix replay; native action-history conversion; privileged teacher, RGB+robot actor inputs','proprio':'measured first frame then previous absolute target, gripper slot zero matching M3 baseline','limitations':'Intervention resets policy history at handoff; no human action timing/labels; selected training failures, not unbiased competence estimate'})
dump(OUT/'dataset.json',{'episodes':entries,'duplicate_states_removed':duplicates,'rejected':rejections,'no_validation_or_confirmatory_sources':True,'frozen_previous_diagnostic_recoveries_excluded':True})
print('ADMITTED',len(entries),'DUPLICATES',duplicates,'REJECTED',rejections,flush=True)
