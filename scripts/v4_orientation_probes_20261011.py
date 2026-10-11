"""Controlled numeral/goal counterfactuals; no policy or training data is changed."""
import argparse,copy,gzip,hashlib,json,math,os,pickle,sys,time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v4_prompt_probes_20261011 import SEEDS,read,save,sha,load_payload,describe
POSITIONS=[[-.14,-.15],[-.08,-.08],[.04,-.14]]

def prepare(bank,out):
 jobs=[]
 for pi,xy in enumerate(POSITIONS):
  for yaw in [0,90,180,270]:
   for goal in [0,180]:jobs.append({'id':f'p{pi}-yaw{yaw:03d}-goal{goal:03d}-original','position_index':pi,'xy':xy,'object_yaw_deg':yaw,'goal_yaw_deg':goal,'visual':'original'})
   if pi==0 and yaw in [0,180]:
    for visual in ['numeral_removed','numeral_rotated180','sham']:
     jobs.append({'id':f'p{pi}-yaw{yaw:03d}-goal000-{visual}','position_index':pi,'xy':xy,'object_yaw_deg':yaw,'goal_yaw_deg':0,'visual':visual})
 assert len(jobs)==30
 out.mkdir(parents=True,exist_ok=False)
 save(out/'protocol.json',{'schema':'v4-orientation-counterfactuals-v1','bank':str(bank),'bank_manifest_sha256':sha(bank/'manifest.json'),
  'jobs':jobs,'models':['v4-m6','vanilla-pt-domain3'],'seeds':SEEDS,'planned_inferences':180,
  'controls':{'identical_lighting_and_cameras':True,'no_pose_input_to_policy':True,'history':'empty at initial noncontact state',
   'goal_contrast':'text only; identical RGB and robot proprio for goal0/180; the stage marker stays at0deg to isolate explicit supplied-goal language',
   'numeral_contrast':'runtime changes only to zero-contact zero-mass blue glyph geoms; orange body, studs, scene physics, camera and robot remain fixed',
   'sham':'write unchanged glyph arrays, forward and re-render',
   'paired_seed_noise':True,'rollouts':False,'test_time_selection':False},
  'limitations':['Goal180 may be outside fine-tuning goal coverage; this is a diagnostic, not a claim about familiar-task performance.',
   'A changed action indicates input sensitivity, not correct orientation understanding. Unchanged short-horizon grasp actions alone cannot prove orientation blindness.',
   'Numeral-only counterfactuals intentionally break the normal physical association between numeral and stud orientation.'],
  'code_sha256':sha(Path(__file__))})
 (out/'probe-source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({'prepared_render_conditions':30,'planned_inferences':180}))

def check_order(root):
 assert (root/'comparison-complete.json').exists(),'Comparison first'
 assert (root/'ordered-work-complete.json').exists(),'Fresh V4 search first'

def render(out,root):
 check_order(root)
 from scipy.spatial.transform import Rotation
 from PIL import Image,ImageDraw
 from supervisor.feeding_loading.environment import FeedingLoadingEnv
 from supervisor.baseline.xvla import task_prompt
 from supervisor.adaptation.observations import object_crop
 from m16_visual_check import physics_signature
 p=read(out/'protocol.json');assert p['code_sha256']==sha(Path(__file__))
 bank=Path(p['bank']);base=load_payload(bank/'A_initial/payload.pkl.gz');meta=read(bank/'A_initial/metadata.json')
 results=[];tiles=[]
 for pi,xy in enumerate(POSITIONS):
  for yaw in [0,90,180,270]:
   cfg=copy.deepcopy(meta['scenario']['config']);rad=math.radians(yaw)
   cfg['initial_state'].update(position_xy_m=xy,position_jitter_m=0.,yaw_rad=rad,yaw_choices_rad=[rad])
   cfg['task']['target_yaw_rad']=0.
   env=FeedingLoadingEnv(cfg)
   try:
    obs=env.reset(61610);sim=env.env.sim;m=sim.model
    marks=[i for i in range(m.ngeom) if (m.geom_id2name(i) or '').startswith('duplo_v2_numeral_')];assert marks
    assert all(m.geom_contype[i]==0 and m.geom_conaffinity[i]==0 for i in marks)
    pos=m.geom_pos.copy();rgba=m.geom_rgba.copy();qpos=sim.data.qpos.copy();qvel=sim.data.qvel.copy();physics=physics_signature(env)
    primary_original=obs['rgb']['primary'].copy();secondary_original=obs['rgb']['secondary'].copy()
    for job in [j for j in p['jobs'] if j['position_index']==pi and j['object_yaw_deg']==yaw]:
     m.geom_pos[:]=pos;m.geom_rgba[:]=rgba
     if job['visual']=='numeral_removed':m.geom_rgba[marks,3]=0
     elif job['visual']=='numeral_rotated180':
      centre=-cfg['object']['depth_m']*(4.8/43.)/2
      m.geom_pos[marks,0]=-pos[marks,0];m.geom_pos[marks,1]=2*centre-pos[marks,1]
     sim.forward();current=env.observe()
     assert np.array_equal(sim.data.qpos,qpos) and np.array_equal(sim.data.qvel,qvel)
     assert physics_signature(env)==physics
     other=[i for i in range(m.ngeom) if i not in marks]
     assert np.array_equal(m.geom_pos[other],pos[other]) and np.array_equal(m.geom_rgba[other],rgba[other])
     diffs={cam:int(np.any(current['rgb'][cam]!=original,axis=2).sum()) for cam,original in [('primary',primary_original),('secondary',secondary_original)]}
     if job['visual'] in ['original','sham']:assert not any(diffs.values()),('Sham changed RGB',job,diffs)
     payload=copy.deepcopy(base);payload['images']=[current['rgb']['primary'],current['rgb']['secondary']];payload['history']=[]
     robot=current['proprio'];rot=Rotation.from_quat(robot['eef_quat_xyzw']).as_matrix()
     payload['proprio']=np.r_[robot['eef_position_m'],rot[:,0],rot[:,1],0.,np.zeros(10)].astype(np.float32)
     goal=copy.deepcopy(current['task']);goal['target_yaw_rad']=math.radians(job['goal_yaw_deg']);payload['instruction']=task_prompt(goal)
     folder=out/'inputs'/job['id'];folder.mkdir(parents=True,exist_ok=False)
     with gzip.open(folder/'payload.pkl.gz','wb',compresslevel=1) as f:pickle.dump(payload,f,protocol=4)
     for cam,rgb in current['rgb'].items():Image.fromarray(rgb).save(folder/(cam+'.png'))
     crop,cropmeta=object_crop(current['rgb']['primary'])
     if crop is not None:Image.fromarray(crop).save(folder/'crop.png')
     record={'job':job,'payload_sha256':sha(folder/'payload.pkl.gz'),'source_bank_sha256':p['bank_manifest_sha256'],
      'measured_eef_position_m':robot['eef_position_m'].tolist(),'measured_eef_quat_xyzw':robot['eef_quat_xyzw'].tolist(),
      'evaluator_truth_separate_not_policy_input':env.truth(),'config':cfg,'crop':cropmeta,'changed_pixels_from_original':diffs,
      'changed_geometry_ids':marks if job['visual'] not in ['original','sham'] else [],'physics_unchanged':True,
      'identical_cameras_lighting_robot_state':True,'rendered_goal_marker_yaw_deg':0,'supplied_goal_yaw_deg':job['goal_yaw_deg']}
     save(folder/'metadata.json',record);results.append(record)
     if job['goal_yaw_deg']==0:
      tile=Image.new('RGB',(448,250),'white');tile.paste(Image.fromarray(current['rgb']['primary']).resize((224,224)),(0,24))
      if crop is not None:tile.paste(Image.fromarray(crop).resize((224,224)),(224,24))
      ImageDraw.Draw(tile).text((4,5),job['id'],fill='black');tiles.append(tile)
   finally:env.close()
 sheet=Image.new('RGB',(448*4,250*math.ceil(len(tiles)/4)),'#ddd')
 for i,tile in enumerate(tiles):sheet.paste(tile,((i%4)*448,(i//4)*250))
 sheet.save(out/'rendered-conditions-contact-sheet.jpg',quality=92)
 save(out/'render-validation.json',{'conditions':len(results),'records':results,'original_sham_exact_pixel_equality':True,
  'physics_changes':False,'actual_model_processor_audit_pending':True})
 print(json.dumps({'rendered':len(results),'sheet':str(out/'rendered-conditions-contact-sheet.jpg')}))

def execute(out,root):
 check_order(root);p=read(out/'protocol.json');assert read(out/'render-validation.json')['conditions']==30
 from m7_evaluate import M7Policy
 from m6_evaluate import M6Policy
 from supervisor.adaptation.m6_contract import reorder
 for model in p['models']:
  d=out/model;d.mkdir(exist_ok=True);policy=None
  try:
   if model=='vanilla-pt-domain3':policy=M6Policy('vanilla-domain3',None,d/'worker.log')
   else:policy=M7Policy(Path('/var/tmp/humanoid-m7-checkpoints-20261009')/(model+'.pt'),d/'worker.log',annotation_conditioned_v2=True)
   save(d/'load-info.json',policy.load_info)
   for job in p['jobs']:
    source=out/'inputs'/job['id'];payload=load_payload(source/'payload.pkl.gz');meta=read(source/'metadata.json')
    for seed in p['seeds']:
     dest=out/'responses'/model/job['id']/str(seed)
     if (dest/'result.json').exists():continue
     dest.mkdir(parents=True,exist_ok=False)
     wire=payload if model!='vanilla-pt-domain3' else {k:payload[k] for k in ['op','images','instruction','proprio']}
     policy._request({'op':'reset','seed':seed});start=time.perf_counter();response=policy._request(wire)
     if model=='vanilla-pt-domain3':actions=np.asarray(response['actions'])
     else:actions=reorder(np.asarray([v['target_generic_interleaved'] for v in response['timed']['waypoints']]),True)
     assert actions.shape==(30,20) and np.isfinite(actions).all()
     with gzip.open(dest/'response.pkl.gz','wb',compresslevel=1) as f:pickle.dump(response,f,protocol=4)
     np.savez_compressed(dest/'actions.npz',actions=actions)
     save(dest/'result.json',{'model':model,'job':job,'seed':seed,'wall_seconds':time.perf_counter()-start,
       'payload_sha256':sha(source/'payload.pkl.gz'),'response_sha256':sha(dest/'response.pkl.gz'),
       'image_mask':response.get('image_mask'),'actual_input_shapes':response.get('input_shapes'),
       'action_metrics':describe(actions,meta,payload),'simulator_execution':False})
     print(json.dumps({'done':str(dest),'seed':seed}),flush=True)
  finally:
   if policy:policy.close()
 count=len(list((out/'responses').glob('*/*/*/result.json')));assert count==180
 save(out/'inference-complete.json',{'responses':count,'planned':180})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','render','execute']);p.add_argument('--bank',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--root',type=Path);a=p.parse_args()
 if a.command=='prepare':prepare(a.bank,a.output)
 elif a.command=='render':render(a.output,a.root)
 else:execute(a.output,a.root)
