"""Replay archived actions; assess the observer without letting it alter actions."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse,gzip,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from exp20261011_hierarchy import DATA,TRAIN,save,sha,serial
from exp20261011_hierarchy_runtime_v3 import PhaseSupervisor
OBSERVER_SOURCE=ROOT/'scripts/exp20261011_hierarchy_runtime_v3.py'
from supervisor.feeding_loading.environment import FeedingLoadingEnv
OUT=Path('/var/tmp/typhon-research-20261012/observer/passive-v3')
def main():
 global OUT,PhaseSupervisor,OBSERVER_SOURCE
 parser=argparse.ArgumentParser();parser.add_argument('--phase-independent',action='store_true');parser.add_argument('--stereo',action='store_true');parser.add_argument('--body-support',action='store_true');args=parser.parse_args()
 if args.phase_independent:
  from exp20261012_observation_hypotheses import PhaseSupervisor as IndependentSupervisor
  PhaseSupervisor=IndependentSupervisor;OUT=OUT.parent/'passive-independent-v1';OBSERVER_SOURCE=ROOT/'scripts/exp20261012_observation_hypotheses.py'
 if args.stereo:
  from exp20261012_stereo_observer import PhaseSupervisor as StereoSupervisor
  PhaseSupervisor=StereoSupervisor;OUT=OUT.parent/'passive-stereo-v1';OBSERVER_SOURCE=ROOT/'scripts/exp20261012_stereo_observer.py'
 if args.body_support:
  from exp20261012_body_support_observer import PhaseSupervisor as BodySupervisor
  PhaseSupervisor=BodySupervisor;OUT=OUT.parent/'passive-body-support-v1';OBSERVER_SOURCE=ROOT/'scripts/exp20261012_body_support_observer.py'
 OUT.mkdir(parents=True,exist_ok=False)
 data=json.loads((DATA/'dataset.json').read_text())['episodes'];cases=[]
 for e in sorted(data,key=lambda e:e['id']):
  if e['id'] not in TRAIN:continue
  folder=DATA/e['source_path'].split('data-canonical/',1)[1]
  cases.append({'id':'expert-'+e['id'],'config':e['config'],'path':str(folder),'expert':True})
 local=Path('/var/tmp/typhon-research-20261012/local-robustness/evaluation/broad-pt-12000')
 for stem in ['local-demo_0002-original-seed7417','local-demo_0003-yawplus15deg-seed7417']:
  folder=local/stem;sc=json.loads((folder/'scenario.json').read_text())
  cases.append({'id':'broad-'+stem,'config':sc['config'],'path':str(folder),'expert':False})
 protocol={'schema':'passive-observer-qualification-v1','cases':cases,'stride_steps':4,'observer_file_sha256':sha(OBSERVER_SOURCE),'no_action_modifications':True,'truth_access':'Independent scoring and replay agreement only. No truth supplied to PhaseSupervisor.','predeclared_qualification':{'max_replay_final_position_difference_m':1e-6,'false_goal_evidence_frames':0,'held_precision_min':.95,'held_recall_min':.90,'observed_held_coverage_min':.60,'observed_position_p95_m':.010,'observed_yaw_p95_deg':10.},'statistical_scope':'Six training expert episodes and two previously inspected development rollouts; not a held-out policy evaluation'}
 save(OUT/'protocol.json',protocol);allrows=[];summaries=[];start=time.monotonic()
 for c in cases:
  folder=Path(c['path']);episode=json.loads((folder/'episode.json').read_text());cfg=c['config'];dest=OUT/c['id'];dest.mkdir();observer=PhaseSupervisor(cfg)
  if c['expert']:
   archived=[json.loads(l) for l in (folder/'robot_actions.jsonl').read_text().splitlines()]
   actions=[e['normalized_command'] for e in archived]
   expected=archived[-1]['post_action_truth']
  else:
   archived=[json.loads(l) for l in gzip.open(folder/'trajectory.jsonl.gz','rt')]
   # Recorder contains an initial state without an action; ignore it explicitly.
   actions=[e['requested_action'] for e in archived if e.get('requested_action') is not None]
   expected=episode['final_truth']
  env=FeedingLoadingEnv(cfg);rows=[]
  try:
   env.reset(episode['seed']);initial_z=env.truth()['object_position_m'][2]
   def inspect(i):
    obs=env.observe();assert set(obs)<={'rgb','proprio','task','cameras'}
    prediction=observer.update(obs,env.steps)
    truth=env.truth();pos=np.array(truth['object_position_m']);yaw=truth['object_yaw_rad'];b=prediction['object']
    # Truth is read after prediction, for scoring only.
    held=bool(truth['bilateral_grasp_contact'] and pos[2]-initial_z>=.01)
    goal=np.array(cfg['task']['target_position_xy_m']);dyaw=math.atan2(math.sin(yaw-cfg['task']['target_yaw_rad']),math.cos(yaw-cfg['task']['target_yaw_rad']))
    rot=np.array([[math.cos(dyaw),-math.sin(dyaw)],[math.sin(dyaw),math.cos(dyaw)]])
    extent=np.abs(rot)@np.array([cfg['object']['width_m'],cfg['object']['depth_m']])/2
    gy=cfg['task']['target_yaw_rad'];gr=np.array([[math.cos(gy),-math.sin(gy)],[math.sin(gy),math.cos(gy)]])
    margins=cfg['task']['target_region_size_m']/2-np.abs(gr.T@(pos[:2]-goal))-extent
    truth_goal=bool(margins.min()>=0 and abs(dyaw)<=cfg['evaluation']['yaw_tolerance_rad'])
    pe=None if b['position_m'] is None else float(np.linalg.norm(np.array(b['position_m'])-pos))
    ye=None if b['yaw_rad'] is None else math.degrees(abs(math.atan2(math.sin(b['yaw_rad']-yaw),math.cos(b['yaw_rad']-yaw))))
    row={'action_index':i,'step':env.steps,'prediction':prediction,'scorer_only':{'truth':truth,'supported_held':held,'footprint_and_yaw_goal':truth_goal,'position_error_m':pe,'yaw_error_deg':ye,'false_goal_positive':prediction['goal']['complete'] is True and not truth_goal}}
    rows.append(row)
    if i==0 or (held and sum(r['scorer_only']['supported_held'] for r in rows)==1):
     import imageio.v2 as imageio
     imageio.imwrite(dest/f'frame-{i:04d}.png',np.concatenate([obs['rgb']['primary'],obs['rgb']['secondary']],1))
   inspect(0)
   for i,a in enumerate(actions,1):
    env.step(a)
    if i%4==0 or i==len(actions):inspect(i)
   final=env.truth();diff=float(np.linalg.norm(np.array(final['object_position_m'])-expected['object_position_m']))
   with gzip.open(dest/'observations.jsonl.gz','wt') as f:
    for r in rows:f.write(json.dumps(r,default=serial,allow_nan=False)+'\n')
   s={'case':c['id'],'samples':len(rows),'archived_actions':len(actions),'replay_final_position_difference_m':diff,'initial_expected_success':episode.get('success'),'observer_final_phase':observer.phase,'phase_transitions':observer.transitions,'false_goal_positive_frames':sum(r['scorer_only']['false_goal_positive'] for r in rows),'action_changes':0}
   summaries.append(s);allrows.extend(rows);save(dest/'summary.json',s);print(json.dumps(s),flush=True)
  finally:env.close()
 truth_held=[r for r in allrows if r['scorer_only']['supported_held']];pred_held=[r for r in allrows if r['prediction']['held']];tp=sum(r['scorer_only']['supported_held'] for r in pred_held)
 obs=[r for r in allrows if r['scorer_only']['position_error_m'] is not None]
 result={'cases':summaries,'samples':len(allrows),'wall_s':time.monotonic()-start,'held_precision':None if not pred_held else tp/len(pred_held),'held_recall':None if not truth_held else tp/len(truth_held),'held_observation_coverage':sum(r['prediction']['object']['status']!='unknown' for r in truth_held)/max(1,len(truth_held)),'observed_position_p95_m':float(np.percentile([r['scorer_only']['position_error_m'] for r in obs],95)),'observed_yaw_p95_deg':float(np.percentile([r['scorer_only']['yaw_error_deg'] for r in obs],95)),'false_goal_evidence_frames':sum(r['scorer_only']['false_goal_positive'] for r in allrows),'max_replay_final_position_difference_m':max(s['replay_final_position_difference_m'] for s in summaries),'actions_changed':0}
 gates=protocol['predeclared_qualification'];result['qualification_passed']=bool(result['max_replay_final_position_difference_m']<=gates['max_replay_final_position_difference_m'] and result['false_goal_evidence_frames']==0 and result['held_precision'] is not None and result['held_precision']>=.95 and result['held_recall']>=.9 and result['held_observation_coverage']>=.6 and result['observed_position_p95_m']<=.01 and result['observed_yaw_p95_deg']<=10)
 save(OUT/'summary.json',result);save(OUT/'complete.json',{'complete':True,'qualification_passed':result['qualification_passed'],'cases':len(cases)});print(json.dumps(result),flush=True)
if __name__=='__main__':main()
