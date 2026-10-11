"""Replay six exact P_train expert prefixes, then inspect measured hold frames."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from pathlib import Path
import json,sys,time,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from exp20261011_hierarchy import DATA,OUT,TRAIN,save,serial
from exp20261011_hierarchy_runtime_v2 import ObservableBlock,goal_evidence
from supervisor.feeding_loading.environment import FeedingLoadingEnv

def gpu():
 return subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True,stdout=subprocess.PIPE).stdout.strip()
def main():
 out=OUT/'stationary-train-qualification-v1';out.mkdir(exist_ok=False);start=time.monotonic();before=gpu()
 evidence=json.loads((OUT/'phase-head-v1/source-evidence.json').read_text());data={e['id']:e for e in json.loads((DATA/'dataset.json').read_text())['episodes']};rows=[]
 for e in evidence:
  if e['take'] not in TRAIN:continue
  source=data[e['take']];cfg=source['config'];release=next(p for p in e['primitives'] if p['label']=='release_open')['action_start'];folder=DATA/source['source_path'].split('data-canonical/',1)[1];ac=[json.loads(l) for l in (folder/'robot_actions.jsonl').read_text().splitlines()];episode=json.loads((folder/'episode.json').read_text());case=out/e['take'];case.mkdir()
  env=FeedingLoadingEnv(cfg);sensor=ObservableBlock(cfg)
  try:
   env.reset(episode['seed'])
   for command in ac[:release]:env.step(command['normalized_command'])
   initial=env.truth();samples=[]
   for relative in range(0,13,4):
    if relative:
     for _ in range(4):env.step(np.r_[np.zeros(6),1.])
    obs=env.observe();estimate=sensor.step(obs,relative,held=False,carrying_phase=True);g=goal_evidence(estimate,cfg)
    truth=env.truth() # Independent qualification only, never passed to sensor.
    error=None if estimate['position_m'] is None else float(np.linalg.norm(estimate['position_m']-truth['object_position_m']))
    samples.append({'relative_control_step':relative,'estimate':estimate,'goal':g,'evaluator_truth':truth,'evaluator_position_error_m':error})
    if relative in [0,12]:
     import imageio.v2 as imageio
     imageio.imwrite(case/('frame-%02d.png'%relative),np.concatenate(list(obs['rgb'].values()),1))
   final=env.truth();row={'take':e['take'],'prefix_steps':release,'prefix_measured_object_match_m':float(np.linalg.norm(np.array(initial['object_position_m'])-ac[release-1]['post_action_truth']['object_position_m'])),'prefix_measured_object_yaw_match_deg':float(np.degrees(abs(initial['object_yaw_rad']-ac[release-1]['post_action_truth']['object_yaw_rad']))),'stationary_control_steps':12,'last_three_conservative_visual_gates_pass':all(x['goal']['complete'] is True for x in samples[-3:]),'position_drift_m':float(np.linalg.norm(np.array(final['object_position_m'])-initial['object_position_m'])),'grasp_preserved':bool(final['bilateral_grasp_contact']),'samples':samples,'gpu_processes':gpu()}
   save(case/'result.json',row);rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['samples','gpu_processes']}),flush=True)
  finally:env.close()
 summary={'takes':len(rows),'qualified':sum(x['last_three_conservative_visual_gates_pass'] for x in rows),'hold_s':.6,'separate_three_frame_span_s':.4,'max_prefix_position_replay_difference_m':max(x['prefix_measured_object_match_m'] for x in rows),'heldout_takes_inspected':0,'wall_s':time.monotonic()-start,'before_gpu_processes':before,'after_gpu_processes':gpu(),'process_pid':os.getpid(),'no_VLA_or_training':True,'truth_in_actor':False}
 save(out/'summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
