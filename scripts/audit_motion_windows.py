"""Matched replay windows, exact joint metadata and immutable artifact checks."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys,json,gzip,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from supervisor.feeding_loading.environment import FeedingLoadingEnv
ROOT=Path('results/motion-diagnostics-20261009')
s=json.loads(Path('frozen_results/personal-m4-20261009/recovery/PR/recovery-already_completed-01/scenario.json').read_text());env=FeedingLoadingEnv(s['config']);env.reset(s['seed']);robot=env.env.robots[0];sim=env.env.sim
indices=list(robot._ref_joint_indexes);names=[sim.model.joint_id2name(i) for i in indices];limits=sim.model.jnt_range[indices].copy();site=sim.model.site_id2name(robot.eef_site_id)
meta={'joint_names':names,'configured_limits_rad':limits.tolist(),'joint_indexing':'zero-based array indices; named joints use Panda numbering1..7','controlled_site':site,'limit_model':'MuJoCo2.3.7 solver constraints; negative margins are measured excursions, not evidence limits disabled. Per-constraint force/activation was not logged in original replay.','source_baseline':'b504eb0','frequency_hz':20};env.close()
output={}
for mode in ['original','fixed-wrist','hold']:
 rows=[json.loads(x) for x in gzip.open(ROOT/f'replay-{mode}/trace.jsonl.gz','rt')];n=len(rows);t=(np.arange(n)+1)/20;q=np.array([x['joint_positions'] for x in rows]);below=np.maximum(limits[:,0]-q,0);above=np.maximum(q-limits[:,1],0);exc=np.maximum(below,above);joint=[]
 for j,name in enumerate(names):
  ix=np.flatnonzero(exc[:,j]>0);joint.append({'name':name,'array_index':j,'outside_range_steps':len(ix),'outside_range_seconds_at20Hz':len(ix)/20,'max_excursion_rad':float(exc[:,j].max()),'first_outside_time_s':float(t[ix[0]]) if len(ix) else None,'limits_rad':limits[j].tolist()})
 windows={}
 for label,mask in [('whole',np.ones(n,dtype=bool)),('before_jump_t_le_16.5s',t<=16.5),('after_limit_t_ge_17.3s',t>=17.3)]:
  r=[v for v,keep in zip(rows,mask) if keep];windows[label]={'steps':len(r),'raw_xyz_error_mm_p50_p95_max':np.percentile([v['position_error_mm'] for v in r],[50,95,100]).tolist(),'raw_rotation_error_deg_p50_p95_max':np.percentile([v['orientation_error_deg'] for v in r],[50,95,100]).tolist(),'effective_xyz_error_mm_p50_p95_max':np.percentile([np.linalg.norm(np.array(v['controller_goal_xyz'])-v['eef_xyz'])*1000 for v in r],[50,95,100]).tolist(),'nearest_joint_limit_rad':min(v['nearest_joint_limit_rad'] for v in r)}
 output[mode]={'windows':windows,'joint_limit_excursions':joint}
(ROOT/'replay_window_audit.json').write_text(json.dumps({'metadata':meta,'replays':output},indent=2)+'\n');print(json.dumps({'joint_names':names,'original_excursions':[r for r in output['original']['joint_limit_excursions'] if r['outside_range_steps']]},indent=2))
