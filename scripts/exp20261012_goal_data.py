"""Goal-grounding pilot data: train-only sources, held-out start/goal combinations."""
import os,sys,json,copy,time,argparse
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));os.chdir(R)
from exp20261012_local_robustness import dump,sha
OUT=Path('/var/tmp/typhon-ost-sweep-20261012/goal-grounding');GOALS=[[.125,-.12],[.125,.03],[0.,-.25]]
def prepare():
 from supervisor.feeding_loading.config import validate_config
 data=R/'frozen_results/personal-m3-20261009/data-canonical/dataset.json';episodes=[e for e in json.loads(data.read_text())['episodes']if e['split']=='P_train'];assert len(episodes)==6
 rows=[]
 for i,e in enumerate(episodes):
  for g,xy in enumerate(GOALS):
   cfg=copy.deepcopy(e['config']);cfg['task'].update(target_position_xy_m=xy,target_position_jitter_m=0.,target_yaw_rad=0.)
   cfg['workspace']['staging_position_xy_m']=xy.copy();validate_config(cfg)
   sid=e['id']+'-goal'+str(g);s={'scenario_id':sid,'cohort':'training_source_goal_reconstruction','source_demo_id':e['source_demo_id'],'category':'normal_feeding','seed':812120,'reconstruction_confidence':'same personal-source initial state, supplied counterfactual goal','initial_sim':{'position_xy_m':cfg['initial_state']['position_xy_m'],'yaw_rad':cfg['initial_state']['yaw_rad']},'target_sim':{'position_xy_m':xy,'yaw_rad':0.},'goal_index':g,'varied_split':'combination_holdout'if g==i%3 else'train','source_index':i}
   rows.append({'scenario':s,'config':cfg})
 OUT.mkdir(exist_ok=False);(OUT/'data').mkdir();(OUT/'data/configs').mkdir()
 spec={'created_before_execution':time.time(),'source_manifest_sha256':sha(data),'sources':[e['id']for e in episodes],'goals':GOALS,'scenarios':rows,'training_rule':'Varied: each source has two of three flat feeding goals; third combination withheld. Fixed control: only original feeding goal, matched source/update count. No loading labels in either.','updates':3000,'initialization':'same fixed broad-Pt12k checkpoint; joint continuation with fixed optimizer settings, not a fresh paper pretraining reproduction','evaluation':'six withheld source-goal combinations and six source starts at loading fixture, two paired seeds each. These are development transfer probes, not independent humans. Original136 comparison untouched.','success_gate':'Every admitted teacher trajectory must pass strict score and deterministic pre-action replay, with all failures retained.','claims':'Goal diversity versus extra identical-goal updates. More goal combinations implies more unique trajectories; expose this difference.'}
 dump(OUT/'protocol.json',spec)
 for x in rows:dump(OUT/'data/configs'/(x['scenario']['scenario_id']+'.json'),x['config'])
 print('PREPARED18',flush=True)
def generate(shard):
 from m15_execute import execute
 from m3_prepare_data import replay
 spec=json.loads((OUT/'protocol.json').read_text());rows=[]
 for i,x in enumerate(spec['scenarios']):
  if i%2!=shard:continue
  s=x['scenario'];out=OUT/'data';r=execute(s,out)
  rec={'scenario':s,'success':r['success'],'failure_modes':r['failure_modes'],'result':str(out/r['artifacts']['directory'])}
  if r['success']:
   e=replay(out/r['artifacts']['directory'],out,s['scenario_id'],s['varied_split'])
   e.update(goal_index=s['goal_index'],source_index=s['source_index']);rec['episode']=e
  rows.append(rec);dump(OUT/f'generation-shard{shard}.json',rows)
 print('FINISHED',shard,len(rows),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('op',choices=['prepare','generate']);p.add_argument('--shard',type=int);a=p.parse_args()
 prepare()if a.op=='prepare'else generate(a.shard)
