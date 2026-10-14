"""LOAD04: matched fixture-counterbalanced continuation; old results immutable."""
import os,sys,json,time,copy,hashlib,gzip,subprocess
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
OUT=Path('/var/tmp/typhon-load04-20261014')
OLD=Path('/var/tmp/typhon-load01-20261013')
BASE=OLD/'train-FL/step3000.pt'
BASE_SHA='76334b9a17ac5fc49f46c2cb850a3c39ef38e0f13b9cdadc324f760567c5f128'
SOURCES=['demo_0001','demo_0002','demo_0003','demo_0005']
from load03_common import read,dump,sha,array_sha
def endpoint(config,task):
 c=copy.deepcopy(config);w=c['workspace']
 if task=='feeding':
  c['task']={'type':'feeding','target_position_xy_m':w['staging_position_xy_m'],'target_yaw_rad':0.,'target_position_jitter_m':0.,'target_region_size_m':w['staging_size_m'],'instruction':read(OUT/'protocol.json')['feeding_template']}
 return c
def scores(row,config):
 from supervisor.feeding_loading.evaluation import score_episode
 return {task:score_episode(row['initial_truth'],row['final_truth'],row['stability_truths'],endpoint(config,task),row['attempts']) for task in ['feeding','loading']}
