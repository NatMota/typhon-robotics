"""LOAD02: instruction-only intervention, fixed physical scene and original scoring."""
import os,sys,json,time,copy,hashlib,gzip,subprocess
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
OUT=Path('/var/tmp/typhon-load03-20261013');OLD=Path('/var/tmp/typhon-load01-20261013')
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def array_sha(a):
 import numpy as np
 a=np.asarray(a);return hashlib.sha256(str(a.dtype).encode()+json.dumps(list(a.shape)).encode()+a.tobytes()).hexdigest()
def endpoint(config,task):
 c=copy.deepcopy(config);w=c['workspace']
 if task=='feeding':
  c['task']={'type':'feeding','target_position_xy_m':w['staging_position_xy_m'],'target_yaw_rad':0.,'target_position_jitter_m':0.,'target_region_size_m':w['staging_size_m'],'instruction':read(OUT/'protocol.json')['feeding_template']}
 return c
def scores(row,config):
 from supervisor.feeding_loading.evaluation import score_episode
 return {task:score_episode(row['initial_truth'],row['final_truth'],row['stability_truths'],endpoint(config,task),row['attempts']) for task in ['feeding','loading']}
