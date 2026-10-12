"""Execute two untouched native baselines on the previously frozen 34-layout panel."""
import os,sys,json,copy,time,subprocess,argparse
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));os.chdir(R)
from exp20261012_local_robustness import dump,sha
OUT=Path('/var/tmp/typhon-ost-sweep-20261012')
SOURCE=Path('/var/tmp/typhon-research-20261012/confirmatory/layouts-protocol.json')
def prepare():
 p=json.loads(SOURCE.read_text());assert sha(SOURCE)=='1a3aa0129c8c247dc9893760aa2b4244c86a9d17b19e57559485206fe2405fce'
 from supervisor.feeding_loading.config import validate_config
 import math
 corrections=[]
 for l in p['layouts']:
  cfg=l['config'];before=copy.deepcopy(cfg['initial_state']['position_xy_m']);radius=math.hypot(cfg['object']['width_m'],cfg['object']['depth_m'])/2
  for axis,v in enumerate(before):
   lo,hi=cfg['control']['workspace_bounds_m'][axis];size=cfg['workspace']['table_width_m' if axis==0 else 'table_depth_m']
   cfg['initial_state']['position_xy_m'][axis]=min(min(hi,size/2)-radius-1e-6,max(max(lo,-size/2)+radius+1e-6,v))
  if before!=cfg['initial_state']['position_xy_m']:corrections.append({'layout':l['layout_id'],'before':before,'after':cfg['initial_state']['position_xy_m'],'reason':'Static footprint boundary validation; no policy action occurred in rejected launch.'})
  validate_config(cfg)
 old=json.loads(Path('/var/tmp/typhon-research-20261012/local-robustness/evaluation-protocol.json').read_text())
 scenarios=[]
 for l in p['layouts']:
  for seed in p['rollout_seeds']:
   scenarios.append({'scenario_id':l['layout_id']+'-seed'+str(seed),'source_demo_id':l.get('parent_source_demo_id'),'group':l['group'],'task':'feeding','seed':seed,'config':copy.deepcopy(l['config']),'attempt_limit':2,'frozen_layout_id':l['layout_id']})
 assert len(scenarios)==68
 spec={'schema':'ost-sweep-baselines-v2','static_preflight':{'passed':True,'layouts':34,'boundary_corrections':corrections},'frozen_before_calls':time.time(),'original_protocol':str(SOURCE),'original_protocol_sha256':sha(SOURCE),'scenarios':scenarios,'methods':old['methods'],'vanilla_base_sha256':sha(R/'.cache-xvla/checkpoint/model.safetensors'),'runs':136,'conditions':['vanilla-Pt-native','broad-Pt-12000'],'third_condition':'not run: no qualified intervention. Baselines can execute independently; cannot call this complete 204-suite.','changes':'Declared minimal initial-centre clipping of invalid layouts before any policy actions; seeds, goals, geometry, budget and adapters unchanged. No training on these scenes.','limitations':'Comparison of native model contracts (domain3 vs29), not pure weights-only causal contrast. Layouts cluster within source parents; not136 independent human demonstrations.','script_sha256':sha(__file__)}
 d=OUT/'confirmatory-v2';d.mkdir(exist_ok=False);dump(d/'protocol.json',spec)
 print('PREPARED136',flush=True)
def run(method):
 import numpy as np
 from motion_policy_diagnostics import PrefixPolicy,audit
 from m5_evaluate import supplemental
 from supervisor.baseline.benchmark import run_episode
 spec=json.loads((OUT/'confirmatory-v2/protocol.json').read_text());folder=OUT/'confirmatory-v2'/method;folder.mkdir(exist_ok=False)
 broad=method=='broad-Pt-12000';entry=spec['methods']['broad-pt-12000'];cp=Path(entry['checkpoint'])
 if broad:assert sha(cp)==entry['expected_sha256']
 else:assert sha(R/'.cache-xvla/checkpoint/model.safetensors')==spec['vanilla_base_sha256']
 class Policy(PrefixPolicy):
  def __init__(self):
   self.horizon=30;self.wrist='native';self.proprio=None;self.inferences=[]
   self.model_manifest={'method':method,'domain':29 if broad else 3,'O2':True,'actor_truth':False,'H':30,'steps':10,'checkpoint_sha256':entry['expected_sha256']if broad else spec['vanilla_base_sha256'],'native_contract':'generic_interleaved converted once'if broad else'native concatenated rotation columns','proprio':'previous executed target'}
   self.log=open(folder/'worker.log','a')
   if broad:cmd=[str(R/'.venv-xvla/bin/python'),'scripts/exp20261011_train_eval_worker.py','--checkpoint',str(cp),'--probe',entry['probe'],'--verification',str(folder/'reload-verification.json')]
   else:cmd=[str(R/'.venv-xvla/bin/python'),'-m','supervisor.adaptation.m6_worker','--version','O2','--condition','vanilla-domain3']
   self.process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,cwd=R,env=dict(os.environ,HF_HUB_OFFLINE='1'))
   self.load_info=self._receive(600);assert self.load_info.get('ready'),self.load_info
 policy=Policy();dump(folder/'manifest.json',{'model':policy.model_manifest,'load':policy.load_info,'protocol_sha256':sha(OUT/'confirmatory-v2/protocol.json'),'script_sha256':sha(__file__)});rows=[]
 try:
  for s in spec['scenarios']:
   dest=folder/s['scenario_id'];r=run_episode(s,'pretrained',dest,policy,{'model':policy.model_manifest},episode_setup=policy.instrument)
   if hasattr(policy,'telemetry'):policy.telemetry.close()
   np.savez_compressed(dest/'raw_predictions.npz',chunks=np.asarray(getattr(policy,'raw_chunks',[])),histories=np.asarray([np.full(20,np.nan)if h is None else h for h in getattr(policy,'histories',[])]))
   errors={k:r[k]for k in ['controller_exceptions','artifact_errors','evaluation_errors']if r[k]}
   row={'id':s['scenario_id'],'parent':s['source_demo_id'],'group':s['group'],'seed':s['seed'],'method':method,'success':r['success'],'errors':errors,'wall_seconds':r['wall_duration_s']}
   if not errors:
    row.update({**audit(r,dest),**supplemental(r,dest)})
    yaw=r['initial_truth']['object_yaw_rad']-s['config']['task']['target_yaw_rad'];initial=abs(float(np.degrees(np.arctan2(np.sin(yaw),np.cos(yaw)))))
    row.update(initial_yaw_error_deg=initial,final_yaw_error_deg=r['yaw_error_deg'],yaw_error_reduction_deg=initial-r['yaw_error_deg'],footprint_inside_target=r['evaluation']['criteria']['footprint_inside_target'],operational_placement=all(r['evaluation']['criteria'][k]for k in ['footprint_inside_target','marked_face_up','placement_height','released','stable']))
   rows.append(row);dump(folder/'summary.json',rows);print(json.dumps({'finished':len(rows),'total':68,**row}),flush=True)
   if errors:raise RuntimeError(errors)
 finally:policy.close()
 dump(folder/'complete.json',{'rollouts':len(rows),'all_scored':len(rows)==68,'script_sha256':sha(__file__)})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('op',choices=['prepare','run']);p.add_argument('--method',choices=['vanilla-Pt-native','broad-Pt-12000']);a=p.parse_args()
 prepare()if a.op=='prepare'else run(a.method)
