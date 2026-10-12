"""Post-hoc closed-loop source competence; copied native evaluator, no additional fit pass."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse,copy,gzip,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
OUT=Path('/var/tmp/typhon-research-20261012/local-robustness')

def dump(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def prepare():
 import math
 import numpy as np
 old=Path('/var/tmp/typhon-research-20261011/broad-pt/source-competence')
 original=json.loads((old/'evaluation-protocol.json').read_text())
 source_rows=json.loads((old/'evaluation/broad-pt-12000/summary.json').read_text())
 # Deterministic selection from previously published successes; no new outcome selection.
 ids=sorted(x['source_demo_id'] for x in source_rows if x['success'])[:2]
 assert ids==['demo_0002','demo_0003']
 centroid=np.mean([s['config']['initial_state']['position_xy_m'] for s in original['scenarios']],axis=0)
 scenarios=[]
 for demo in ids:
  parent=next(s for s in original['scenarios'] if s['source_demo_id']==demo)
  initial=np.array(parent['config']['initial_state']['position_xy_m']);direction=centroid-initial
  assert np.linalg.norm(direction)>1e-6;direction/=np.linalg.norm(direction)
  for name,offset,angle in [('original',0.,0.),('inward20mm',.02,0.),('outward20mm',-.02,0.),('yawplus15deg',0.,math.radians(15))]:
   for seed in [7417,7418]:
    s=copy.deepcopy(parent);cfg=s['config'];state=cfg['initial_state'];xy=initial+offset*direction
    yaw=float(math.atan2(math.sin(state['yaw_rad']+angle),math.cos(state['yaw_rad']+angle)))
    state.update(position_xy_m=xy.tolist(),yaw_rad=yaw,yaw_choices_rad=[yaw],position_jitter_m=0.)
    assert all(lo<float(v)<hi for v,(lo,hi) in zip(xy,cfg['control']['workspace_bounds_m'][:2]))
    s.update(scenario_id=f'local-{demo}-{name}-seed{seed}',group='training_parent_local_development',seed=seed,parent_scenario_id=parent['scenario_id'],perturbation={'kind':name,'translation_m':(xy-initial).tolist(),'yaw_delta_rad':angle,'reference':'six P_train initial-position centroid'},source_config_sha256=sha(old/'evaluation-protocol.json'))
    scenarios.append(s)
 p={**original,'schema':'typhon-local-robustness-v1','created_unix_s':time.time(),'scenarios':scenarios,'total_trials':16,'trial_count_per_method':16,'seeds':[7417,7418],'statistical_status':'Development: 2 training-derived parent scenes ×4 variants ×2 seeds; not independent human demonstrations or confirmatory holdout','selection':'Lexicographically first two previously strictly successful P_train scenes, frozen before new trials','centroid_xy_m':centroid.tolist(),'reason':'Measure spatial/yaw basin around source successes without fitting weights or changing controller','original_protocol_sha256':sha(old/'evaluation-protocol.json'),'original_evaluator_sha256':sha(ROOT/'scripts/exp20261011_train_source_competence.py'),'parent_model_successes':ids}
 assert len({x['scenario_id'] for x in scenarios})==16
 assert sum(x['perturbation']['kind']=='original' for x in scenarios)==4
 for s in scenarios:
  parent=next(x for x in original['scenarios'] if x['source_demo_id']==s['source_demo_id'])
  # Only initial state changes; supplied task, cameras, geometry and physical controller are exact copies.
  assert {k:v for k,v in s['config'].items() if k!='initial_state'}=={k:v for k,v in parent['config'].items() if k!='initial_state'}
 OUT.mkdir(parents=True,exist_ok=False)
 dump(OUT/'evaluation-protocol.json',p)
 dump(OUT/'preflight.json',{'passed':True,'trial_count':16,'unchanged_non_initial_config':True,'parent_ids':ids,'checkpoint':original['methods']['broad-pt-12000'],'script_sha256':sha(__file__)})
 print(json.dumps({'prepared':16,'output':str(OUT),'protocol_sha256':sha(OUT/'evaluation-protocol.json'),'parents':ids}),flush=True)

def evaluate(method,resume=False):
 import numpy as np
 from motion_policy_diagnostics import PrefixPolicy,audit
 from m5_evaluate import supplemental
 from supervisor.baseline.benchmark import run_episode
 protocol=json.loads((OUT/'evaluation-protocol.json').read_text());entry=protocol['methods'][method];checkpoint=Path(entry['checkpoint']);actualsha=sha(checkpoint)
 expected=entry.get('expected_sha256') or json.loads((OUT/('checkpoint-'+str(entry['step'])+'.json')).read_text())['sha256'];assert expected==actualsha
 folder=OUT/'evaluation'/method
 if resume:
  previous=json.loads((folder/'manifest.json').read_text());assert previous['checkpoint_sha256']==actualsha and previous['protocol_sha256']==sha(OUT/'evaluation-protocol.json');assert not (folder/'complete.json').exists()
 else:folder.mkdir(parents=True,exist_ok=False)
 manifest={'method':method,'protocol_sha256':sha(OUT/'evaluation-protocol.json'),'checkpoint_sha256':actualsha,'checkpoint':str(checkpoint),'eval_script_sha256':sha(__file__),'worker_sha256':sha(ROOT/'scripts/exp20261011_train_eval_worker.py'),'extra_guard':False}
 dump(folder/('resume-manifest.json' if resume else 'manifest.json'),manifest)
 if resume:dump(folder/'postprocess-recovery.json',{'reason':'First completed episode retained after dict.update summary assembly error; no policy/controller change','previous_manifest':previous,'new_manifest':manifest,'reused_episode_ids':[s['scenario_id'] for s in protocol['scenarios'] if (folder/s['scenario_id']/'episode.json').exists()]})
 class Policy(PrefixPolicy):
  def __init__(self):
   self.horizon=30;self.wrist='native';self.proprio=None;self.inferences=[]
   self.model_manifest={**manifest,'observation_version':'O2','domain_id':29,'model_rotation':'generic_interleaved','returned_rotation':'concatenated columns','H':30,'steps':10,'actor_truth':False}
   self.log=open(folder/'worker.log','a');self.process=subprocess.Popen([str(ROOT/'.venv-xvla/bin/python'),'scripts/exp20261011_train_eval_worker.py','--checkpoint',str(checkpoint),'--probe',entry['probe'],'--verification',str(folder/'reload-verification.json')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,cwd=ROOT,env=dict(os.environ,HF_HUB_OFFLINE='1'))
   try:self.load_info=self._receive(600);assert self.load_info.get('ready'),self.load_info
   except Exception:self.close();raise
 policy=Policy();rows=[]
 try:
  for s in protocol['scenarios']:
   dest=folder/s['scenario_id']
   if resume and (dest/'episode.json').exists():
    assert json.loads((dest/'scenario.json').read_text())==s and (dest/'raw_predictions.npz').exists()
    r=json.loads((dest/'episode.json').read_text());assert r['scenario_id']==s['scenario_id']
   else:
    r=run_episode(s,'pretrained',dest,policy,{'broad_pt':manifest,'model':policy.model_manifest},episode_setup=policy.instrument)
    if hasattr(policy,'telemetry'):policy.telemetry.close()
    np.savez_compressed(dest/'raw_predictions.npz',chunks=np.asarray(policy.raw_chunks),histories=np.asarray([np.full(20,np.nan) if h is None else h for h in policy.histories]))
   errors={k:r[k] for k in ['controller_exceptions','artifact_errors','evaluation_errors'] if r[k]}
   row={'id':s['scenario_id'],'source_demo_id':s['source_demo_id'],'seed':s['seed'],'method':method,'success':r['success'],'wall_seconds':r['wall_duration_s'],'errors':errors}
   if not errors:
    row.update({**audit(r,dest),**supplemental(r,dest)});goal=s['config']['task']['target_yaw_rad'];initialyaw=r['initial_truth']['object_yaw_rad'];initial_error=abs(float(np.degrees(np.arctan2(np.sin(initialyaw-goal),np.cos(initialyaw-goal)))))
    row.update(initial_yaw_error_deg=initial_error,final_yaw_error_deg=r['yaw_error_deg'],yaw_error_reduction_deg=initial_error-r['yaw_error_deg'],footprint_inside_target=r['evaluation']['criteria']['footprint_inside_target'],operational_placement=all(r['evaluation']['criteria'][k] for k in ['footprint_inside_target','marked_face_up','placement_height','released','stable']),evaluation_criteria=r['evaluation']['criteria'])
   rows.append(row);dump(folder/'summary.json',rows);print(json.dumps(row),flush=True)
 finally:policy.close()
 dump(folder/'complete.json',{'trials':len(rows),'protocol_sha256':manifest['protocol_sha256'],'completed':len(rows)==len(protocol['scenarios']),'runtime_errors':sum(bool(x['errors']) for x in rows)})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('op',choices=['prepare','evaluate']);p.add_argument('--method',choices=['broad-pt-12000'],default='broad-pt-12000');p.add_argument('--resume',action='store_true');a=p.parse_args()
 if a.op=='prepare':prepare()
 else:evaluate(a.method,a.resume)
