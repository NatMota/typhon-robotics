"""Post-hoc closed-loop source competence; copied native evaluator, no additional fit pass."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse,copy,gzip,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
OUT=Path('/var/tmp/typhon-research-20261011/broad-pt/source-competence')

def dump(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def prepare():
 p=OUT/'evaluation-protocol.json';assert not p.exists()
 data=ROOT/'frozen_results/personal-m3-20261009/data-canonical/dataset.json';manifest=json.loads(data.read_text());scenarios=[]
 names=['0001','0002','0003','0005','0012','0013']
 assert sorted(e['id'].replace('personal_demo_','') for e in manifest['episodes'] if e['split']=='P_train')==names
 for name in names:
  e=next(e for e in manifest['episodes'] if e['id']=='personal_demo_'+name);assert e['split']=='P_train'
  cfg=copy.deepcopy(e['config']);assert cfg['control']['max_steps']==1500
  scenarios.append({'scenario_id':'ptrain-'+name+'-seed7417','source_demo_id':e['source_demo_id'],'group':'post_hoc_source_competence','task':'feeding','seed':7417,'config':cfg,'attempt_limit':2})
 primary=OUT.parent;original=json.loads((primary/'evaluation-protocol.json').read_text());checkpoint=json.loads((primary/'checkpoint-12000.json').read_text())
 entry=copy.deepcopy(original['methods']['broad-pt-12000']);entry['expected_sha256']=checkpoint['sha256']
 protocol={**{k:v for k,v in original.items() if k not in ['created_unix_s','methods','scenarios','total_trials','seeds','statistical_status','selection']},'schema':'broad-pt-posthoc-source-competence-v1','created_unix_s':time.time(),'methods':{'broad-pt-12000':entry},'scenarios':scenarios,'total_trials':6,'seeds':[7417],'statistical_status':'Post-hoc diagnostic on all six exact source scene configurations, one fixed stochastic seed, no generalization claim','selection':'Frozen12k only, no tuning, no additional training, no checkpoint selection; source outcomes cannot overwrite primary18rollout results','reason':'Good expert-conditioned source fit with zero holdout grasps: test closed-loop execution on familiar starts','authorised_by':'root research scheduler before GPU calls','primary_protocol_sha256':sha(primary/'evaluation-protocol.json'),'source_manifest_sha256':sha(data),'original_evaluator_sha256':sha(ROOT/'scripts/exp20261011_train_eval.py'),'metrics_plus':['raw_close_commands','closest_executed_EEF_to_object_centre','closest_commanded_target_to_object_centre','bilateral_contact','requested_versus_applied_actions'],'fit_diagnostic':False}
 dump(p,protocol);print(json.dumps({'frozen_protocol_sha256':sha(p),'total_trials':6,'scenarios':[x['scenario_id'] for x in scenarios],'checkpoint_sha256':entry['expected_sha256'],'protocol_created_before_execution':True},indent=2))

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
 dump(folder/'complete.json',{'trials':len(rows),'protocol_sha256':manifest['protocol_sha256'],'completed':len(rows)==6,'runtime_errors':sum(bool(x['errors']) for x in rows)})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('op',choices=['prepare','evaluate']);p.add_argument('--method',choices=['broad-pt-12000'],default='broad-pt-12000');p.add_argument('--resume',action='store_true');a=p.parse_args()
 if a.op=='prepare':prepare()
 else:evaluate(a.method,a.resume)
