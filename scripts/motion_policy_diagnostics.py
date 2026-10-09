"""Bounded paired inference diagnostics with explicit execution contracts."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys,json,copy,argparse,hashlib,gzip
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation
from supervisor.adaptation.policy import AdaptedPolicy
from supervisor.baseline.benchmark import run_episode
from supervisor.feeding_loading.environment import FeedingLoadingEnv
FROZEN=Path('frozen_results');OUT=Path('results/motion-diagnostics-20261009')
def prepare_prefix(actions,horizon,history,wrist='native'):
 a=np.asarray(actions).copy();assert a.shape==(30,20) and 1<=horizon<=30
 a=a[:horizon].copy()
 if wrist=='downward-yaw':
  base=FeedingLoadingEnv.wrist_orientation(0)
  for row in a:
   b1=row[3:6]/np.linalg.norm(row[3:6]);b2=row[6:9]-np.dot(b1,row[6:9])*b1;b2/=np.linalg.norm(b2);r=np.column_stack((b1,b2,np.cross(b1,b2)))
   relative=r@base.T;yaw=np.arctan2(relative[1,0]-relative[0,1],relative[0,0]+relative[1,1]);r=FeedingLoadingEnv.wrist_orientation(yaw)
   row[3:6]=r[:,0];row[6:9]=r[:,1]
 history[:9]=a[-1,:9]  # last EXECUTED prefix target, never discarded batch end
 return a
class PrefixPolicy(AdaptedPolicy):
 def __init__(self,horizon,wrist,*args,**kw):
  self.horizon=horizon;self.wrist=wrist;super().__init__(*args,**kw)
  self.model_manifest.update(execute_actions_per_chunk=horizon,execution_diagnostic=True,wrist_projection=wrist,proprio_history='last executed prefix target; baseline previous-command convention')
 def reset(self,seed):
  super().reset(seed);self.raw_chunks=[];self.histories=[];self.pending_target=None
 def predict(self,obs):
  before=None if self.proprio is None else self.proprio.copy()
  raw=super().predict(obs);self.raw_chunks.append(raw.copy());self.histories.append(before)
  return prepare_prefix(raw,self.horizon,self.proprio,self.wrist)
 def to_action(self,target,obs,config):
  self.pending_target=np.asarray(target).copy()
  return super().to_action(target,obs,config)
 def instrument(self,env,out):
  from motion_pose_diagnostics import details
  self.telemetry=gzip.open(out/'controller_trace.jsonl.gz','wt');step=env.step
  def wrapped(command):
   step(command)
   if self.pending_target is not None:
    self.telemetry.write(json.dumps(details(env,self.pending_target))+'\n');self.pending_target=None
  env.step=wrapped

def panel():
 original=json.loads((FROZEN/'personal-m3-20261009/protocol/scenarios.json').read_text())['primary'];recovery=json.loads((FROZEN/'personal-m4-20261009/protocol/scenarios.json').read_text())['primary'];data=json.loads((FROZEN/'personal-m3-20261009/data-canonical/dataset.json').read_text())
 personal=next(e for e in data['episodes'] if e['id']=='personal_demo_0001')
 source={'scenario_id':'diagnostic-source-personal0001','group':'training_configuration_diagnostic','task':'feeding','seed':7317,'config':copy.deepcopy(personal['config']),'attempt_limit':2,'source_demo_id':personal['id'],'evaluation_status':'seen training configuration; not held-out evidence'}
 return [source,next(s for s in original if s['scenario_id']=='feeding-standard-01')]+[next(s for s in recovery if s['scenario_id']=='recovery-'+suffix) for suffix in ['wrong_orientation-01','already_completed-01','already_completed-03']]
def audit(row,folder):
 from supervisor.feeding_loading.evaluation import score_episode
 sc=json.loads((folder/'scenario.json').read_text());cfg=copy.deepcopy(sc['config'])
 score=score_episode(row['initial_truth'],row['final_truth'],row['stability_truths'],cfg,row['attempts']);assert score['success']==row['success']
 records=[json.loads(s) for s in gzip.open(folder/'trajectory.jsonl.gz','rt')];initial=np.array(row['initial_truth']['object_position_m']);xyz=np.array([row['initial_truth']['eef_position_m']]+[r['truth']['eef_position_m'] for r in records])
 return {'success':row['success'],'position_error_mm':row['position_error_mm'],'yaw_error_deg':row['yaw_error_deg'],'contact_steps':sum(r['truth']['grasp_contact'] for r in records),'bilateral_contact_steps':sum(r['truth']['bilateral_grasp_contact'] for r in records),'max_object_displacement_mm':max(float(np.linalg.norm(np.array(r['truth']['object_position_m'])-initial)*1000) for r in records),'eef_path_m':float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()),'closing_events':row['grasp_attempts'],'final_criteria':row['evaluation']['criteria'],'audit_passed':True}
def main(args):
 scenarios=panel();folder=OUT/'policy'/f'{args.model}-h{args.horizon}-{args.wrist}';folder.mkdir(parents=True,exist_ok=False)
 checkpoint=FROZEN/('personal-m3-20261009/training/P17/prompt.pt' if args.model=='P17' else 'personal-m4-20261009/training/PR/prompt.pt')
 manifest={'model':args.model,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'horizon':args.horizon,'wrist':args.wrist,'baseline_commit':'b504eb0deda28f0efed773781c426c916af04cc5','panel':scenarios,'no_training':True,'no_completion_guard':True,'diagnostic_not_frozen_evaluation':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');policy=PrefixPolicy(args.horizon,args.wrist,'O2',checkpoint,folder/'worker.log');rows=[]
 try:
  for s in scenarios:
   dest=folder/s['scenario_id'];r=run_episode(s,'pretrained',dest,policy,{'diagnostic':manifest,'model':policy.model_manifest},episode_setup=policy.instrument);assert not (r['controller_exceptions'] or r['artifact_errors'] or r['evaluation_errors']),r['controller_exceptions']
   policy.telemetry.close()
   np.savez_compressed(dest/'raw_predictions.npz',chunks=np.asarray(policy.raw_chunks),histories=np.asarray([np.full(20,np.nan) if h is None else h for h in policy.histories]))
   metrics={'id':s['scenario_id'],**audit(r,dest)};rows.append(metrics);(folder/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(args.model,args.horizon,args.wrist,metrics,flush=True)
 finally:policy.close()
def test():
 a=np.zeros((30,20));a[:,:3]=np.arange(30)[:,None];base=FeedingLoadingEnv.wrist_orientation(.7);a[:,3:6]=base[:,0];a[:,6:9]=base[:,1];h=np.zeros(20)
 b=prepare_prefix(a,5,h);assert b.shape==(5,20);np.testing.assert_array_equal(h[:9],a[4,:9]);assert h[0]!=a[-1,0]
 b=prepare_prefix(a,30,h);np.testing.assert_array_equal(b,a);np.testing.assert_array_equal(h[:9],a[-1,:9])
 b=prepare_prefix(a,10,h,'downward-yaw');np.testing.assert_allclose(b[:,3:9],a[:10,3:9],atol=1e-12)
 print('Prefix/history and yaw-preservation self-checks passed')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['P17','PR']);p.add_argument('--horizon',type=int,default=30);p.add_argument('--wrist',choices=['native','downward-yaw'],default='native');p.add_argument('--test',action='store_true');a=p.parse_args()
 if a.test:test()
 else:main(a)
