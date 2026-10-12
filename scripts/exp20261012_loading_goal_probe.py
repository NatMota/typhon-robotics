"""Fixed-camera, fixed-proprio supplied-goal diagnostic; no actions executed."""
import os,sys,json,gzip,copy,subprocess,time,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from supervisor.baseline.xvla import XvlaPolicy,task_prompt
OUT=Path('/var/tmp/typhon-research-20261012/loading/goal-probe')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
class Worker(XvlaPolicy):
 def __init__(self,entry):
  self.log=open(OUT/'worker.log','a')
  self.process=subprocess.Popen([str(ROOT/'.venv-xvla/bin/python'),'scripts/exp20261011_train_eval_worker.py','--checkpoint',entry['checkpoint'],'--probe',entry['probe'],'--verification',str(OUT/'reload-verification.json')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,cwd=ROOT,env=dict(os.environ,HF_HUB_OFFLINE='1'))
  self.load_info=self._receive(600);assert self.load_info['ready']

def main():
 OUT.mkdir(parents=True,exist_ok=False)
 run=Path('/var/tmp/typhon-research-20261012/loading/factorial/evaluation/broad-pt-12000/b01-familiar-loading-seed7417')
 io=np.load(run/'policy_io.npz');raw=np.load(run/'raw_predictions.npz');episode=json.loads((run/'episode.json').read_text());sc=json.loads((run/'scenario.json').read_text())
 states={r['step']:r['truth'] for r in map(json.loads,gzip.open(run/'trajectory.jsonl.gz','rt'))}
 held=[i for i in range(1,len(io['primary'])) if states[40+30*i]['bilateral_grasp_contact'] and states[40+30*i]['object_position_m'][2]-episode['initial_truth']['object_position_m'][2]>.01]
 assert len(held)>=2
 selected=[('approach',0),('first_held',held[0]),('last_held_chunk',held[-1])]
 tasks={'feeding':copy.deepcopy(sc['config']['task']),'loading':copy.deepcopy(sc['config']['task'])}
 tasks['feeding'].update(type='feeding',target_position_xy_m=[.125,-.12],target_region_size_m=.1,instruction='Grasp the orange block, orient its blue 2, and place it in the staging square.')
 protocol=json.loads(Path('/var/tmp/typhon-research-20261012/loading/factorial/evaluation-protocol.json').read_text());entry=protocol['methods']['broad-pt-12000']
 plan={'schema':'typhon-goal-only-probe-v1','source_episode':str(run),'source_io_sha256':sha(run/'policy_io.npz'),'checkpoint_sha256':entry['expected_sha256'],'states':selected,'instruction_conditions':{k:task_prompt(t) for k,t in tasks.items()},'noise_seeds':[51001,51002,51003],'queries':18,'physical_execution':False,'fixed':'Identical recorded RGB and native proprio for each instruction pair; only supplied goal text changes. Scorer truth selects diagnostic times, never policy features.','status':'Offline development; target shift270.046mm; endpoint-command response is not task success'}
 dump(OUT/'protocol.json',plan);assert sha(entry['checkpoint'])==entry['expected_sha256']
 worker=Worker(entry);rows=[];arrays={}
 try:
  for label,i in selected:
   hist=raw['histories'][i]
   if np.isnan(hist).all():
    rot=Rotation.from_quat(io['eef_quat_xyzw'][i]).as_matrix();hist=np.r_[io['eef_position_m'][i],rot[:,0],rot[:,1],0.,np.zeros(10)].astype(np.float32)
   assert np.isfinite(hist).all()
   np.savez_compressed(OUT/f'input-{label}.npz',primary=io['primary'][i],secondary=io['secondary'][i],proprio=hist)
   for seed in plan['noise_seeds']:
    predictions={}
    for kind in ['feeding','loading']:
     worker._request({'op':'reset','seed':seed})
     result=worker._request({'op':'predict','images':[io['primary'][i],io['secondary'][i]],'instruction':task_prompt(tasks[kind]),'proprio':hist.copy()})
     pred=result['actions'];predictions[kind]=pred;arrays[f'{label}_{seed}_{kind}']=pred
    delta=predictions['loading'][-1,:3]-predictions['feeding'][-1,:3]
    rows.append({'state':label,'source_chunk':i,'noise_seed':seed,'endpoint_shift_xyz_mm':(delta*1000).tolist(),'endpoint_shift_norm_mm':float(np.linalg.norm(delta)*1000),'feeding_endpoint':predictions['feeding'][-1,:3].tolist(),'loading_endpoint':predictions['loading'][-1,:3].tolist(),'grip_max_abs_change':float(np.max(np.abs(predictions['loading'][:,9]-predictions['feeding'][:,9]))),'requested_goal_delta_xy_mm':[-5.,270.],'max_action_abs_change':float(np.max(np.abs(predictions['loading']-predictions['feeding'])))})
    dump(OUT/'summary.json',rows);print(json.dumps(rows[-1]),flush=True)
 finally:worker.close()
 np.savez_compressed(OUT/'predictions.npz',**arrays)
 dump(OUT/'complete.json',{'complete':len(rows)==9,'paired_queries':len(rows),'total_queries':2*len(rows),'source_input_unchanged':sha(run/'policy_io.npz')==plan['source_io_sha256'],'no_physical_rollout':True})
if __name__=='__main__':main()
