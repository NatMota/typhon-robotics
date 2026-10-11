"""12 offline chunks: two saved release states, two checkpoints, three explicit paired noises."""
import argparse, gzip, hashlib, json, os, pickle, shutil, subprocess, sys, time, types
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SEEDS = [61601, 61602, 61603]
STATES = ['v3-m6-film-demo_0002', 'v4-m6-film-demo_0006']
def read(p): return json.loads(Path(p).read_text())
def save(p, v): Path(p).write_text(json.dumps(v, indent=2, default=str)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
 return h.hexdigest()
def payload(p):
 with gzip.open(p, 'rb') as f: return pickle.load(f)
def prepare(root, out):
 out.mkdir(parents=True, exist_ok=False)
 (out/'noise').mkdir(); (out/'inputs').mkdir()
 shutil.copy2(__file__, out/'diagnostic-source.py')
 states=[]
 from scipy.spatial.transform import Rotation
 for state in STATES:
  folder=out/'inputs'/state; folder.mkdir()
  prior=root/'diagnostics-v4/v3-v4-existing-contrast'/state
  source=root/'benchmark-native'/state[:5]/state[6:]
  audit_path=(prior/'grasp-frame/grasp-frame-audit.json') if state.startswith('v3') else root/'diagnostics-v4/grasp-frame-comparison0006/grasp-frame-audit.json'
  grasp=read(audit_path); episode=read(source/'episode.json'); release=read(prior/'release/release-audit.json')
  query=episode['inference_measurements'][2]
  assert query['control_step']==60 and grasp['query_policy_s']==3.
  with gzip.open(source/'trajectory.jsonl.gz','rt') as f:
   rows=[json.loads(line) for line in f]
  truth=rows[59]['truth']; assert truth['control_steps']==grasp['query_step']==100
  event=next(e for e in release['events'] if e.get('query_index')==2 and e.get('query_measured_eef_quat_xyzw'))
  rotation=Rotation.from_quat(event['query_measured_eef_quat_xyzw']).as_matrix()
  objq=truth['object_quat_wxyz']; actual_obj=Rotation.from_quat(objq[1:]+objq[:1]).as_matrix()
  R_GO=np.array(grasp['measured_v4_grasp_matrix_R_GO'])
  frame_error=np.degrees(Rotation.from_matrix(rotation@R_GO@actual_obj.T).magnitude())
  assert frame_error<1e-4, frame_error
  shutil.copy2(prior/'release-input.pkl.gz',folder/'payload.pkl.gz')
  p=payload(folder/'payload.pkl.gz'); assert len(p['request']['anchor_times_s'])==30
  meta={'source':str(source),'query_index':2,'query_policy_s':3.,'measured_eef_position_m':truth['eef_position_m'],
   'measured_eef_quat_xyzw':event['query_measured_eef_quat_xyzw'],'grasp_matrix_R_GO':R_GO.tolist(),
   'grasp_source':str(audit_path),'grasp_source_sha256':sha(audit_path),'frame_composition_error_deg':frame_error,
   'target_yaw_rad':read(source/'scenario.json')['config']['task']['target_yaw_rad'],
   'original_first_open_waypoint':grasp['first_open_waypoint'], 'grasp_vs_expert_rotation_deg':grasp['grasp_transform_difference_deg'],
   'processor_reference':str(root/'diagnostics-v4/v3-v4-existing-contrast/release-processor/inputs'/state/'actual-processor-inputs.npz')}
  save(folder/'metadata.json',meta)
  states.append({'id':state,'payload_sha256':sha(folder/'payload.pkl.gz')})
 save(out/'protocol.json',{'schema':'same-input-v3-v4-offline-v1','states':states,'checkpoints':['v3-m6','v4-m6'],
  'seeds':SEEDS,'chunks':12,'steps':10,'physical_rollouts':0,'new_training':False,'harness':False,
  'noise':'Three explicitly saved float32 CUDA Gaussian tensors, shared across checkpoints and states; exact one randn replacement in a diagnostic-local function globals copy.',
  'model_inputs':'Unmodified archived RGB/history/instruction/proprio/held annotation; actual processor tensors verified against archived reference and across checkpoints.',
  'scoring':'Measured state-specific object-to-gripper transform only for offline rotation scoring. Fixed grasp is not a dynamics prediction; release/final waypoint focus.',
  'code_sha256':sha(__file__),'source_hashes':{f:sha(REPO/f) for f in ['supervisor/adaptation/m7_training.py','supervisor/adaptation/m7_eval_worker.py']}})
class ExplicitNoiseTorch:
 """Local globals proxy; does not mutate torch or production generation code."""
 def __init__(self, torch, noise): self.torch=torch; self.noise=noise; self.calls=0
 def __getattr__(self,k): return getattr(self.torch,k)
 def randn(self,*shape,**kwargs):
  assert tuple(shape)==tuple(self.noise.shape) and set(kwargs)=={'device'}
  assert self.torch.device(kwargs['device'])==self.noise.device
  self.calls+=1; assert self.calls==1
  return self.noise.clone()
def infer(root,out,name):
 import torch
 from transformers import AutoProcessor
 from supervisor.adaptation.m7_training import load_checkpoint,generate_requested,HISTORY_AUDIT
 from supervisor.adaptation.m7_eval_worker import prepare_inputs
 from supervisor.adaptation.m6_contract import reorder
 checkpoint=Path('/var/tmp/humanoid-m7-checkpoints-20261009')/(name+'.pt')
 checksum=sha(checkpoint); assert checksum==read(root/'suite.json')['checkpoint_sha256'][name]
 start=time.perf_counter()
 model,projection,restored=load_checkpoint(checkpoint,REPO/'.cache-xvla/checkpoint')
 config=restored['config']; assert config['condition']=='crosshair' and config['history_mode']=='sparse'
 assert restored['domain_id']==29
 model.eval();projection.eval()
 dest=out/name; dest.mkdir()
 save(dest/'load.json',{'checkpoint':str(checkpoint),'sha256':checksum,'config':config,'step':restored['step'],
  'load_s':time.perf_counter()-start,'strict_load':True,'device':torch.cuda.get_device_name(0),'torch':str(torch.__version__)})
 del restored
 processor=AutoProcessor.from_pretrained(REPO/'.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True)
 processor.language_max_length=256
 noises={}
 for seed in SEEDS:
  nf=out/'noise'/('seed'+str(seed)+'.npy')
  if not nf.exists():
   g=torch.Generator(device='cuda').manual_seed(seed)
   np.save(nf,torch.randn(1,30,20,device='cuda',generator=g).cpu().numpy())
  noises[seed]=torch.from_numpy(np.load(nf)).to('cuda')
 for state in STATES:
  sd=dest/state; sd.mkdir(); packet=payload(out/'inputs'/state/'payload.pkl.gz')
  inputs,held,valid,annotation=prepare_inputs(processor,packet,config,'cuda')
  arrays={k:v.cpu().numpy() for k,v in inputs.items()}
  reference=np.load(read(out/'inputs'/state/'metadata.json')['processor_reference'])
  assert set(arrays)==set(reference.files)
  for k,a in arrays.items(): np.testing.assert_array_equal(a,reference[k])
  if name=='v4-m6':
   other=np.load(out/'v3-m6'/state/'actual-processor-inputs.npz')
   for k,a in arrays.items(): np.testing.assert_array_equal(a,other[k])
  np.savez_compressed(sd/'actual-processor-inputs.npz',**arrays)
  for seed in SEEDS:
   response_dir=sd/('seed'+str(seed)); response_dir.mkdir()
   torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);HISTORY_AUDIT.clear()
   proxy=ExplicitNoiseTorch(torch,noises[seed])
   local_fn=types.FunctionType(generate_requested.__code__,dict(generate_requested.__globals__,torch=proxy),generate_requested.__name__,generate_requested.__defaults__,generate_requested.__closure__)
   assert local_fn.__code__ is generate_requested.__code__
   torch.cuda.synchronize(); began=time.perf_counter()
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    timed=local_fn(model,projection,inputs,packet['request'],held,valid,steps=10)
   torch.cuda.synchronize(); elapsed=time.perf_counter()-began
   assert proxy.calls==1
   actions=reorder(np.array([w['target_generic_interleaved'] for w in timed['waypoints']]),True)
   assert actions.shape==(30,20) and np.isfinite(actions).all()
   np.savez_compressed(response_dir/'actions.npz',actions=actions)
   save(response_dir/'response.json',{'timed':timed,'inference_s':elapsed,'history_audit':list(HISTORY_AUDIT),
    'annotation':annotation,'noise_sha256':sha(out/'noise'/('seed'+str(seed)+'.npy')),'noise_replacements':proxy.calls,
    'raw_payload_sha256':sha(out/'inputs'/state/'payload.pkl.gz'),'model_input_tensor_sha256':sha(sd/'actual-processor-inputs.npz')})
   print(json.dumps({'finished':name,'state':state,'seed':seed,'inference_s':elapsed}),flush=True)
 save(dest/'complete.json',{'chunks':6,'reference_processor_arrays_equal':True,'paired_noise_saved_and_injected':True})
def score(root,out):
 import math,csv
 from scipy.spatial.transform import Rotation
 from v4_prompt_probes_20261011 import matrix,describe
 rows=[]
 for name in ['v3-m6','v4-m6']:
  assert read(out/name/'complete.json')['chunks']==6
  for state in STATES:
   meta=read(out/'inputs'/state/'metadata.json');p=payload(out/'inputs'/state/'payload.pkl.gz')
   grasp=np.array(meta['grasp_matrix_R_GO']);goal=Rotation.from_euler('z',meta['target_yaw_rad']).as_matrix()
   for seed in SEEDS:
    folder=out/name/state/('seed'+str(seed)); actions=np.load(folder/'actions.npz')['actions']
    metrics=describe(actions,meta,p)
    for i,a in enumerate(actions):
     obj=matrix(a)@grasp; gap=math.atan2(obj[1,0],obj[0,0])-meta['target_yaw_rad']
     metrics[i].update(time_s=p['request']['anchor_times_s'][i],
      implied_yaw_error_deg=abs(math.degrees(math.atan2(math.sin(gap),math.cos(gap)))),
      implied_full_rotation_error_deg=math.degrees(Rotation.from_matrix(obj@goal.T).magnitude()),
      xyz_displacement_from_measured_mm=(1000*(a[:3]-meta['measured_eef_position_m'])).tolist())
    opening=next((i for i,a in enumerate(actions) if a[9]<=.5),None)
    selected={'first_opening':None if opening is None else metrics[opening],'final':metrics[-1]}
    save(folder/'scored.json',{'action_metrics':metrics,'selected_metrics':selected,'executed':False,
     'limitation':'Fixed measured grasp rotation; release-time/final raw commands, not physically measured outcomes.'})
    row={'checkpoint':name,'state':state,'seed':seed,'first_open_waypoint':opening,
     'first_open_time_s':None if opening is None else metrics[opening]['time_s'],
     'opening_yaw_error_deg':None if opening is None else metrics[opening]['implied_yaw_error_deg'],
     'opening_full_rotation_error_deg':None if opening is None else metrics[opening]['implied_full_rotation_error_deg'],
     'final_yaw_error_deg':metrics[-1]['implied_yaw_error_deg'],'final_full_rotation_error_deg':metrics[-1]['implied_full_rotation_error_deg'],
     'final_wrist_yaw_deg':metrics[-1]['projected_wrist_x_yaw_deg'],
     'final_xyz_displacement_mm':metrics[-1]['xyz_displacement_from_measured_mm'],
     'opening_xyz_displacement_mm':None if opening is None else metrics[opening]['xyz_displacement_from_measured_mm']}
    rows.append(row)
 with (out/'summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 groups=[]
 for state in STATES:
  for name in ['v3-m6','v4-m6']:
   rr=[r for r in rows if r['state']==state and r['checkpoint']==name]
   g={'state':state,'checkpoint':name,'samples':len(rr)}
   for k in ['first_open_time_s','opening_yaw_error_deg','opening_full_rotation_error_deg','final_yaw_error_deg','final_full_rotation_error_deg','final_wrist_yaw_deg']:
    v=[r[k] for r in rr if r[k] is not None];g[k]={'mean':float(np.mean(v)),'min':min(v),'max':max(v)} if v else None
   groups.append(g)
 save(out/'complete.json',{'chunks':12,'physical_rollouts':0,'same_processor_tensors_across_checkpoints':True,
  'noise_tensor_checksums':{str(s):sha(out/'noise'/('seed'+str(s)+'.npy')) for s in SEEDS},'groups':groups})
 print(json.dumps(groups,indent=2),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--infer',choices=['v3-m6','v4-m6']);p.add_argument('--score-only',action='store_true');a=p.parse_args()
 os.chdir(REPO)
 if a.infer: infer(a.root,a.output,a.infer);return
 if a.score_only:score(a.root,a.output);return
 prepare(a.root,a.output)
 for name in ['v3-m6','v4-m6']:
  with (a.output/(name+'-worker.log')).open('w') as log:
   subprocess.run([str(REPO/'.venv-xvla/bin/python'),str(Path(__file__).resolve()),'--root',str(a.root),'--output',str(a.output),'--infer',name],
    stdout=log,stderr=log,check=True,env=dict(os.environ,HF_HOME=str(REPO/'.cache-xvla/hf'),HF_HUB_OFFLINE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
  print(name+' complete',flush=True)
 score(a.root,a.output)
if __name__=='__main__':main()
