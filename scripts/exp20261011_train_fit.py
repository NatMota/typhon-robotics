"""Fixed-window teacher-forced action fit; diagnostics only, never checkpoint selection."""
import argparse, collections, hashlib, json, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/var/tmp/typhon-research-20261011/broad-pt')
DATA=ROOT/'frozen_results/personal-m3-20261009/data-canonical/dataset.json'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def prepare():
 p=OUT/'fit-protocol.json';assert not p.exists()
 m=json.loads(DATA.read_text());rows=[]
 for ei,e in enumerate(e for e in m['episodes'] if e['split'] in ['P_train','P_validation']):
  source=DATA.parent/e['path'];assert sha(source)==e['sha256']
  with np.load(source,allow_pickle=False) as a:
   ix=a['frame_action_indices'];target=a['targets'];valid=np.flatnonzero(ix+30<=len(target))
   trans=np.flatnonzero(np.diff(target[:,9])!=0)+1;assert len(trans)==2
   selected=set(valid[np.linspace(0,len(valid)-1,8,dtype=int)].tolist())
   for event in trans:
    for offset in [-8,0]:selected.add(int(valid[np.argmin(abs(ix[valid]-(event+offset)))]))
   for fi in sorted(selected):
    ai=int(ix[fi]);phase='pre_close' if ai<trans[0] else ('commanded_closed' if ai<trans[1] else 'post_open')
    rows.append({'id':e['id']+':frame'+str(fi),'episode':e['id'],'split':e['split'],'source_demo_id':e['source_demo_id'],'npz_path':str(source),'npz_sha256':e['sha256'],'frame':fi,'first_action':ai,'phase':phase,'instruction':e['instruction'],'forward_noise_seed':8517+1000*ei+fi,'generation_noise_seed':9517+1000*ei+fi})
 assert {r['episode'] for r in rows if r['split']=='P_train'}=={'personal_demo_'+n for n in ['0001','0002','0003','0005','0012','0013']}
 assert {r['episode'] for r in rows if r['split']=='P_validation'}=={'personal_demo_'+n for n in ['0006','0014','0016']}
 dump(p,{'created_unix_s':time.time(),'schema':'broad-pt-fixed-fit-v1','source_manifest_sha256':sha(DATA),'sample_selection':'Eight evenly spaced complete windows plus nearest valid starts at close/release and eight actions before each; unique frames, no outcomes used','methods':['narrow-m6-3000','broad-pt-3000','broad-pt-12000'],'windows':rows,'window_counts':dict(collections.Counter(r['split'] for r in rows)),'H':30,'steps':10,'teacher_forcing':'Expert frame and expert preceding-command proprio at each independent window; predictions never rolled forward','noise':'One native forward random noise/time seed and one generation seed per window, identical across methods','metrics':'First Panda arm only: mean/RMSE XYZ target distance mm; wrist SO(3) geodesic error degrees; gripper close probability MAE and >0.5 binary accuracy. Full native loss reported separately. These are EEF action errors, not object pose errors.','aggregation':'Window metrics -> equal-window mean per take -> equal-take mean per split. Train and holdout reported separately.','selection':'Evaluation only; both predetermined checkpoints reported without using this diagnostic for training or checkpoint choice','limitations':['Teacher forcing does not measure stability under self-induced states','Only three development holdout trajectories previously inspected','A single fixed noise draw per window; no statistical significance claim']})
 print(json.dumps({'counts':dict(collections.Counter(r['split'] for r in rows)),'total_windows':len(rows)}))

def rotations(a):
 b1=a[:,3:6];n=np.linalg.norm(b1,axis=1);assert np.all(n>1e-8);b1=b1/n[:,None]
 b2=a[:,6:9]-np.sum(b1*a[:,6:9],axis=1)[:,None]*b1;n=np.linalg.norm(b2,axis=1);assert np.all(n>1e-8);b2=b2/n[:,None]
 return np.stack([b1,b2,np.cross(b1,b2)],axis=2)
def metrics(pred,target):
 assert pred.shape==target.shape==(30,20) and np.isfinite(pred).all()
 d=np.linalg.norm(pred[:,:3]-target[:,:3],axis=1)*1000
 r=rotations(pred);t=rotations(target);cos=(np.einsum('nij,nij->n',r,t)-1)/2
 angle=np.degrees(np.arccos(np.clip(cos,-1,1)));binary=pred[:,9]>.5;truth=target[:,9]>.5
 return {'xyz_mean_mm':float(d.mean()),'xyz_rmse_mm':float(np.sqrt(np.mean(d*d))),'wrist_rotation_mean_deg':float(angle.mean()),'wrist_rotation_max_deg':float(angle.max()),'gripper_probability_mae':float(np.mean(abs(pred[:,9]-target[:,9]))),'gripper_binary_accuracy':float(np.mean(binary==truth)),'gripper_true_closed':int(truth.sum()),'gripper_predicted_closed':int(binary.sum()),'gripper_false_open':int(np.sum(truth&~binary)),'gripper_false_close':int(np.sum(~truth&binary))}

def run(policy,folder,method):
 protocol=json.loads((OUT/'fit-protocol.json').read_text());assert method in protocol['methods']
 output=folder/'teacher-forced';output.mkdir(exist_ok=False);allrows=[];raw=[];current=None;arrays=None;t0=time.perf_counter()
 for row in protocol['windows']:
  if current!=row['episode']:
   assert sha(row['npz_path'])==row['npz_sha256']
   with np.load(row['npz_path'],allow_pickle=False) as a:arrays={k:a[k] for k in ['primary','secondary','proprio','targets']}
   current=row['episode']
  fi=row['frame'];ai=row['first_action'];target=arrays['targets'][ai:ai+30]
  result=policy._request({'op':'diagnostic','images':[arrays['primary'][fi],arrays['secondary'][fi]],'instruction':row['instruction'],'proprio':arrays['proprio'][fi],'targets':target,'forward_noise_seed':row['forward_noise_seed'],'generation_noise_seed':row['generation_noise_seed']})
  pred=result.pop('actions');raw.append(pred)
  entry={**row,'method':method,**result,**metrics(pred,target)};allrows.append(entry)
  with (output/'windows.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
 np.savez_compressed(output/'predictions.npz',predictions=np.asarray(raw),sample_ids=np.asarray([r['id'] for r in allrows]))
 keys=['xyz_mean_mm','xyz_rmse_mm','wrist_rotation_mean_deg','gripper_probability_mae','gripper_binary_accuracy']
 losses=allrows[0]['native_forward_losses'];takes=[]
 for e in sorted(set(r['episode'] for r in allrows)):
  rs=[r for r in allrows if r['episode']==e]
  takes.append({'episode':e,'split':rs[0]['split'],'windows':len(rs),**{k:float(np.mean([r[k] for r in rs])) for k in keys},'native_forward_losses':{k:float(np.mean([r['native_forward_losses'][k] for r in rs])) for k in losses}})
 splits={}
 for split in ['P_train','P_validation']:
  rs=[r for r in takes if r['split']==split]
  splits[split]={'independent_takes':len(rs),**{k:float(np.mean([r[k] for r in rs])) for k in keys},'native_forward_losses':{k:float(np.mean([r['native_forward_losses'][k] for r in rs])) for k in losses}}
 dump(output/'summary.json',{'method':method,'protocol_sha256':sha(OUT/'fit-protocol.json'),'script_sha256':sha(__file__),'wall_seconds':time.perf_counter()-t0,'windows':len(allrows),'takes':takes,'splits':splits})
 return splits

def self_check():
 a=np.zeros((30,20));a[:,3]=a[:,7]=1;a[:,9]=1
 r=metrics(a,a);assert r['xyz_mean_mm']==0 and r['wrist_rotation_mean_deg']==0 and r['gripper_binary_accuracy']==1
 b=a.copy();b[:,0]=.01;b[:,3:9]=[0,1,0,-1,0,0];b[:,9]=0
 r=metrics(b,a);assert abs(r['xyz_mean_mm']-10)<1e-8 and abs(r['wrist_rotation_mean_deg']-90)<1e-8 and r['gripper_false_open']==30
 print('Action fit metric checks passed')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('op',choices=['prepare','test']);a=p.parse_args();prepare() if a.op=='prepare' else self_check()
