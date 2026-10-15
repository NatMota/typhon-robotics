import os,sys,json,time,hashlib
from pathlib import Path
W=Path(__file__).resolve().parents[1];os.chdir(W);sys.path[:0]=[str(W),str(W/'.m5-deps'),str(W/'.m6-deps')]
import torch,numpy as np
from supervisor.adaptation.m6_contract import reorder
O=Path('/var/tmp/typhon-init03-20261015');D=Path('/var/tmp/typhon-init02-20261015')
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def rot(a):
 v=a[...,3:9].reshape(-1,3,2).astype('float64');x=v[:,:,0];x/=np.linalg.norm(x,axis=1)[:,None];y=v[:,:,1];y-=np.sum(x*y,axis=1)[:,None]*x;y/=np.linalg.norm(y,axis=1)[:,None];return np.stack([x,y,np.cross(x,y)],axis=-1)
def metrics(a,b,prop,n):
 a=a[:n];b=b[:n];diff=rot(a)@np.swapaxes(rot(b),1,2)
 deg=np.degrees(np.arccos(np.clip((np.trace(diff,axis1=1,axis2=2)-1)/2,-1,1)))
 return {'xyz_rmse_m':float(np.sqrt(np.mean((a[:,:3]-b[:,:3])**2))),'xyz_l2_mean_m':float(np.linalg.norm(a[:,:3]-b[:,:3],axis=1).mean()),'rotation_geodesic_mean_deg':float(deg.mean()),'gripper_probability_mae':float(np.abs(a[:,9]-b[:,9]).mean()),'predicted_closes':int(np.sum(a[:,9]>.5)),'target_closes':int(np.sum(b[:,9]>.5)),'predicted_gripper_mean':float(a[:,9].mean()),'wrist_displacement_from_initial_mean_m':float(np.linalg.norm(a[:,:3]-prop[:3],axis=1).mean())}
def main():
 q=json.loads((O/'packet-qualification.json').read_text());assert q['passed']
 assert all(all(v.values()) for r in q['audit'] for v in r['runtime_matches'].values())
 packets=torch.load(O/'packets.pt',weights_only=False,map_location='cpu');rows=[];sources=[];hashes={}
 for model in ['R','T']:
  for parent in ['demo_0001','demo_0002','demo_0003','demo_0005']:
   for seed in [912171,912172]:
    noises=[]
    for task in ['loading','feeding']:
     folder=D/'episodes'/model/f'init-{parent}-{task}-{seed}'
     pp=list(folder.rglob('predictions.npz'));assert len(pp)==1,(folder,pp)
     with np.load(pp[0]) as z:
      assert 'chunks' in z.files,z.files
      pred=reorder(z['chunks'][0]);assert pred.shape==(30,20)
     npth=folder/'input-audit/noise-000.npz'
     with np.load(npth) as z:noises.append(z['draw_0'].copy())
     for label in ['loading','feeding']:
      target=packets[parent+'-'+label]['action'][0].numpy();prop=packets[parent+'-'+label]['proprio'][0].numpy()
      for prefix in [1,10,30]:rows.append({'model':model,'parent':parent,'seed':seed,'instruction':task,'label':label,'prefix':prefix,**metrics(pred,target,prop,prefix)})
     for p in [pp[0],npth,folder/'input-audit/initial-prepared.npz']:hashes[str(p)]=sha(p)
     sources.append({'model':model,'parent':parent,'seed':seed,'instruction':task,'predictions':str(pp[0]),'noise':str(npth),'prepared':str(folder/'input-audit/initial-prepared.npz')})
    assert np.array_equal(*noises)
 result={'completed_epoch':time.time(),'distinct_archived_chunks':len(sources),'new_generation_calls':0,'new_physics_episodes':0,'rows':rows,'sources':sources,'source_sha256':hashes,'source_packet_qualification_sha256':sha(O/'packet-qualification.json'),'native_loss_matrix':{'status':'blocked','reason':'native objective instrumentation was not qualified by the predeclared T+12 deadline; no surrogate loss substituted','scored_cells':0},'fresh_native_generation_reproduction':'not performed in this diagnostic; archival-only analysis; INIT02 prior fresh reload probes already passed','label_limit':'both labels are open throughout the first 30 commands; no inference about closing or release selection'}
 dump(O/'archived-part-A.json',result)
 counts={}
 for model in ['R','T']:
  counts[model]={}
  for pref in [1,10,30]:
   x=[r for r in rows if r['model']==model and r['prefix']==pref]
   same=0;hold=0;load=0
   for task in ['feeding','loading']:
    for parent in ['demo_0001','demo_0002','demo_0003','demo_0005']:
     for seed in [912171,912172]:
      rr=[r for r in x if r['parent']==parent and r['seed']==seed and r['instruction']==task]
      best=min(rr,key=lambda r:r['xyz_rmse_m'])['label'];same+=best==task;hold+=best=='feeding';load+=best=='loading'
   counts[model][str(pref)]={'matched_target_xyz_preference':same,'hold_preference':hold,'loading_preference':load,'total':16}
 dump(O/'part-A-summary.json',{'counts':counts,'completed_epoch':result['completed_epoch'],'source_rows':len(rows),'fresh_generation_validation_missing':True})
 print(json.dumps({'counts':counts,'epoch':time.time()}),flush=True)
if __name__=='__main__':main()
