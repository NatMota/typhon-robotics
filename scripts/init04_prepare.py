"""INIT04: exact saved-state onset audit; no model calls or physics."""
import os,sys,json,gzip,hashlib,time,traceback,shutil
from pathlib import Path
W=Path(__file__).resolve().parents[1];os.chdir(W);sys.path[:0]=[str(W),str(W/'scripts'),str(W/'.m5-deps'),str(W/'.m6-deps')]
import numpy as np,torch
from transformers import AutoProcessor
from supervisor.adaptation.observations import views
from supervisor.adaptation.m6_contract import reorder
from load04_common import array_sha
O=Path('/var/tmp/typhon-init04-20261015');D=Path('/var/tmp/typhon-init02-20261015')
H="Maintain the currently supplied end-effector position and orientation. Keep the gripper open. Do not approach, touch, grasp or move the block."
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def lines(p):
 with gzip.open(p,'rt')as f:return [json.loads(x)for x in f]
def first(a):
 idx=np.flatnonzero(a);return int(idx[0]) if len(idx)else None
def event(i,ep):
 if i is None:return None
 return {'executed_action_index':i,'control_step':ep['control_step_start']+i+1,'seconds_since_request':(i+1)/20,'query':i//30,'waypoint':i%30}
def contact(r):
 return bool(r['object_truth']['grasp_contact']) or any('gripper' in str(c) and 'cube' in str(c)for c in r['contacts'])
def prepare(proc,io,hist,q,text):
 b=dict(proc(images=views({'primary':io['primary'][q],'secondary':io['secondary'][q]},'O2'),language_instruction=text))
 b.update(proprio=reorder(torch.from_numpy(hist[q].copy()).unsqueeze(0)),domain_id=torch.tensor([29]))
 return b
def main():
 clock=read(O/'clock.json');assert time.time()<clock['preparation_deadline']
 bank=read(D/'data/qualified-dataset.json');rows=[];source_hashes={};io_cache={};hist_cache={};hold_cache={}
 proc=AutoProcessor.from_pretrained(W/'.cache-xvla/checkpoint',trust_remote_code=True,local_files_only=True);proc.language_max_length=128
 for parent in ['demo_0001','demo_0002','demo_0003','demo_0005']:
  hold=next(e for e in bank['episodes']if e['id']==parent+'-F-preserve')
  with np.load(hold['absolute_path'])as z:
   holdprop=reorder(z['proprio'].copy())
   holdimgs=np.stack([proc(images=views({k:z[v][i]for k,v in hold['views'].items()},'O2'),language_instruction=hold['instruction'])['image_input'][0].numpy()for i in range(len(holdprop))])
   holdinfo={'source':hold['absolute_path'],'sha256':sha(hold['absolute_path']),'actions':len(z['targets']),'observation_frames':len(holdprop),'frame_action_indices':z['frame_action_indices'].tolist(),'duration_s':len(z['targets'])/20,'position_range_m':np.ptp(holdprop[:,:3],axis=0).tolist()}
  for seed in [912171,912172]:
   folder=D/'episodes/T'/f'init-{parent}-feeding-{seed}'
   ep=read(folder/'episode.json');summary=read(folder/'summary.json');tr=lines(folder/'controller_trace.jsonl.gz')
   assert ep['model_sha256']=='4d7a4a2547f084b6026164f27a6b7d43b5cb4d3b83ee4875343d108908f35636'
   assert len(tr)==1460 and all(r['step']==ep['control_step_start']+i+1 for i,r in enumerate(tr))
   with np.load(folder/'predictions.npz')as z:chunks=z['chunks'].copy();hist=z['histories'].copy()
   assert chunks.shape==(48,30,20)and hist.shape==(48,20)
   for q in range(1,48):np.testing.assert_array_equal(hist[q,:9],chunks[q-1,-1,:9])
   targets=np.array([r['target']for r in tr[:1436]]);np.testing.assert_array_equal(targets,chunks.reshape(-1,20)[:1436])
   close=targets[:,9]>.5;executed=np.array([r['command'][-1]>0 for r in tr[:1436]])
   assert np.array_equal(close,executed),'Executed closing differs from native active-arm threshold'
   touch=np.array([contact(r)for r in tr]);z0=ep['initial_truth']['object_position_m'][2]
   supported=np.array([r['object_truth']['bilateral_grasp_contact'] and r['object_truth']['object_position_m'][2]-z0>=.01 for r in tr])
   lift=next((i for i in range(len(supported)-2)if supported[i:i+3].all()),None)
   assert (lift is not None)==summary['supported_lift_10mm_3steps']
   close_i=first(close);touch_i=first(touch);evs=[i for i in [close_i,touch_i]if i is not None];qe=min(evs)//30 if evs else None
   upto=qe if qe is not None else 47
   with np.load(folder/'policy_io.npz')as z:io={k:z[k].copy() for k in ['primary','secondary','eef_position_m','eef_quat_xyzw']}
   inference=read(folder/'inference.json');queries=[];first_ins=read(folder/'input-audit/input-000.json')['instruction'];request_defects=[]
   for q in range(upto+1):
    au=read(folder/'input-audit'/f'input-{q:03d}.json');b=prepare(proc,io,hist,q,au['instruction']);hashes={k:array_sha(v.numpy())for k,v in b.items()}
    assert hashes==au['input_hashes'],('Actual saved tensor reconstruction failed',parent,seed,q)
    fulltok=proc.tokenizer(au['instruction'])['input_ids'];assert len(fulltok)<=128
    decoded=proc.tokenizer.decode(b['input_ids'][0],skip_special_tokens=True)
    if au['instruction']!=first_ins:request_defects.append({'q':q,'instruction':au['instruction']})
    obj=np.array(ep['initial_truth']['object_position_m'] if q==0 else tr[q*30-1]['object_truth']['object_position_m'])
    measured=io['eef_position_m'][q];supplied=hist[q,:3];pred=chunks[q,:,:3]
    xyz_to_supplied=np.linalg.norm(pred-supplied,axis=1);xyz_to_measured=np.linalg.norm(pred-measured,axis=1)
    img=b['image_input'][0].numpy();delta=holdimgs-img[None];image_dist=np.sqrt(np.mean(delta*delta,axis=(1,2,3,4)));del delta
    hd=np.linalg.norm(holdprop[:,:3]-supplied,axis=1)
    queries.append({'query':q,'instruction':au['instruction'],'token_hash':hashes['input_ids'],'decoded_instruction':decoded,'token_count_untruncated':len(fulltok),'input_hashes':hashes,'input_shapes':{k:list(v.shape)for k,v in b.items()},'effective_mask':b['image_mask'].tolist(),'crop':inference[q]['crop_metadata'],'supplied_xyz':supplied.tolist(),'measured_xyz':measured.tolist(),'object_xyz_offline_only':obj.tolist(),'supplied_measured_gap_mm':float(np.linalg.norm(supplied-measured)*1000),'first_target_xyz':pred[0].tolist(),'last_target_xyz':pred[-1].tolist(),'maximum_departure_supplied_mm':float(xyz_to_supplied.max()*1000),'maximum_departure_measured_mm':float(xyz_to_measured.max()*1000),'min_proposed_object_distance_mm':float(np.linalg.norm(pred-obj,axis=1).min()*1000),'measured_object_distance_mm':float(np.linalg.norm(measured-obj)*1000),'gripper_minmax':[float(chunks[q,:,9].min()),float(chunks[q,:,9].max())],'hold_min_supplied_xyz_distance_mm':float(hd.min()*1000),'hold_min_processed_image_rmse':float(image_dist.min()),'hold_exact_processed_image_match':bool((image_dist==0).any()),'effective_domain_id':b['domain_id'].tolist()})
   wrist=np.array([r['eef_xyz']for r in tr]);w0=np.array(ep['initial_truth']['eef_position_m']);dist=np.linalg.norm(wrist-w0,axis=1)
   row={'parent':parent,'seed':seed,'folder':str(folder),'request_defects':request_defects,'request_stable_through_onset':not request_defects,'tokens_stable_through_onset':len(set(q['token_hash']for q in queries))==1,'first_close':event(close_i,ep),'first_contact':event(touch_i,ep),'first_supported_lift_run_start':event(lift,ep),'lift_qualification':'bilateral contacts >=10mm above initial for >=3 consecutive20Hz samples','first_executed_10mm_wrist_departure':event(first(dist>.01),ep),'q_e':qe,'queries':queries,'hold_supervision':holdinfo,'native_actor_fields':['input_ids','image_input','image_mask','proprio','domain_id'],'no_elapsed_time_input':True,'proprio_contract':'q0 measured wrist; q>0 previous predicted last XYZ+6D pose; both gripper/inactive arm inputs zero; no temporal image stack','supported_lift':lift is not None}
   rows.append(row)
   for name in ['episode.json','summary.json','controller_trace.jsonl.gz','predictions.npz','policy_io.npz','inference.json']:source_hashes[str(folder/name)]=sha(folder/name)
   if lift is not None and not io_cache:io_cache={'id':(parent,seed),'io':io,'hist':hist,'chunks':chunks,'tr':tr,'ep':ep}
   dump(O/'onset-audit.json',{'rows':rows,'complete':len(rows)==8,'epoch':time.time(),'source_sha256':source_hashes})
  del holdimgs
 selected=next(r for r in rows if r['supported_lift']);assert io_cache['id']==(selected['parent'],selected['seed'])
 qs=sorted(set([0,max(0,selected['q_e']-1),selected['q_e']]))
 folder=Path(selected['folder']);parent=selected['parent'];seed=selected['seed'];other=912172 if seed==912171 else 912171
 otherfolder=D/'episodes/T'/f'init-{parent}-feeding-{other}'
 ftext=selected['queries'][0]['instruction'];ltext=read(D/'episodes/T'/f'init-{parent}-loading-{seed}'/'input-audit/input-000.json')['instruction']
 prompts={'F-original':ftext,'L-original':ltext,'H-explicit':H};packets={};freeze=[]
 for q in qs:
  variants={cond:prepare(proc,io_cache['io'],io_cache['hist'],q,text)for cond,text in prompts.items()}
  for cond,b in variants.items():
   assert all(torch.equal(v,variants['F-original'][k])for k,v in b.items()if k!='input_ids')
   assert len(proc.tokenizer(prompts[cond])['input_ids'])<=128
  noises=[]
  for f in [folder,otherfolder]:
   n=f/'input-audit'/f'noise-{q:03d}.npz'
   with np.load(n)as z:noises.append(z['draw_0'].copy())
   assert noises[-1].shape==(1,30,20)
   source_hashes[str(n)]=sha(n)
  obj=np.array(io_cache['ep']['initial_truth']['object_position_m'] if q==0 else io_cache['tr'][q*30-1]['object_truth']['object_position_m'])
  packets[q]={'inputs':variants,'noises':noises,'expected_executor':io_cache['chunks'][q],'measured_xyz':io_cache['io']['eef_position_m'][q],'measured_quat_xyzw':io_cache['io']['eef_quat_xyzw'][q],'object_xyz_offline_only':obj,'native_history_executor':io_cache['hist'][q]}
  freeze.append({'query':q,'hashes':{c:{k:array_sha(v.numpy())for k,v in b.items()}for c,b in variants.items()},'noise_hashes':[array_sha(n)for n in noises],'expected_action_sha256':array_sha(io_cache['chunks'][q])})
 assert not any(r['request_defects']for r in rows),'Request defect discovered: stop before counterfactuals'
 assert time.time()<clock['preparation_deadline']
 torch.save(packets,O/'packets.pt')
 protocol={'experiment':'INIT04','ticket':11075,'selected_parent':parent,'selected_seed':seed,'selection':'lowest parent then lowest seed with verified supported lift; post-hoc mechanistic selection','selected_queries':qs,'q_e':selected['q_e'],'prompts':prompts,'explicit_hold_framing':'supplied no-motion primitive; not autonomous task completion recognition; no elapsed duration field exists in original request','packets':freeze,'max_distinct_outputs':len(qs)*3*2,'weight_sha256':'4d7a4a2547f084b6026164f27a6b7d43b5cb4d3b83ee4875343d108908f35636','weight_path':str(D/'train-T/step3000.pt'),'generation':{'mode':'eval','precision':'bf16 autocast; float32 weights','requires_grad':'allTrue','horizon':30,'denoise_steps':10,'domain':29,'observation':'O2'},'source_sha256':source_hashes,'new_rollouts':0,'new_training':0,'freeze_epoch':time.time(),'scratch_free_bytes':shutil.disk_usage(O).free,'packets_sha256':sha(O/'packets.pt')}
 assert protocol['scratch_free_bytes']>512*1024**2
 dump(O/'protocol.json',protocol);print(json.dumps({'passed':True,'selected':(parent,seed),'queries':qs,'rows':len(rows),'epoch':time.time()}),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:dump(O/'PREPARATION_STOPPED.json',{'error':traceback.format_exc(),'epoch':time.time()});raise
