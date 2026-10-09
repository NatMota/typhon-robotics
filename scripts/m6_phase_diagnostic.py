"""Post-freeze teacher-observation diagnostic; never feeds target data into rollouts."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from supervisor.adaptation.m6_joint import OUT,load_model,make_data,inference
from supervisor.adaptation.m6_contract import reorder

def rotation(v):
 a=v[...,:3];b=v[...,3:6];an=np.linalg.norm(a,axis=-1);u=a/np.maximum(an[...,None],1e-9)
 orth=b-(u*b).sum(-1,keepdims=True)*u;bn=np.linalg.norm(orth,axis=-1);w=orth/np.maximum(bn[...,None],1e-9)
 return np.stack([u,w,np.cross(u,w)],axis=-1),an,bn

e,a,chosen,batch=make_data();starts=a['frame_action_indices'][chosen]
model=load_model();s=torch.load(OUT/'training/final.pt',map_location='cpu',weights_only=False);model.load_state_dict(s['model']);del s
rows=[]
for label,index in [('initial',0),('approach',24),('approach_end',124),('descend',140),('close',158),('lift',176),('transfer',276),('place',328),('release',344),('retreat',364)]:
 w=int(np.argmin(abs(starts-index)));inputs=batch([w]);truth=inputs.pop('action').cpu().numpy()[0]
 actual=inference(model,inputs,'bf16').numpy()[0]
 pred=reorder(actual,True);truth=reorder(truth,True)
 pr,an,bn=rotation(pred[:,3:9]);tr,_,_=rotation(truth[:,3:9]);rel=pr@np.swapaxes(tr,-1,-2)
 angle=np.degrees(np.arccos(np.clip((np.trace(rel,axis1=-2,axis2=-1)-1)/2,-1,1)))
 pos=np.linalg.norm(pred[:,:3]-truth[:,:3],axis=-1)*1000
 rows.append({'phase':label,'requested_action_index':index,'window_start':int(starts[w]),'frame_index':int(chosen[w]),'xyz_error_mm_mean':float(pos.mean()),'xyz_error_mm_first':float(pos[0]),'xyz_error_mm_last':float(pos[-1]),'rotation_error_deg_mean':float(angle.mean()),'minimum_rotation_first_column_norm':float(an.min()),'minimum_rotation_orthogonal_second_norm':float(bn.min()),'target_closed_fraction':float((truth[:,9]>.5).mean()),'predicted_closed_fraction':float((pred[:,9]>.5).mean()),'predicted_close_probability_mean':float(pred[:,9].mean()),'first_pred_xyz':pred[0,:3].tolist(),'first_target_xyz':truth[0,:3].tolist()})
 print(rows[-1],flush=True)
(OUT/'teacher_observation_diagnostic.json').write_text(json.dumps({'condition':'FT3000','post_evaluation_diagnostic':True,'source':e['id'],'no_checkpoint_selection_or_training':True,'meaning':'Fully denoised predictions with recorded expert observations/history; not closed-loop success. Each window uses fixed inference seed7381.','rows':rows},indent=2)+'\n')
