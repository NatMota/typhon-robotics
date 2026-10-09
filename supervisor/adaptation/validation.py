"""Shared held-out feeding panel; no loading examples or checkpoint selection."""
import json
from pathlib import Path
import numpy as np
import torch
from .observations import views,object_crop


def rotation6d(x):
    a=x[:3];a=a/np.linalg.norm(a);b=x[3:]-np.dot(a,x[3:])*a;b=b/np.linalg.norm(b)
    r=np.column_stack([a,b,np.cross(a,b)])
    if not np.isfinite(r).all() or abs(np.linalg.det(r)-1)>1e-5:raise ValueError('Invalid decoded rotation')
    return r


def validate_model(model,processor,dataset,out,stage):
    entries=[e for e in json.loads(dataset.read_text())['episodes'] if e['split'] in ['P_validation','S_validation']]
    assert len(entries)==6
    flags={n:p.requires_grad for n,p in model.named_parameters()}
    for p in model.parameters():p.requires_grad_(True)  # same normal generation path as worker
    rows=[];arrays_out={}
    try:
        for index,e in enumerate(entries):
            with np.load(dataset.parent/e['path']) as z:a={k:z[k] for k in z.files}
            valid=np.flatnonzero(a['frame_action_indices']+30<=len(a['targets']))
            selected=valid[np.linspace(0,len(valid)-1,4,dtype=int)]
            images=[views({k:a[k][i] for k in ['primary','secondary']},'O2') for i in selected]
            inputs=processor(images=images,language_instruction=[e['instruction']]*4)
            inputs.update(proprio=torch.from_numpy(a['proprio'][selected]),domain_id=torch.full((4,),3,dtype=torch.long))
            inputs={k:v.cuda() for k,v in inputs.items()}
            targets=np.stack([a['targets'][int(a['frame_action_indices'][i]):int(a['frame_action_indices'][i])+30] for i in selected])
            torch.manual_seed(4820+index)
            with torch.no_grad():losses=model(**inputs,action=torch.from_numpy(targets).cuda())
            torch.manual_seed(5820+index)
            with torch.inference_mode():generated=model.generate_actions(**inputs,steps=10).cpu().numpy()
            angles=[];rejected=0;clipped=0;commands=[]
            for b,i in enumerate(selected):
                # Offline conversion diagnostic uses the supplied command-history pose,
                # not an invented future measured robot state.
                p=a['proprio'][i]
                for pred,target in zip(generated[b],targets[b]):
                    try:
                        pr=rotation6d(pred[3:9]);tr=rotation6d(target[3:9])
                        angles.append(float(np.degrees(np.arccos(np.clip((np.trace(pr@tr.T)-1)/2,-1,1)))))
                        raw_pos=(pred[:3]-p[:3])/e['config']['control']['translation_step_m']
                        clipped+=bool(np.any(abs(raw_pos)>1))
                    except ValueError:rejected+=1
            row={'id':e['id'],'source_group':e['split'],'source_sha256':e['sha256'],'frame_indices':selected.tolist(),
                'native_losses':{k:float(v) for k,v in losses.items()},'active_xyz_rmse_mm':float(1000*np.sqrt(np.mean((generated[...,:3]-targets[...,:3])**2))),
                'active_rotation_mean_error_deg':float(np.mean(angles)) if angles else None,
                'gripper_decision_agreement':float(np.mean((generated[...,9]>.5)==(targets[...,9]>.5))),
                'decoded_rotation_rejections':rejected,'translation_step_clipped_commands':clipped,'generated_commands':120,
                'valid_detail_crops':sum(object_crop(a['primary'][i])[0] is not None for i in selected),
                'clipping_scope':'translation-only offline check against window initial command-history pose; full OSC conversion audited separately in simulation environment'}
            rows.append(row);arrays_out[e['id']+'_generated']=generated;arrays_out[e['id']+'_targets']=targets
        report={'stage':stage,'selection_use':False,'same_panel_all_methods':True,'windows_per_trajectory':4,'trajectories':rows,
                'groups':{g:{'mean_xyz_rmse_mm':float(np.mean([r['active_xyz_rmse_mm'] for r in rows if r['source_group']==g])),
                    'mean_rotation_error_deg':float(np.mean([r['active_rotation_mean_error_deg'] for r in rows if r['source_group']==g])),
                    'gripper_decision_agreement':float(np.mean([r['gripper_decision_agreement'] for r in rows if r['source_group']==g]))} for g in ['P_validation','S_validation']}}
        (out/f'validation-{stage}.json').write_text(json.dumps(report,indent=2)+'\n')
        np.savez_compressed(out/f'validation-{stage}-chunks.npz',**arrays_out)
        return report
    finally:
        for n,p in model.named_parameters():p.requires_grad_(flags[n])
