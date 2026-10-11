"""Post-hoc trace diagnostics and representative final-frame QA; no new rollouts."""
import csv,gzip,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
OUT=Path('/var/tmp/typhon-research-20261011/broad-pt')
METHODS=['narrow-m6-3000','broad-pt-3000','broad-pt-12000']
FAMILIES=['0006','0014','0016']
def read_gzip(path):
 with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]
def main():
 rows=[]
 for method in METHODS:
  for take in FAMILIES:
   for seed in [7417,7418]:
    folder=OUT/'evaluation'/method/f'pval-{take}-seed{seed}'
    episode=json.loads((folder/'episode.json').read_text());trajectory=read_gzip(folder/'trajectory.jsonl.gz');trace=read_gzip(folder/'controller_trace.jsonl.gz')
    by_step={x['step']:x for x in trajectory};assert len(by_step)==len(trajectory)
    assert all(x['step'] in by_step for x in trace)
    trajectory=[by_step[x['step']] for x in trace];assert len(trajectory)==len(trace)
    truth=[x['truth'] for x in trajectory]
    object_xyz=np.array([x['object_position_m'] for x in truth]);eef=np.array([x['eef_position_m'] for x in truth]);target=np.array([x['target'] for x in trace]);initial=np.array(episode['initial_truth']['object_position_m'])
    requested=np.array([x['requested_action'] for x in trajectory]);applied=np.array([x['applied_action'] for x in trajectory]);delta=np.abs(applied-requested)
    dist=np.linalg.norm(eef-object_xyz,axis=1);target_dist=np.linalg.norm(target[:,:3]-object_xyz,axis=1);close=target[:,9]>.5
    np.testing.assert_array_equal(np.where(close,1.0,-1.0),requested[:,-1])
    assert np.isfinite(target).all() and np.isfinite(eef).all()
    nearest=int(dist.argmin());near_target=int(target_dist.argmin())
    rows.append({'method':method,'source_demo_id':'demo_'+take,'seed':seed,'steps':len(truth),'minimum_eef_to_object_centre_mm':float(dist.min()*1000),'minimum_eef_xy_to_object_centre_mm':float(np.linalg.norm(eef[:,:2]-object_xyz[:,:2],axis=1).min()*1000),'minimum_target_to_object_centre_mm':float(target_dist.min()*1000),'minimum_target_height_above_object_centre_mm':float((target[:,2]-object_xyz[:,2]).min()*1000),'nearest_eef_step':int(trace[nearest]['step']),'eef_xyz_at_nearest':eef[nearest].tolist(),'target_xyz_at_nearest_target':target[near_target,:3].tolist(),'target_close_command_steps':int(close.sum()),'gripper_physically_open_steps':sum(x['gripper_open'] for x in truth),'any_grasp_contact_steps':sum(x['grasp_contact'] for x in truth),'bilateral_contact_steps':sum(x['bilateral_grasp_contact'] for x in truth),'minimum_gripper_probability':float(target[:,9].min()),'maximum_gripper_probability':float(target[:,9].max()),'raw_vs_applied_action_changed_steps':int(np.any(delta>1e-10,axis=1).sum()),'raw_vs_applied_action_max_difference':float(delta.max()),'maximum_object_displacement_mm':float(np.linalg.norm(object_xyz-initial,axis=1).max()*1000),'mean_target_tracking_error_mm':float(np.mean([x['position_error_mm'] for x in trace]))})
 assert len(rows)==18
 summary={'scope':'Exploratory post-hoc diagnostics on model-driven steps of the existing 18 frozen evaluation traces, excluding the separate terminal stability steps; no extra rollouts or changed metrics. EEF and target distances reference object centre; no new grasp thresholds are implied. Active Panda gripper slot9 >0.5 follows the existing native decoder and is asserted against actual requested gripper commands; slot19 is unused. Raw/applied equality tests only the environment application boundary, not prior native controller clipping.','rows':rows}
 (OUT/'motion-diagnostics.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 with (OUT/'motion-diagnostics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 dest=OUT/'figures';dest.mkdir(exist_ok=True)
 canvas=Image.new('RGB',(1660,850),'white');d=ImageDraw.Draw(canvas)
 try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
 except OSError:font=ImageFont.load_default()
 d.text((12,8),'Representative paired seed7417: initial state versus final states (overhead + oblique)',fill='black',font=font)
 for row,take in enumerate(FAMILIES):
  y=45+row*258
  columns=[('Initial state','narrow-m6-3000','initial.png')]+[(m,m,'final.png') for m in METHODS]
  for col,(label,method,name) in enumerate(columns):
   path=OUT/'evaluation'/method/f'pval-{take}-seed7417'/name
   im=Image.open(path).convert('RGB');im.thumbnail((400,210))
   x=12+col*412;d.text((x,y),f'Take {take} | {label}',fill='black',font=font);canvas.paste(im,(x,y+27))
 d.text((12,820),'This contact sheet shows endpoints only; complete videos and step traces are retained. No brick manipulation succeeded.',fill='black',font=font)
 canvas.save(dest/'representative-endpoints.jpg',quality=91)
 (dest/'endpoint-contact-sheet-provenance.json').write_text(json.dumps({'seed':7417,'families':FAMILIES,'models':METHODS,'columns':['shared initial overhead+oblique','narrow3k final','broad3k final','broad12k final'],'source':'evaluation/<model>/pval-<take>-seed7417/{initial,final}.png'},indent=2)+'\n')
 print(json.dumps([{k:r[k] for k in ['method','source_demo_id','seed','minimum_eef_to_object_centre_mm','minimum_target_to_object_centre_mm','target_close_command_steps','bilateral_contact_steps','mean_target_tracking_error_mm','raw_vs_applied_action_changed_steps','maximum_object_displacement_mm']} for r in rows],indent=2))
if __name__=='__main__':main()
