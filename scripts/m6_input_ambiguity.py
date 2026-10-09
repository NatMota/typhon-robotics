"""Post-freeze audit of input similarity versus conflicting source action chunks."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from supervisor.adaptation.m6_joint import make_data,OUT
torch.set_num_threads(4)
e,a,chosen,batch=make_data();samples=[]
for i in range(48):
 b=batch([i]);im=b['image_input'].cpu()[0];p=b['proprio'].cpu()[0];p[9]=p[19]=0
 samples.append({'image':im,'proprio':p,'mask':b['image_mask'].cpu()[0],'target':b['action'].cpu()[0],'start':int(a['frame_action_indices'][chosen[i]])})
pairs=[];exact=0
for i,x in enumerate(samples):
 for j in range(i+1,len(samples)):
  y=samples[j]
  if not torch.equal(x['mask'],y['mask']):continue
  pose=(x['proprio']-y['proprio']).abs()
  if float(pose[:3].max())>.001 or float(pose[3:9].max())>.01:continue
  xyz=torch.linalg.vector_norm(x['target'][:,:3]-y['target'][:,:3],dim=-1)*1000
  grip=((x['target'][:,9]>.5)!=(y['target'][:,9]>.5)).float().mean()
  if xyz.max()<20 and grip==0:continue
  imdelta=(x['image']-y['image']).abs()
  is_exact=torch.equal(x['image'],y['image']) and torch.equal(x['proprio'].bfloat16(),y['proprio'].bfloat16())
  exact+=int(is_exact)
  pairs.append({'window_indices':[i,j],'action_starts':[x['start'],y['start']],'max_proprio_xyz_difference_mm':float(pose[:3].max()*1000),'max_proprio_rot6d_difference':float(pose[3:9].max()),'image_normalized_MAE':float(imdelta.mean()),'image_exact_equal':torch.equal(x['image'],y['image']),'effective_bf16_proprio_equal':torch.equal(x['proprio'].bfloat16(),y['proprio'].bfloat16()),'future_xyz_difference_mm_mean':float(xyz.mean()),'future_xyz_difference_mm_max':float(xyz.max()),'first_action_xyz_difference_mm':float(xyz[0]),'gripper_label_disagreement_fraction':float(grip)})
pairs.sort(key=lambda r:r['image_normalized_MAE'])
report={'source':e['id'],'training_windows':48,'thresholds':{'max_current_proprio_xyz_difference_m':.001,'max_current_proprio_rot6d_difference':.01,'minimum_future_XYZ_difference_mm_or_any_gripper_disagreement':20},'mask_and_instruction_matched':True,'exact_image_and_bf16_proprio_conflicting_pairs':exact,'candidates':pairs,'interpretation':'Exact matches at evaluated inputs indicate ambiguous future chunks for those windows. Near matches are evidence of weak phase cues, not proof that no visual cue distinguishes the frames. No training or labels changed.'}
(OUT/'input_ambiguity_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'exact_conflicts':exact,'candidate_pairs':len(pairs),'first12':pairs[:12]},indent=2),flush=True)
