"""Summarize the frozen bounded diagnostics; no additional model calls or rollouts."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from v4_prompt_probes_20261011 import read,save,matrix,sha,SEEDS

def contrast(a,b):
 rotations=[math.degrees(Rotation.from_matrix(matrix(y)@matrix(x).T).magnitude()) for x,y in zip(a,b)]
 positions=np.linalg.norm(b[:,:3]-a[:,:3],axis=1)*1000
 return {'mean_rotation_difference_deg':float(np.mean(rotations)),'max_rotation_difference_deg':max(rotations),
         'final_rotation_difference_deg':rotations[-1],'mean_position_difference_mm':float(np.mean(positions)),
         'max_position_difference_mm':float(np.max(positions)),'final_position_difference_mm':float(positions[-1])}

def run(root):
 base=root/'diagnostics-v4';p=base/'priority-bundle';out=base/'bounded-summary';out.mkdir(exist_ok=False)
 assert read(p/'complete.json')['generated_chunks']==25
 assert read(base/'proprio-counterfactual/complete.json')['newly_generated']==9
 rows=[];contrasts=[];proprio=[]
 for f in sorted((p/'responses').glob('*/result.json')):
  j=read(f);a=j['action_metrics'];opening=next((x['waypoint'] for x in a if not x['would_close']),None)
  rows.append({'job_id':f.parent.name,'case':j['job']['case'],'seed':j['job']['seed'],
    'first_open_waypoint':opening,'final_wrist_x_yaw_deg':a[-1]['projected_wrist_x_yaw_deg'],
    'final_rotation_from_input_z_deg':a[-1]['rotation_from_input_proprio_world_rotvec_deg'][2],
    'final_rotation_from_measured_z_deg':a[-1]['rotation_from_measured_world_rotvec_deg'][2],
    'max_position_from_input_mm':max(float(np.linalg.norm(x['delta_from_input_proprio_xyz_mm'])) for x in a),
    'final_position_from_input_mm':float(np.linalg.norm(a[-1]['delta_from_input_proprio_xyz_mm'])),
    'final_close_probability':a[-1]['gripper_close_probability'],'all30_keep_closed':all(x['would_close'] for x in a)})
 for first,second in [('initial-object+00-goal00','initial-object+00-goal30'),('initial-object+90-goal00','initial-object+90-goal30'),('initial-object+00-goal00','initial-object+90-goal00'),('release-P0','release-P1'),('release-P2','release-P3')]:
  for seed in SEEDS:
   a=np.load(p/'responses'/f'{first}-seed{seed}'/'actions.npz')['actions'];b=np.load(p/'responses'/f'{second}-seed{seed}'/'actions.npz')['actions']
   contrasts.append({'first':first,'second':second,'seed':seed,**contrast(a,b)})
 for f in sorted((base/'proprio-counterfactual/responses').glob('*/result.json')):
  j=read(f);a=j['selected_metrics'];proprio.append({'condition':j['condition'],'seed':j['seed'],
    'opening_waypoint':j['selected_waypoints']['new_first_opening_waypoint'],
    'implied_yaw_error_at_original_opening_deg':a['original_opening_waypoint']['fixed_grasp_implied_yaw_error_deg'],
    'implied_yaw_error_at_new_opening_deg':a['new_first_opening_waypoint']['fixed_grasp_implied_yaw_error_deg'],
    'implied_yaw_error_at_final_deg':a['final_waypoint']['fixed_grasp_implied_yaw_error_deg'],
    'best_implied_yaw_error_deg':a['best_implied_yaw_waypoint']['fixed_grasp_implied_yaw_error_deg']})
 for name,records in [('chunks',rows),('paired-contrasts',contrasts),('proprio',proprio)]:
  with (out/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 result={'priority_chunks':25,'additional_proprio_chunks':9,'physical_replay_branches':2,
         'repeatability':read(p/'inference-complete.json'),'input_preparation':read(p/'input-preparation.json'),
         'paired_contrasts':contrasts,'chunks':rows,'proprio':proprio,
         'replays':[{k:v for k,v in r.items() if k in ['branch','qualified_unchanged_replay','final_object_yaw_error_deg','last_bilateral_contact','max_replay_position_error_m','max_replay_object_rotation_error_deg']} for r in read(p/'replay-summary.json')['branches']],
         'limits':['Single selected scene/input packet, three paired stochastic seeds. No broad reliability estimate.',
                   'Whole-object rotation changes body, studs and numeral together; it is not a numeral-specific visual intervention.',
                   'Only the saved original and opening-suppressed action branches were physically simulated. All new prompt/proprio outputs are unexecuted proposals.',
                   'Fixed-grasp implied object rotations are kinematic, not dynamics predictions.','No training, controller repair or benchmark change applied.'],
         'script_sha256':sha(Path(__file__))}
 save(out/'summary.json',result);(out/'analysis-source.py').write_bytes(Path(__file__).read_bytes())
 print(json.dumps({'contrasts':contrasts,'primitive_rows':[r for r in rows if r['case'] in ['release-P2','release-P3']],'proprio':proprio},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();run(a.root)
