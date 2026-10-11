"""Six P_train takes only; verify RGB localizer before frozen comparison."""
from pathlib import Path
import json,sys,math,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from exp20261011_hierarchy import DATA,OUT,TRAIN,save
from exp20261011_hierarchy_runtime_v2 import ObservableBlock,goal_evidence

def main():
 out=OUT/'observable-pose-train-qualification-v1';out.mkdir(exist_ok=False)
 data={e['id']:e for e in json.loads((DATA/'dataset.json').read_text())['episodes']};phases=json.loads((OUT/'phase-head-v1/source-evidence.json').read_text());rows=[]
 for e in phases:
  if e['take'] not in TRAIN:continue
  source=data[e['take']];cfg=source['config'];release=next(p for p in e['primitives'] if p['label']=='release_open')['action_start'];folder=DATA/source['source_path'].split('data-canonical/',1)[1];ac=[json.loads(l) for l in (folder/'robot_actions.jsonl').read_text().splitlines()];episode=json.loads((folder/'episode.json').read_text());sensor=ObservableBlock(cfg)
  with np.load(DATA/source['path']) as z:
   choices=[0]+np.where((z['frame_action_indices']>=release-12)&(z['frame_action_indices']<=release))[0].tolist()
   for fi in choices:
    idx=int(z['frame_action_indices'][fi]);truth=episode['initial_truth'] if idx==0 else ac[idx-1]['post_action_truth']
    # EEF quaternion was not recorded here; it is unused because carry is disabled in this optical qualification.
    obs={'rgb':{k:z[k][fi] for k in ['primary','secondary']},'proprio':{'eef_position_m':truth['eef_position_m'],'eef_quat_xyzw':[0,0,0,1],'gripper_joint_position_m':truth['gripper_joint_position_m']},'task':cfg['task']}
    pred=sensor.step(obs,idx,held=False,carrying_phase=idx!=0);g=goal_evidence(pred,cfg)
    pe=None if pred['position_m'] is None else float(np.linalg.norm(pred['position_m'][:2]-np.array(truth['object_position_m'][:2])))
    ye=None if pred['yaw_rad'] is None else math.degrees(abs(math.atan2(math.sin(pred['yaw_rad']-truth['object_yaw_rad']),math.cos(pred['yaw_rad']-truth['object_yaw_rad']))))
    rows.append({'take':e['take'],'condition':'initial' if fi==0 else 'pre_release','action_index':idx,'seconds_before_release':(release-idx)/20,'estimate':pred,'goal':g,'evaluator_only_position_error_m':pe,'evaluator_only_yaw_error_deg':ye})
 save(out/'results.json',rows)
 r={'train_takes':6,'heldout_takes_inspected':0,'samples':len(rows),'observed':sum(e['estimate']['status']=='observed' for e in rows),'max_position_error_mm':max(e['evaluator_only_position_error_m']*1000 for e in rows if e['evaluator_only_position_error_m'] is not None),'max_yaw_error_deg':max(e['evaluator_only_yaw_error_deg'] for e in rows if e['evaluator_only_yaw_error_deg'] is not None),'takes_with_three_complete_prerelease_frames':sum(sum(e['take']==take and e['condition']=='pre_release' and e['goal']['complete'] is True for e in rows)>=3 for take in TRAIN),'uncertainty_is_engineering_bound_not_calibrated_probability':True}
 save(out/'summary.json',r);print(json.dumps(r))
if __name__=='__main__':main()
