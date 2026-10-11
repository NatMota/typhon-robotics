"""P_train-only qualification of secondary-view prerelease evidence."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from exp20261011_hierarchy import DATA,OUT,TRAIN,save
from exp20261011_hierarchy_visual import rectified_secondary
from supervisor.planning.simple_goal_classifier import GoalSuccessClassifier
from supervisor.planning.typhon_service import calibrated_goal_context

def main():
 out=OUT/'rectification-train-qualification-v1';out.mkdir(exist_ok=False)
 evidence=json.loads((OUT/'phase-head-v1/source-evidence.json').read_text());meta={e['id']:e for e in json.loads((DATA/'dataset.json').read_text())['episodes']}
 clf=GoalSuccessClassifier();rows=[]
 for e in evidence:
  if e['take'] not in TRAIN:continue
  source=meta[e['take']];cfg=source['config'];context=calibrated_goal_context(cfg)
  release=next(p for p in e['primitives'] if p['label']=='release_open')['action_start']
  folder=DATA/source['source_path'].split('data-canonical/',1)[1];ac=[json.loads(l) for l in (folder/'robot_actions.jsonl').read_text().splitlines()]
  with np.load(DATA/source['path']) as z:
   indices=z['frame_action_indices'];chosen=np.where((indices>=release-24)&(indices<=release))[0]
   evaluations=[]
   for fi in chosen:
    idx=int(indices[fi]);eef_z=ac[idx-1]['post_action_truth']['eef_position_m'][2]
    tests=[]
    for label,plane in [('resting_top',cfg['workspace']['table_z_m']+cfg['object']['height_m']),('eef_offset_hypothesis',eef_z+.005)]:
     rect=rectified_secondary(z['secondary'][fi],cfg,plane);r=clf(rect,context)
     tests.append({'hypothesis':label,'plane_z_m':plane,'status':r['status'],'reason':r['reason'],'visual_checks':r['visual_checks'],'evidence':{k:v for k,v in r['evidence'].items() if k!='glyph_signature'}})
    evaluations.append({'action_index':idx,'relative_to_release_s':(idx-release)/20,'tests':tests})
   rows.append({'take':e['take'],'release':release,'evaluations':evaluations})
 save(out/'results.json',rows)
 result={'train_takes':6,'heldout_takes_inspected':0,'fixed_hypotheses':['resting_top','eef_offset_hypothesis'],'any_complete_before_release':{name:sum(any(t['status']=='complete' for f in r['evaluations'] for t in f['tests'] if t['hypothesis']==name) for r in rows) for name in ['resting_top','eef_offset_hypothesis']},'qualification_gate':'all6 must have>=3 consecutive complete frames before release; else no completion authority'}
 result['consecutive_complete']={name:sum(len(r['evaluations'])>=3 and all(next(t for t in f['tests'] if t['hypothesis']==name)['status']=='complete' for f in r['evaluations'][-3:]) for r in rows) for name in ['resting_top','eef_offset_hypothesis']}
 save(out/'summary.json',result);print(json.dumps(result))
if __name__=='__main__':main()
