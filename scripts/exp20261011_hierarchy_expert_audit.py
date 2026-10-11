"""Offline independent audit: would image gates veto recorded expert release?"""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from exp20261011_hierarchy import DATA,OUT,TRAIN,VAL,save,color_measure
from supervisor.planning.simple_goal_classifier import GoalSuccessClassifier
from supervisor.planning.typhon_service import calibrated_goal_context

def main():
 out=OUT/'observable-release-audit-v1';out.mkdir(exist_ok=False)
 evidence=json.loads((OUT/'phase-head-v1/source-evidence.json').read_text())
 meta={e['id']:e for e in json.loads((DATA/'dataset.json').read_text())['episodes']}
 clf=GoalSuccessClassifier();rows=[]
 for e in evidence:
  source=meta[e['take']];context=calibrated_goal_context(source['config']);parts=e['primitives'];release=next(p for p in parts if p['label']=='release_open')['action_start']
  with np.load(DATA/source['path']) as z:
   indices=z['frame_action_indices'];chosen=np.where((indices>=release-24)&(indices<=release+8))[0]
   evaluations=[]
   for fi in chosen:
    result=clf(z['primary'][fi],context)
    evaluations.append({'action_index':int(indices[fi]),'relative_to_release_s':float((indices[fi]-release)/20),'status':result['status'],'reason':result['reason'],'confidence':result['confidence'],'visual_checks':result['visual_checks'],'evidence':{k:v for k,v in result['evidence'].items() if k!='glyph_signature'}})
   before=[r for r in evaluations if r['action_index']<=release]
   rows.append({'take':e['take'],'split':e['split'],'expert_release_index':release,'image_at_or_before_release':before[-1] if before else None,'three_consecutive_visual_complete_before_release':len(before)>=3 and all(r['status']=='complete' for r in before[-3:]),'any_visual_complete_before_release':any(r['status']=='complete' for r in before),'evaluations':evaluations})
 save(out/'expert-release-visibility.json',rows)
 summary={'episodes':len(rows),'pre_release_single_image_complete':sum(r['image_at_or_before_release']['status']=='complete' for r in rows),'pre_release_three_frame_complete':sum(r['three_consecutive_visual_complete_before_release'] for r in rows),'unknown_nearest':sum(r['image_at_or_before_release']['status']=='unknown' for r in rows),'limitation':'offline RGB5Hz audit; existing expert outcomes successful; image gate is a counterfactual veto audit, not online rollout','recommendation':'Do not enable a release veto as authoritative if it blocks valid expert releases; improve pre-release observability or sensor evidence first.'}
 save(out/'summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
