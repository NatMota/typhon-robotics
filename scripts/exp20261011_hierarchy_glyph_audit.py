from pathlib import Path
import json,sys,math,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from exp20261011_hierarchy import DATA,OUT,TRAIN,save
from exp20261011_hierarchy_visual import GlyphYaw,rectified_secondary

def main():
 out=OUT/'glyph-train-qualification-v1';out.mkdir(exist_ok=False)
 classifier=GlyphYaw(ROOT/'assets/duplo_31110pb074_v2/numeral_2_mask.png');data={e['id']:e for e in json.loads((DATA/'dataset.json').read_text())['episodes']}
 phases=json.loads((OUT/'phase-head-v1/source-evidence.json').read_text());rows=[]
 for e in phases:
  if e['take'] not in TRAIN:continue
  source=data[e['take']];cfg=source['config'];release=next(p for p in e['primitives'] if p['label']=='release_open')['action_start'];folder=DATA/source['source_path'].split('data-canonical/',1)[1]
  ac=[json.loads(l) for l in (folder/'robot_actions.jsonl').read_text().splitlines()]
  with np.load(DATA/source['path']) as z:
   tests=[('initial_primary',0),('release_secondary_rectified',int(np.where(z['frame_action_indices']<=release)[0][-1]))]
   for condition,fi in tests:
    idx=int(z['frame_action_indices'][fi]);truth=json.loads((folder/'episode.json').read_text())['initial_truth'] if idx==0 else ac[idx-1]['post_action_truth']
    rgb=z['primary'][fi] if fi==0 else rectified_secondary(z['secondary'][fi],cfg,truth['eef_position_m'][2]+.005)
    pred=classifier(rgb);error=None if pred['yaw_rad'] is None else math.degrees(abs(math.atan2(math.sin(pred['yaw_rad']-truth['object_yaw_rad']),math.cos(pred['yaw_rad']-truth['object_yaw_rad']))))
    rows.append({'take':e['take'],'condition':condition,'action_index':idx,'estimate':pred,'evaluator_only_yaw_deg':math.degrees(truth['object_yaw_rad']),'error_deg':error})
 save(out/'results.json',rows);print(json.dumps(rows))
if __name__=='__main__':main()
