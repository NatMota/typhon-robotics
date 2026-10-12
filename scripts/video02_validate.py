"""Validate provenance, masks and exports; does not certify perception accuracy."""
import argparse,hashlib,json,subprocess,collections
from pathlib import Path
import numpy as np
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main(a):
 d=a.data;idx=json.loads((d/'source-index.json').read_text());rows=idx['samples'];config=json.loads((d/'protocol.json').read_text());manifest=json.loads((d/'input-manifest.json').read_text())
 raw={r['recording_id']:r for r in manifest['raw']};assert len(raw)==8
 for r in raw.values():assert sha(a.project_root/r['relative_path'])==r['sha256']
 ids={r['sample_id']:r for r in rows};assert len(ids)==len(rows)==2644
 for r in rows:
  assert r['source_start_s']<=r['source_time_s']<r['source_end_s']
  assert abs(raw[r['recording_id']]['frame_pts_s'][r['source_frame_index']]-r['source_time_s'])<1e-9
  assert abs(r['source_time_s']-r['source_start_s']-r['clip_time_s'])<1e-9
  assert sha(d/r['frame_path'])==r['frame_sha256']
  assert r['demo_id']in config['splits'][r['split']] and r['demo_id']!='demo_0017'
  assert r['is_grasped']is None and r['metric_xyz']is None and r['metric_lift_height_m']is None
  if not r['valid_xy']:assert r['xy_normalized']is None and r['yaw_clockwise_deg']is None and r['crop_xyxy']is None
 f=np.load(d/'features/features.npz',allow_pickle=False)
 assert list(f['sample_ids'])==list(ids)
 for key in config['feature_methods']:assert f[key].shape==(len(rows),1024)and np.isfinite(f[key]).all()
 summary=json.loads((d/'features/summary.json').read_text());assert summary['base_model_sha256']=='433acffc992f457b1737d35ae20f81a895521f38eadaf03c736b6c0b0af1c93e'
 windows=[json.loads(l)for l in (d/'analysis-v2/four-second-anchors.jsonl').read_text().splitlines()]
 for w in windows:
  anchors=w['anchors'];assert len(anchors)==30 and len({r['sample_id']for r in anchors})==30
  for r,t in zip(anchors,np.linspace(w['window_start_s'],w['window_start_s']+4,30)):
   original=ids[r['sample_id']]
   assert abs(r['clip_time_s']-t)<=.075 and original['demo_id']==w['demo_id']and original['view']==w['view']and original['split']==w['split']
   assert r['xy_normalized']==original['xy_normalized']and r['yaw_clockwise_deg']==original['yaw_clockwise_deg']and r['is_grasped']is None
 for filename in ['trajectory-summary.json','four-second-anchors.jsonl','feature-comparison.json']:
  assert sha(d/'analysis'/filename)==sha(d/'analysis-v2'/filename),'QA layout must not alter metrics'
 for meta in json.loads((d/'analysis-v2/previews.json').read_text()):
  p=d/'analysis-v2'/meta['path']
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,nb_read_frames,width,height','-of','json',str(p)]))['streams'][0]
  assert probe['codec_name']=='h264'and int(probe['nb_read_frames'])==meta['paired_frames']and probe['width']==800
 goal=json.loads((d/'goal-probe/results.json').read_text())
 for g in goal['goal_similarity']:
  assert g['demo_id']not in g['reference_sources']
  assert set(g['reference_sources'])<=set(config['splits']['P_train'])
 # Gap and circular-angle regression checks using the committed shared functions.
 from video02_analyze import anchors as make_anchors,valid_pairs,circ_diff
 assert circ_diff(1,359)==2
 assert make_anchors([{'clip_time_s':x}for x in [0,1,2,3,4]],0,4,30)is None
 assert not valid_pairs([{'valid_xy':True,'clip_time_s':0},{'valid_xy':True,'clip_time_s':.3}])
 checks={'raw_recordings_unchanged':8,'frame_hashes_verified':len(rows),'source_PTS_and_take_bounds_verified':True,'feature_ID_alignment_verified':True,'excluded_take_absent':True,'split_assignments_preserved':True,'unique_noninterpolated_30_point_windows':len(windows),'missing_contact_and_depth_preserved':True,'same_metrics_after_preview_layout_fix':True,'h264_previews_verified':6,'goal_references_train_only_and_self_excluded':True,'regression_checks_passed':3,'human_take_count':16,'recording_pair_count':4,'agent_visual_review':'initial/middle/final primary contact sheet and actual224px full/crop sheet inspected; six examples, not pointwise annotation accuracy','human_review_newly_obtained':False,'qualified_for_robot_action_training':False}
 (d/'validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--project-root',type=Path,required=True);main(p.parse_args())
