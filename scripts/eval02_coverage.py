"""Read-only training coverage from actual sample logs and sealed manifests."""
import os,sys,json,gzip,hashlib,collections,time,csv,re
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'scripts')];os.chdir(R)
from supervisor.feeding_loading.evaluation import score_episode
O=Path('/var/tmp/typhon-eval02-coverage-20261015')
def read(p):return json.loads(Path(p).read_text())
sources={}
def sha(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 value=h.hexdigest();sources[str(p)]={'sha256':value,'bytes':p.stat().st_size};return value
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def label(e):
 t=e['config']['task'];return (t['type'],tuple(t['target_position_xy_m']),t['target_yaw_rad'])
roots={'LOAD04-B':Path('/var/tmp/typhon-load04-20261014'),'LOAD01-FL':Path('/var/tmp/typhon-load01-20261013'),'GOAL-varied':Path('/var/tmp/typhon-ost-sweep-20261012/goal-grounding')}
def main():
 rows=[];stage_details=[];configs={};completes={}
 for stage,root in roots.items():
  dirname={'LOAD04-B':'train-B','LOAD01-FL':'train-FL','GOAL-varied':'train-varied'}[stage]
  cfg=read(root/dirname/'config.json');complete=read(root/dirname/'complete.json')
  configs[stage]=cfg;completes[stage]=complete
  losses=[json.loads(s) for s in (root/dirname/'losses.jsonl').read_text().splitlines()]
  assert [x['step'] for x in losses]==list(range(1,3001))
  assert all(len(x['sample_ids'])==4 for x in losses)
  if stage=='GOAL-varied':
   manifest=root/'data/dataset-v2.json';data=read(manifest)
   assert sha(manifest)==cfg['data_hash']
   entries={e['id']:e for e in data['episodes'] if e['split']=='train'};ws=read(root/dirname/'window-index.json')
   windows={w['sample_id']:{**w,'episode':w['sample_id'].rsplit(':',1)[0]} for w in ws}
   assert set(entries)==set(cfg['training_sources'])
   assert not(set(entries)&set(cfg['combination_holdouts']))
   matched_plan=None
  else:
   manifest=root/'data/qualified-dataset.json';data=read(manifest)
   assert sha(manifest)==cfg['data_gate']['dataset_sha256']==complete['dataset_sha256']
   entries={e['id']:e for e in data['episodes']};ws=data['windows'];windows={w['id']:w for w in ws}
   planpath=root/'data/sampling-plan.json.gz'
   assert sha(planpath)==cfg['data_gate']['sampling_sha256']==complete['sampling_sha256']
   plan=json.load(gzip.open(planpath,'rt'))['plan']['B' if stage=='LOAD04-B' else 'FL']
   assert all(row['sample_ids']==[ws[i]['id'] for i in plan[k]] for k,row in enumerate(losses))
   matched_plan=True
  counts=collections.Counter();cells=collections.Counter();seen=set()
  for row in losses:
   for sid in row['sample_ids']:
    w=windows[sid];e=entries[w['episode']];key=label(e);counts[key]+=1
    cells[(e['source_demo_id'],*key,w.get('visibility','native'))]+=1;seen.add(sid)
  details=[]
  for e in entries.values():
   task,xy,yaw=label(e);instruction=e['instruction']
   assert all(f'{axis}={v:.6f}' in instruction for axis,v in zip(['x','y'],xy)),e['id']
   assert f'{yaw*180/np.pi:.6f} degrees' in instruction,e['id']
   episode_path=Path(e['source_path'])/'episode.json'
   actual=read(episode_path);actual_sha=sha(episode_path)
   if e.get('source_episode_sha256'):assert actual_sha==e['source_episode_sha256']
   sc=score_episode(actual['initial_truth'],actual['final_truth'],actual['stability_truths'],e['config'],actual['attempts'])
   assert sc['success'],(stage,e['id'],'placement association fails',sc['criteria'])
   details.append({'id':e['id'],'actual_placement_xy_m':actual['final_truth']['object_position_m'][:2],'actual_placement_yaw_rad':actual['final_truth']['object_yaw_rad'],'actual_goal_score_recomputed_strict':sc['success'],'actual_position_error_mm':1000*sc['position_error_m'],'source_episode_sha256_checked':actual_sha,'source_demo_id':e['source_demo_id'],'task':task,'goal_xy_m':xy,'goal_yaw_rad':yaw,'instruction':instruction,'action_labels':e.get('action_labels'),'actions':e['actions'],'frames':e['frames'],'archive':e.get('absolute_path',e.get('path')),'archive_sha256_recorded':e['sha256']})
  for (task,xy,yaw),count in sorted(counts.items()):
   rows.append({'stage':stage,'task':task,'goal_x_m':xy[0],'goal_y_m':xy[1],'goal_yaw_rad':yaw,'actual_presentations':count})
  stage_details.append({'stage':stage,'steps':len(losses),'actual_presentations':sum(counts.values()),'unique_episode_count':len(entries),'source_demo_ids':sorted(set(e['source_demo_id'] for e in entries.values())),'unique_actual_windows':len(seen),'available_windows':len(windows),'all_samples_resolve_to_manifest':True,'actual_sample_order_matches_frozen_plan':matched_plan,'episodes':details,'cell_counts':[{'source':k[0],'task':k[1],'xy':k[2],'yaw':k[3],'visibility':k[4],'presentations':v} for k,v in sorted(cells.items())]})
  for f in [root/dirname/'config.json',root/dirname/'complete.json',root/dirname/'losses.jsonl']:sha(f)
 assert configs['LOAD04-B']['parent_sha256']==completes['LOAD01-FL']['checkpoint_sha256']
 assert configs['LOAD01-FL']['parent_sha256']==completes['GOAL-varied']['checkpoint_sha256']
 assert completes['LOAD04-B']['checkpoint_sha256']=='de861da493be2a851e2fae5289f798cd34ff714e120f974f9c2197847732e379'
 # Check the actual final continuation archives without decompressing RGB.
 target_checks=[]
 dataset=read(roots['LOAD04-B']/'data/qualified-dataset.json')
 for e in dataset['episodes']:
  path=Path(e['absolute_path']);assert sha(path)==e['sha256']
  with np.load(path) as z:
   target=z['targets'];idx=z['frame_action_indices'];pro=z['proprio']
   assert target.shape==(e['actions'],20) and pro.shape==(e['frames'],20)
   assert np.isfinite(target).all() and np.isfinite(pro).all()
   assert np.all(np.diff(idx)==4)
   target_checks.append({'id':e['id'],'archive_hash_matches':True,'targets_shape':list(target.shape),'proprio_shape':list(pro.shape),'frame_action_stride':4,'target_xyz_min_m':target[:,:3].min(axis=0).tolist(),'target_xyz_max_m':target[:,:3].max(axis=0).tolist(),'target_semantics':'absolute EEF commands, not object-goal coordinates; array variation is motion-phase variation, not additional task goals'})
 source=R/'scripts/load04_train.py';assert sha(source)==configs['LOAD04-B']['script_sha256']
 base={'path':configs['GOAL-varied']['base'],'recorded_sha256':configs['GOAL-varied']['base_sha256'],'metadata_available_in_checkpoint_folder':False,'scope':'Full earlier pretraining and broad-Pt data not re-audited here. No claim that model has never encountered other goals.'}
 result={'passed':True,'read_only':True,'rollouts':0,'training':False,'no_holdout_actions_or_outcomes_read':True,'lineage_metadata_verified':True,'lineage_order':['broad-Pt12k','GOAL-varied3000','LOAD01-FL3000','LOAD04-B3000'],'earlier_base':base,'goal_counts':rows,'stages':stage_details,'latest_action_archive_checks':target_checks,'latest_loader_source_hash_matches_training':True,'limits':'Ancestry hash links verified from sealed metadata; this audit did not rehash all4GB historical checkpoints. Latest B bytes were verified in EVAL02-XY. Original broad/pretraining not exhaustively re-audited. Coverage is not a behavioural retention test.','finished_epoch':time.time()}
 assert time.time()<read(O/'clock.json')['review_deadline']
 dump(O/'coverage.json',result);dump(O/'sources.json',sources)
 with (O/'goals.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'goal_counts':rows,'stages':[{k:x[k] for k in ['stage','actual_presentations','unique_episode_count','source_demo_ids','unique_actual_windows','available_windows','actual_sample_order_matches_frozen_plan']} for x in stage_details]}),flush=True)
if __name__=='__main__':main()
