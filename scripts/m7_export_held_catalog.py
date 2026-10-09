"""Export causal annotation-only grasp states for a native robot catalog; no GPU."""
import argparse,collections,gzip,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from m7_prepare_visual_enrichment import oracle_states, self_check, THRESHOLDS
from supervisor.adaptation.m7_visual import load_held_lookup

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(temporal,out):
 snapshot_path=temporal/'source_snapshot.json.gz';evidence_path=temporal/'semantic_evidence.json.gz'
 snap=json.load(gzip.open(snapshot_path,'rt'));evidence=json.load(gzip.open(evidence_path,'rt'))
 out.mkdir(parents=True,exist_ok=False);checks=self_check();states=collections.Counter();labels=collections.Counter();episodes=[];rows=[]
 for ep in snap['episodes']:
  assert ep['split'].endswith('_train'),(ep['id'],ep['split'])
  eid=ep['id'];ev=evidence['episodes'][eid];actions=ev['actions'];episode=ev['episode'];truths=[episode['initial_truth']]+[a['post_action_truth'] for a in actions];commands=[a['normalized_command'][-1] for a in actions]
  oracle=oracle_states(truths,commands);assert ep['verified_source_sha256']==ep['sha256']==ep['source_sha256_after_read']
  for end in sorted(set([1,len(truths)//3,len(truths)//2,len(truths)-1])):
   assert oracle_states(truths[:end],commands[:max(0,end-1)])==oracle[:end]
  local=collections.Counter();hz=ep['config']['control']['frequency_hz']
  for frame,k in enumerate(ep['frame_action_indices']):
   label=oracle[k];assert label['latest_consumed_action_index']==k-1
   assert truths[k]['control_steps']==episode['reset_warmup_steps']+k
   row=dict(schema='m7-visual-enrichment-v1-oracle-only',export_version='m7-training-catalog-v2',frame_id=f'{eid}:f{frame:04d}',recording_id=eid,source_demo_id=ep.get('source_demo_id'),category=ep['category'],split=ep['split'],frame_index=frame,pre_action_index=k,preceding_action_index=k-1,time_since_first_exported_action_s=k/hz,simulator_time_s=truths[k]['sim_time_s'],source_npz_sha256=ep['sha256'],source_actions_sha256=ep['source_action_sha256'],source_npz_uri=ep['source_npz_uri'],annotations=label,input_forbidden=True,annotation_conditioned_experiment_authorized=True,runtime_available=False,source='simulator-truth annotation from current/past contact and lift; no future maxima, phase duration, terminal success or future frame consumed')
   rows.append(row);states[label['state']]+=1;labels[str(label['is_grasped'])]+=1;local[str(label['is_grasped'])]+=1
  episodes.append(dict(episode_id=eid,split=ep['split'],category=ep['category'],frames=len(ep['frame_action_indices']),held_counts=dict(local),source_npz_sha256=ep['sha256'],causal_prefix_invariance_passed=True))
 dest=out/'oracle_annotations.jsonl'
 with dest.open('w') as f:
  for row in rows:f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
 lookup=load_held_lookup(dest);assert len(lookup)==len(rows);assert len(set((r['recording_id'],r['frame_index']) for r in rows))==len(rows)
 summary=dict(schema='m7-held-catalog-validation-v2',episodes=episodes,episode_count=len(episodes),frame_count=len(rows),states=dict(states),held=dict(labels),checks=checks,all_frame_indices_causal=True,all_sources_training_only=True,loader_lookup_count=len(lookup),annotation_file=str(dest),annotation_sha256=sha(dest),snapshot_sha256=sha(snapshot_path),semantic_evidence_sha256=sha(evidence_path),exporter_sha256=sha(__file__),oracle_rule_sha256=sha(ROOT/'scripts/m7_prepare_visual_enrichment.py'),thresholds=THRESHOLDS,raw_npz_modified=False,robot_actions_modified=False,runtime_available=False,interpretation='Optional annotation-conditioned experiment with causal privileged simulator contact/object pose. This is not a live RGB grasp detector. Unknown renders OFF with false validity mask. RGB-localization is recomputed separately by shared augmenter.')
 (out/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 (out/'REPRODUCE.txt').write_text(f'.venv-xvla/bin/python scripts/m7_export_held_catalog.py --temporal {temporal} --output {out}\nOutput directory must not already exist. Native reader: load_held_lookup(oracle_annotations.jsonl); index=(canonical episode id, original NPZ RGB frame index).\n')
 print(json.dumps({k:v for k,v in summary.items() if k not in ['episodes','thresholds']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--temporal',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.temporal,a.output)
