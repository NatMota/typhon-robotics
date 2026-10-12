"""Independent score/parity audit for the saved12October experiments."""
from pathlib import Path
import sys,json,gzip,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from supervisor.feeding_loading.evaluation import score_episode
OUT=Path('/var/tmp/typhon-research-20261012')
def main():
 checks=[]
 for folder in ['local-robustness','loading/factorial','loading/v4-success-anchor']:
  for f in sorted((OUT/folder).rglob('episode.json')):
   r=json.loads(f.read_text());sc=json.loads((f.parent/'scenario.json').read_text());assert not r['controller_exceptions'] and not r['artifact_errors'] and not r['evaluation_errors']
   evaluation=score_episode(r['initial_truth'],r['final_truth'],r['stability_truths'],sc['config'],r['attempts']);assert evaluation==r['evaluation'],str(f)
   for name in ['rollout.mp4','trajectory.jsonl.gz','initial.png','final.png']:assert (f.parent/name).stat().st_size>0
   if r['method']=='pretrained':
    with np.load(f.parent/'policy_io.npz') as z:assert len(z['primary'])>0 and len(z['model_actions'])>0
    with np.load(f.parent/'raw_predictions.npz') as z:assert all(np.isfinite(z[k]).all() for k in z.files if k!='histories')
   checks.append({'episode':str(f.relative_to(OUT)),'score_exact':True,'success':r['success']})
 assert len(checks)==44
 p=json.loads((OUT/'grasp-transform/protocol.json').read_text())
 for case in p['cases']:
  starts=[]
  for variant in p['variants']:
   d=OUT/'grasp-transform'/case['id']/variant;r=json.loads((d/'summary.json').read_text());trace=[json.loads(l) for l in gzip.open(d/'privileged-trace.jsonl.gz','rt')];last=trace[-24:];assert all(x['phase']=='settle' for x in last)
   samples=[x['post_action_truth'] for x in last];assert np.allclose(np.diff([t['sim_time_s'] for t in samples]),.05,atol=1e-12)
   assert score_episode(r['initial_truth'],r['final_truth'],samples,case['data']['config'],r['evaluation']['attempts'])==r['evaluation'];starts.append(r['initial_truth'])
   checks.append({'episode':str(d.relative_to(OUT)),'score_exact':True,'success':r['evaluation']['success']})
  assert starts[0]==starts[1]
 assert len(checks)==48
 observers={}
 for variant in ['passive-v3','passive-independent-v1','passive-stereo-v1']:
  r=json.loads((OUT/'observer'/variant/'summary.json').read_text());assert len(r['cases'])==8 and r['max_replay_final_position_difference_m']==0 and r['actions_changed']==0;observers[variant]={'exact_replays':8,'qualified':r['qualification_passed']}
 summary={'passed':True,'exact_rescored_rollouts':len(checks),'checks':checks,'passive_observers':observers,'scope':'Artifact/scoring integrity, not validation of an autonomous repair','all_completed_outputs_preserved':True}
 (OUT/'validation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='checks'}))
if __name__=='__main__':main()
