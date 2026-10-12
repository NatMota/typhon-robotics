"""Descriptive audit of the observed LIVE07 divergence; no new policy run."""
import sys,json,gzip,pickle
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);sid='film-demo_0006'
def packed(p):
 with gzip.open(p,'rb') as f:return pickle.load(f)
def trace(p):
 with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]
a=root/'first_admissible'/sid;b=root/'decisions'/sid
qa,qb=[packed(p/'proposal-0000.pkl.gz') for p in [a,b]]
np.testing.assert_array_equal(qa['response']['actions'],qb['response']['actions'])
np.testing.assert_array_equal(qa['response']['initial_noise'],qb['response']['initial_noise'])
ta,tb=[trace(p/'trajectory.jsonl.gz') for p in [a,b]]
first=next((i for i,(x,y) in enumerate(zip(ta,tb)) if x!=y),None)
raw=json.loads((b/'ranking/00000/result.json').read_text())
actions=qa['response']['actions'];g0=actions[0,:,9];g2=actions[2,:,9]
events=[[json.loads(line) for line in (p/'events.jsonl').read_text().splitlines()] for p in [a,b]]
v={'scene':sid,'initial_candidate_sets_and_noises_match_exactly':True,'identical_control_and_truth_prefix_steps':first,
 'first_divergence_control_index':first,'first_divergence_simulator_step':ta[first]['step'] if 'step' in ta[first] else None,
 'first_selected_live_api_result':raw,
 'A_C_target_xyz_separation_mm':{'median':float(np.median(np.linalg.norm(actions[0,:,:3]-actions[2,:,:3],axis=1))*1000),'max':float(np.max(np.linalg.norm(actions[0,:,:3]-actions[2,:,:3],axis=1))*1000)},
 'A_C_all_first_chunk_gripper_commands_equal':bool(np.array_equal(g0,g2)),
 'initial_activated':[[r for r in ev if r['event']=='activated'][0] for ev in events],
 'activation_wait_ticks':{mode:[r['held_steps'] for r in ev if r['event']=='activated'] for mode,ev in zip(['first_admissible','decisions'],events)},
 'interpretation':'First actual control divergence follows the executed A versus C choice with an identical prior trajectory. Later wait schedules differ; final success causality still needs fixed-timing intervention. Low API confidence is descriptive and no threshold was fitted.'}
(root/'first-divergence-audit.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(v,indent=2))
