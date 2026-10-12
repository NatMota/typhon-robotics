"""Offline audit of the exact LIVE04 preview evidence, without new API calls."""
import os,sys,json,gzip,pickle,math,itertools,csv,collections
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));os.chdir(R)
import numpy as np
from scipy.spatial.transform import Rotation
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.planning.decisions_ranker import project
from supervisor.baseline.benchmark import save
from live04_boundary import decode
OLD=Path('/var/tmp/typhon-live04-boundary-20261012-v3')
OUT=Path('/var/tmp/typhon-live05-isolation-20261012-v2/preview-audit')
OUT.mkdir(exist_ok=False)
protocol={'audit':'Geometry of the exact accepted candidate paths shown to LIVE04; no API calls, future simulator outcome scores or new candidate generation.',
 'metric':'Mean/max per-timepoint projected EEF separation at320px thumbnail size, plus gripper information actually transmitted.',
 'diagnostic_reference_px':2,'interpretation':'2px is a display-scale diagnostic, not a psychometric or API-discrimination threshold. Near-overlapping paths do not establish equivalent physical outcome.',
 'limitations':['Only tool trajectories were displayed, not ghost arms or simulated object futures','No independent labels of candidate usefulness','Prompt gives gripper command atfirst/middle/last, not validated contact','Need controlled feasible-vs-infeasible calibration before interpreting UNKNOWN as model failure']}
save(OUT/'protocol.json',protocol)
rows=[];requests=[]
for scenario in json.loads((OLD/'protocol.json').read_text())['scenes']:
 sid=scenario['scenario_id'];env=FeedingLoadingEnv(scenario['config']);env.reset(scenario['seed'])
 cameras=env.observe()['cameras'];env.close()
 base=OLD/'decisions'/sid;submitted=[json.loads(x) for x in (base/'ranking/events.jsonl').read_text().splitlines() if json.loads(x)['type']=='submitted']
 for event in submitted:
  index=int(event['context'].rsplit(':',1)[1]);folder=base/'ranking'/f"{event['index']:05d}"
  with gzip.open(base/f'proposal-{index:04d}.pkl.gz','rb') as f:packet=pickle.load(f)
  actions=decode(packet['response'])['actions'];labels=list(event['candidate_sha256'])
  outcome=json.loads((folder/'result.json').read_text())
  close={label:[bool(actions[ord(label)-65,i,9]>.5) for i in [0,15,29]] for label in labels}
  item={'scenario_id':sid,'request_index':event['index'],'proposal':index,'choice':outcome['choice'],'n_candidates':len(labels),'same_displayed_gripper_commands':len({tuple(v) for v in close.values()})==1}
  pairrows=[]
  for left,right in itertools.combinations(labels,2):
   a,b=actions[ord(left)-65],actions[ord(right)-65]
   for camera,calib in cameras.items():
    scale=320/max(calib['width'],calib['height'])
    delta=np.linalg.norm(project(a[:,:3],calib)-project(b[:,:3],calib),axis=1)*scale
    row={**item,'pair':left+right,'camera':camera,'mean_path_separation_px':float(np.mean(delta)),
     'max_path_separation_px':float(np.max(delta)),'final_target_separation_px':float(delta[-1]),
     'fraction_samples_within_2px':float(np.mean(delta<=2)),
     'shown_gripper_equal':close[left]==close[right],
     'source_preview':str(folder/'candidates.jpg')}
    rows.append(row);pairrows.append(row)
  item['maximum_pair_mean_separation_px']=max(r['mean_path_separation_px'] for r in pairrows)
  item['all_pairs_mean_within_2px']=all(r['mean_path_separation_px']<=2 for r in pairrows)
  requests.append(item)
with (OUT/'pairs.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
save(OUT/'requests.json',requests)
summary={'requests':len(requests),'pair_view_comparisons':len(rows),'choices':dict(collections.Counter(r['choice'] for r in requests)),
 'all_pair_means_within_2px_requests':sum(r['all_pairs_mean_within_2px'] for r in requests),
 'same_displayed_gripper_commands_requests':sum(r['same_displayed_gripper_commands'] for r in requests),
 'mean_path_separation_px_p50_p95':np.quantile([r['mean_path_separation_px'] for r in rows],[.5,.95]).tolist(),
 'fraction_pair_views_mean_within_2px':float(np.mean([r['mean_path_separation_px']<=2 for r in rows])),
 'evidence_type':'Preview geometry audit; no ranking-accuracy claim.'}
save(OUT/'summary.json',summary);print(json.dumps(summary),flush=True)
