import pathlib,numpy as np,json,hashlib
root=pathlib.Path('/var/tmp/typhon-load06-20261014');donors={'B':pathlib.Path('/var/tmp/typhon-load05r-20261014/continuations/B-to-B/predictions.npz'),'C':pathlib.Path('/var/tmp/typhon-load05r2-20261014/continuations/B-to-C/predictions.npz')};chunks={m:np.load(p)['chunks'][0] for m,p in donors.items()}
conditions={}
for key in ['BB','BC','CB','CC']:
 x=chunks['B'].copy();x[:,:9]=chunks[key[0]][:,:9];x[:,9]=chunks[key[1]][:,9];conditions[key]=x
np.savez_compressed(root/'donor-chunks.npz',**conditions)
pose=np.flatnonzero(np.any(chunks['B'][:,:9]!=chunks['C'][:,:9],axis=1));continuous=np.flatnonzero(chunks['B'][:,9]!=chunks['C'][:,9]);applied=np.flatnonzero((chunks['B'][:,9]>.5)!=(chunks['C'][:,9]>.5))
report={'sources':{m:{'path':str(p),'file_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'first_chunk_sha256':hashlib.sha256(chunks[m].tobytes()).hexdigest()} for m,p in donors.items()},'first_pose_difference_tick':101+int(pose[0]) if len(pose) else None,'first_continuous_grip_difference_tick':101+int(continuous[0]) if len(continuous) else None,'applied_grip_difference_ticks':(applied+101).tolist(),'identical_applied_grip_through_tick110':bool(np.array_equal(chunks['B'][:10,9]>.5,chunks['C'][:10,9]>.5)),'first_close':{m:101+int(np.flatnonzero(x[:,9]>.5)[0]) if np.any(x[:,9]>.5) else None for m,x in chunks.items()},'duplicate_conditions':[],'inactive_arm':'B donor channels10:20 retained unchanged; executor ignores inactive arm','handoff':'Previous-target history first9 from actual executed chunk; other channels unchanged per native convention'}
for i,a in enumerate(conditions):
 for b in list(conditions)[i+1:]:
  x,y=conditions[a],conditions[b]
  if np.array_equal(x[:,:9],y[:,:9]) and np.array_equal(x[:,9]>.5,y[:,9]>.5):report['duplicate_conditions'].append([a,b])
(root/'donor-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
