from pathlib import Path
import sys,json,hashlib
import numpy as np
w=Path.cwd();sys.path.insert(0,str(w))
from supervisor.feeding_loading.observations import policy_observation
from supervisor.planning.telemetry_provenance import TelemetryProvenance
out=Path('/var/tmp/typhon-live01-t-20261013')
events=[json.loads(l) for l in (out/'labelled-film-demo_0006-audit/provenance-events.jsonl').read_text().splitlines()]
prefix=events[:4]
raw={'primary':np.arange(48,dtype=np.uint8).reshape(4,4,3),'secondary':np.arange(48,96,dtype=np.uint8).reshape(4,4,3)}
ctx=TelemetryProvenance.__new__(TelemetryProvenance);ctx.frames={}
for name,im in raw.items():ctx.bind_frame(im,{'frame_id':'probe-'+name,'camera':name})
copied=policy_observation(raw,{},{});checks={}
for name,im in raw.items():
 other=copied['rgb'][name]
 checks[name]={'raw_has_identity':ctx.frame_info(im) is not None,
 'output_has_identity':ctx.frame_info(other) is not None,'same_object':im is other,
 'same_pixels':bool(np.array_equal(im,other)),'shares_memory':bool(np.shares_memory(im,other)),
 'raw_sha256':hashlib.sha256(im.tobytes()).hexdigest(),'output_sha256':hashlib.sha256(other.tobytes()).hexdigest()}
assert all(c['raw_has_identity'] and not c['output_has_identity'] and c['same_pixels'] and not c['same_object'] for c in checks.values())
p=w/'supervisor/feeding_loading/observations.py'
result={'scope':'CPU identity probe of real policy_observation(), no policy inference or simulator execution',
 'source_file':str(p.relative_to(w)),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
 'copy_statement':"'rgb': {name: np.asarray(frame, dtype=np.uint8).copy() for name, frame in rgb.items()}",
 'checks':checks,'first_live_counterexample':prefix,
 'conclusion':'Render ndarray identity is not propagated across policy_observation copy. The new registry therefore cannot bind acquisition metadata to returned observation arrays or their history copies.',
 'candidate_status':'failed; not repaired or rerun','proposed_followup':'Bind returned arrays to per-view render records captured inside the same existing observe call, validating unchanged pixels. Add an adapter-copy integration fixture before a separately frozen live qualification.'}
(out/'copy-boundary-counterexample.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='first_live_counterexample'},indent=2))
