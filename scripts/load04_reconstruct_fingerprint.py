"""Forensically explain original composite hashes; no new executions."""
from load04_common import *
import numpy as np,re
D=OUT/'reset-diagnostic';m=read(OUT/'data/manifest.json')['episodes'];results=[]
for e in m:
 task=e['task'];a=np.load(D/(task+'-state.npz'));xml=(D/(task+'-model.xml')).read_text()
 prefix=hashlib.sha256()
 for k in ['qpos','qvel','qacc_warmstart','act','ctrl','mocap_pos','mocap_quat']:prefix.update(k.encode());prefix.update(a[k].tobytes())
 prefix.update(str(float(a['time'])).encode())
 match=re.search(r'(<statistic[^>]*meansize=")([^"]+)(")',xml);assert match
 before=xml[:match.start(2)].encode();after=xml[match.end(2):].encode()
 prefix.update(before);found=None
 for integer in range(240000,250001):
  value=format(integer/1000000,'.6g')
  h=prefix.copy();h.update(value.encode());h.update(after)
  if h.hexdigest()==e['initial_fingerprint']:
   found=value;break
 assert found is not None,('cannot reproduce original fingerprint',task)
 recovered=xml[:match.start(2)]+found+xml[match.end(2):]
 (D/(task+'-original-recovered-model.xml')).write_text(recovered)
 results.append({'task':task,'original_fingerprint':e['initial_fingerprint'],'reproduced_exactly':True,'recovered_original_statistic_meansize':found,'diagnostic_statistic_meansize':match.group(2),'reconstruction':'Saved diagnostic arrays and XML; only statistic meansize varied; exact full SHA-256 match binds all other bytes to the original composite fingerprint.'})
dump(D/'original-fingerprint-reconstruction.json',{'passed':True,'results':results,'physical_state_bytes_bound_to_original':True,'only_missing_original_XML_attribute_recovered':'statistic.meansize','time':time.time()})
print(json.dumps(results,indent=2))
