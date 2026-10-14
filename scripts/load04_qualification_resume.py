"""Resume only after qualifying the explained XML-only discrepancy."""
from load04_data import canonical_fingerprint
from load04_common import *
import numpy as np,shutil
d=OUT/'reset-diagnostic';q=read(d/'original-fingerprint-reconstruction.json');c=read(d/'report.json')
r=read(OUT/'reset-component-qualification/report.json')
assert q['passed'] and r['controller_fields_exact'] and all(r['compiled_render_equality'].values()) and r['config_differing_top_level_fields']==['task']
assert all(c['component_equality'].values()) and all(c['model_array_equality'].values())
assert time.time()<read(OUT/'clock.json')['data_deadline']
manifest=read(OUT/'data/manifest.json')
shutil.copyfile(OUT/'data/manifest.json',OUT/'data/manifest-initial-stop.json')
for e in manifest['episodes']:
 task=e['task'];a=np.load(d/(task+'-state.npz'));xml=(d/(task+'-original-recovered-model.xml')).read_text()
 h=hashlib.sha256()
 for k in ['qpos','qvel','qacc_warmstart','act','ctrl','mocap_pos','mocap_quat']:h.update(k.encode());h.update(a[k].tobytes())
 h.update(str(float(a['time'])).encode());h.update(xml.encode());assert h.hexdigest()==e['initial_fingerprint']
 e['initial_canonical_fingerprint']=canonical_fingerprint(a,xml)
 e['initial_controller']=read(OUT/'reset-component-qualification'/(task+'-controller.json'))
 e['initial_controller_provenance']='same-sequence reset reconstruction; original sim.data.ctrl proven by original composite fingerprint reconstruction'
assert manifest['episodes'][0]['initial_canonical_fingerprint']==manifest['episodes'][1]['initial_canonical_fingerprint']
assert manifest['episodes'][0]['initial_controller']==manifest['episodes'][1]['initial_controller']
dump(OUT/'data/manifest.json',manifest)
shutil.copyfile(OUT/'data/generation-gate.json',OUT/'data/generation-gate-initial-stop.json')
addendum={'resume_qualified':True,'original_clock_unchanged':True,'original_two_trajectories_retained':True,'extra_expert_or_policy_executions':0,'reset_only_diagnostic_instances':4,
 'explained_difference':'Only XML statistic.meansize: original0.244383 loading,0.244374 feeding. Exact original full SHA256 fingerprints reconstructed. All other original hashed bytes agree with saved diagnostics.',
 'exemption':'Canonical comparison removes ONLY statistic.meansize. Full source fingerprints/XML preserved. Runtime simulator not edited.',
 'documentation':'https://mujoco.readthedocs.io/en/2.3.7/XMLreference.html#statistic','state_checks':c,'controller_and_render_checks':r,'original_fingerprint_reconstruction':q,
 'controller_evidence_limit':'Original sim.data.ctrl proven byte-for-byte; higher-level controller dictionaries verified in identical-sequence resets, not individually saved for the first pair.',
 'planner':'PLANNER_QUALIFICATION.md permits remaining six original executions after serialization-only proof tied to original evidence, with original clock and trajectories retained.','time':time.time()}
dump(OUT/'qualification-addendum.json',addendum)
history=R/'experiment-history/2026-10-14-load04'
for n in ['qualification-addendum.json','reset-diagnostic/original-fingerprint-reconstruction.json','reset-diagnostic/report.json','reset-component-qualification/report.json']:
 shutil.copyfile(OUT/n,history/n.replace('/','-'))
print('RESUME_QUALIFIED',time.time())
