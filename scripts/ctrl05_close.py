"""CTRL05 closure: preserve failed qualification without assigning S/R outcomes."""
from pathlib import Path
import json,gzip
O=Path("/var/tmp/typhon-ctrl05-20261015");A=Path("/var/tmp/typhon-ctrl04-20261015/E80")
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in gzip.open(p,"rt")]
diff=read(O/"S/historical510-comparison.json")
expected={"/python/wrapper/move_to/code_sha256","/python/wrapper/step/code_sha256"}
assert {x["path"] for x in diff["differences"]}==expected
counts={}
for name in ["controller_trace.jsonl.gz","trajectory.jsonl.gz"]:
 old={x["step"]:x for x in rows(A/"continuation"/name)};current=[x for x in rows(O/"S/continuation"/name) if x["step"]<=510]
 assert all(x==old[x["step"]] for x in current),name
 counts[name]=len(current)
old={x["step"]:x for x in rows(A/"extra-measurements.jsonl.gz")}
m=rows(O/"S/extra-measurements.jsonl.gz");assert all(x==old[x["step"]] for x in m)
result={"experiment":"CTRL05","status":"qualification_blocker","physical_reconstructions_started":1,"expert_approach_ticks":50,
 "S_descent_actions":0,"R_descent_actions":0,"evaluated_conditions":0,"success_rate":None,"full_historical590_equality_claim":False,
 "observed_prefix_equality_counts":counts,"extra_measurement_rows_exact":len(m),
 "only_mismatches":sorted(expected),"semantic_comparator_passed":False,
 "new_model_queries":0,"new_training":False,"holdout_use":False,
 "runtime_stop":"tick510 before added hold or either descent condition; unchanged24tick passive tail retained",
 "interpretation":"Wrapper code identities changed before the required historical boundary. This is an implementation/qualification failure, not a result about smooth descent.",
 "disposition":"No retry or comparator amendment. Park nominal approach-repair branch for remainder of overnight. Prospective follow-up must retain original wrappers until qualified boundary.",
 "audit":"reference-actuation-audit.json"}
(O/"validation-summary.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
