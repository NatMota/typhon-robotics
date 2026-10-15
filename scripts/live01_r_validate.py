"""Read-only validation of retained evidence; no simulator creation."""
import sys,pickle,gzip,json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/"scripts")]
from live01_r_capture_v2 import different
from supervisor.planning.rgb_lineage import digest
from obs06_aperture import Observer
BASE=Path(sys.argv[1]);D=BASE/"attempt-02"
def pkl(p):
 with gzip.open(p,"rb") as f:return pickle.load(f)
def read(p):return json.loads(p.read_text())
def diff(a,b):
 d=b.astype(np.int16)-a.astype(np.int16);ys,xs=np.nonzero(np.any(d!=0,axis=2))
 return {"changed_pixels":len(ys),"maximum_channel_difference":int(np.abs(d).max()),"changed_xy":[[int(x),int(y)] for y,x in zip(ys,xs)]}
def main():
 bundles=pkl(D/"captures.pkl.gz");pairs=pkl(D/"state-pairs.pkl.gz")
 baseline=pkl(D/"qualified-state.pkl.gz");first=bundles[0]
 assert len(bundles)==len(pairs)==24
 for b,p in zip(bundles,pairs):
  assert not different(baseline,p["before"]) and not different(baseline,p["after"])
  assert not different(first["observer"],b["observer"])
  assert b["observer"]["temporal_updates_at_tick190"]==1
  assert b["observer"]["estimate_bundle"]==0 and b["observer"]["estimate_control_tick"]==190
  for cam,ref in b["observer"]["input_frame_references"].items():
   assert ref["content_sha256"]==digest(first["rgb"][cam])
   assert ref["audit_capture_id"]==next(g["audit_capture_id"] for g in first["graphics"] if g["camera"]==cam)
 audit_ids=[g["audit_capture_id"] for b in bundles for g in b["graphics"]];assert len(audit_ids)==len(set(audit_ids))==48
 real_ids=[m["capture_id"] for b in bundles if b["mode"]=="P" for m in b["lineage"].values()]
 assert len(real_ids)==len(set(real_ids))==24
 slots=0
 for b in bundles:
  if b["mode"]!="P":continue
  meta=b["prepared"]["lineage"]
  assert meta["image_mask"]==b["prepared"]["arrays"]["image_mask"].tolist()
  for slot in meta["slots"]:
   cam=slot["capture"]["camera"]
   assert slot["capture"]["capture_id"]==b["lineage"][cam]["capture_id"]
   assert slot["tensor_sha256"]==digest(b["prepared"]["arrays"]["image_input"][0,slot["slot"]])
   slots+=1
 events=read(D/"lineage-events.json")
 assert sum(x["event"]=="capture" for x in events)==24
 assert sum(x["event"]=="publication" for x in events)==12
 assert sum(x["event"]=="observation" for x in events)==12
 # A unit-only reproduction of attempt01's call-schedule error; observer is unchanged.
 e={"span_m":.03,"contour":(False,False),"closed":True}
 o=Observer();o.update(e,190)
 try:o.update(e,190)
 except ZeroDivisionError:duplicate_time_reproduced=True
 else:raise AssertionError("Unexpected observer contract")
 o=Observer();o.update(e,190);o.update(e,191)
 prev=pkl(BASE/"attempt-01/captures.pkl.gz")[0]
 exploratory={"label":"separate failed-attempt first N vs corrected-attempt first N; not part of24-bundle matrix","camera_differences":{c:diff(prev["rgb"][c],first["rgb"][c]) for c in ["primary","secondary"]},"available_graphics_fields_equal":all(next(g["graphics"] for g in prev["graphics"] if g["camera"]==c)==next(g["graphics"] for g in first["graphics"] if g["camera"]==c) for c in ["primary","secondary"]),"causal_attribution":None,"different_processes_and_test_composition":True}
 out={"passed":True,"physical_controller_rng_before_after_pairs":24,"unique_common_audit_capture_ids":48,"unique_actual_P_capture_ids":24,"P_preprocessed_slots_resolved":slots,"cached_observer_references_match_original_bundle":True,"temporal_observer_updates_at_frozen_tick":1,"duplicate_timestamp_error_reproduced_offline":duplicate_time_reproduced,"positive_delta_timestamp_unit_check_passed":True,"exploratory_cross_attempt_comparison":exploratory,"no_extra_rendering_physics_or_model_calls":True}
 (D/"evidence-validation.json").write_text(json.dumps(out,indent=2)+"\n")
 original=read(BASE/"attempt-01-preservation.json")
 for n,sha in original["files"].items():assert hashlib.sha256((BASE/"attempt-01"/n).read_bytes()).hexdigest()==sha
 src=read(D/"source-hashes.json")
 for f,sha in src.items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==sha
 (D/"preservation-validation.json").write_text(json.dumps({"passed":True,"first_attempt_original_files_verified":len(original["files"]),"run_sources_verified":len(src)},indent=2)+"\n")
 print(json.dumps(out,indent=2))
if __name__=="__main__":main()
