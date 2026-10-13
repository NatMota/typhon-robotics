"""Four fixed consumer fixtures from saved LIVE01 payloads; no physics."""
from pathlib import Path
import copy,gzip,hashlib,json,sys
from types import SimpleNamespace
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from supervisor.planning.telemetry_provenance import TelemetryProvenance,TelemetryConsumer
SOURCE=Path("/var/tmp/typhon-live01-shadow-20261013/shadow-film-demo_0006-audit/publisher.jsonl.gz")
OUT=R/"configs/experiments/live01-t"
def metadata(payload):
 # Historical acquisition timestamps were not captured. Keep them unknown.
 tick=payload["control_step"]
 clock={"control_tick":tick,"control_time_s":tick/20,"policy_origin_control_tick":40,
        "policy_tick":tick-40,"policy_time_s":(tick-40)/20,
        "simulation_time_at_read_s":None,"physics_step_identity":None}
 refs=[{"source":{"frame_id":f"fixture-archive-publication-{payload['sequence']}-{camera}",
   "camera":camera,"acquisition_interval_monotonic_s":None,"clock":clock,
   "identity_status":"fixture identity derived from snapshot; acquisition was not archived"},
   "usage":"current","acquisition_status":"unknown"} for camera in ["primary","secondary"]]
 ctx=TelemetryProvenance.__new__(TelemetryProvenance)
 ctx.clock=lambda:clock
 ctx.obs_meta=lambda _:{"frames":refs,"clock":clock,"proprio_read_interval_monotonic_s":None}
 ctx.command={"control_tick":tick,"source_mode":"settling" if tick>=1477 else "policy_execution",
    "applied_workspace_adjusted_normalized_osc":copy.deepcopy(payload["command"]["applied_action"]),
    "policy_target_active_as_command_source":tick<1477}
 ctx.origin=40;ctx.observer=None
 ctx.target={"kind":"latest reconstructed policy target","query_id":None,"waypoint_index":None,
   "generation_interval_monotonic_s":None,"last_selection_control_tick":None,
   "provenance_status":"unknown in historical snapshot; no retroactive certification"}
 return ctx.publish_metadata(None,None,SimpleNamespace(sequence=payload["sequence"]))
def main():
 rows=[json.loads(l) for l in gzip.open(SOURCE,"rt")]
 last=rows[-1];first=rows[0];fixtures=[]
 def labelled(p):
  result=copy.deepcopy(p);result["source_contract"]=metadata(p)
  assert {k:v for k,v in result.items() if k!="source_contract"}==p
  return result
 p=labelled(last);r=TelemetryConsumer().consume(p)["incoming"]
 assert r["command_source"]=="settling" and not r["last_target_active"]
 assert r["effective_command"]==last["command"]["applied_action"]
 fixtures.append({"id":"settling_tail","inputs":[p],"expected":{"command_source":"settling","last_target_active":False},"actual":r})
 c=TelemetryConsumer();a=labelled(last);b=copy.deepcopy(a)
 b["source_contract"]["publication"]["sequence"]+=1;b["source_contract"]["publication"]["monotonic_s"]+=1
 c.consume(a);r=c.consume(b)["incoming"]
 assert r["same_frames_as_previous"] and r["acquisition_age_status"]=="unknown"
 assert a["source_contract"]["rgb"]==b["source_contract"]["rgb"]
 fixtures.append({"id":"repeat_image","inputs":[a,b],"expected":{"same_frames_as_previous":True,"acquisition_age_status":"unknown"},"actual":r})
 c=TelemetryConsumer();a=labelled(last);b=labelled(first)
 c.consume(a);r=c.consume(b)
 assert r["incoming"]["out_of_order"] and r["latest"]["source_control_tick"]==last["control_step"]
 fixtures.append({"id":"out_of_order","inputs":[a,b],"expected":{"out_of_order":True,"latest_tick":last["control_step"]},"actual":r})
 p=labelled(first);r=TelemetryConsumer().consume(p)["incoming"]
 assert r["integrated_phase"]!=r["derived_phase"] and r["simultaneity_known"] is False
 assert p["source_contract"]["derived_wrist_site_pose"]["computation_time_s"] is None
 fixtures.append({"id":"cached_vs_integrated","inputs":[p],"expected":{"simultaneity_known":False,"site_computation_time":"unknown"},"actual":r})
 report={"source":str(SOURCE),"source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
  "fixture_scope":"Saved numerical snapshots, newly fabricated delivery ordering only. Historical acquisition/generation/compute times remain unknown.",
  "tests_passed":4,"legacy_numerical_fields_unchanged":True,"control_authority":False,"fixtures":fixtures}
 (OUT/"consumer-fixtures.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
 print(json.dumps({"tests_passed":4,"legacy_numerical_fields_unchanged":True,"source_sha256":report["source_sha256"]}))
if __name__=="__main__":main()
