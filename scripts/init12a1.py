"""INIT12-A1: only the ten unexecuted cases; original failure and clock retained."""
import os,sys,time,json,gzip,traceback,hashlib
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import numpy as np
import init12 as base
from init12_pair_audit import pair,within_run,self_check
O=Path("/var/tmp/typhon-init12-a1-20261016")
ORIGINAL=Path("/var/tmp/typhon-init12-20261016")
read,dump,sha=base.read,base.dump,base.sha

def summaries(row,source,out):
    condition=base.CONDITION
    row=base.old.extra(row,source,"Q1",out)
    ep=read(out/"episode.json")
    ep.update(experiment="INIT12-A1",protocol_revision="A1_after22known_outcomes",
              condition=condition,harness=condition=="D5",TTC=False,training=False,
              correction_data_admission=False,original_clock_retained=True)
    dump(out/"episode.json",ep)
    rule=base.LAST_POLICY.rule.summary()
    expected=0 if rule["trigger"]is None else min(60,1436-rule["trigger"]["policy_action_zero_based"])
    assert rule["modified_actions_executed"]==expected
    dump(out/"rule-summary.json",rule)
    with gzip.open(out/"controller-force-trace.jsonl.gz","rt")as f:forces=[json.loads(x)for x in f]
    forbidden=[]
    for step in forces:
        for c in step["force_telemetry"]["contacts"]:
            a,b=str(c["geom1"]),str(c["geom2"])
            if ("gripper"in a and("table"in b or"fixture"in b))or("gripper"in b and("table"in a or"fixture"in a)):
                forbidden.append({"step":step["step"],"geom1":a,"geom2":b})
    row.update(condition=condition,rule=rule,forbidden_contact_samples=len(forbidden),
       forbidden_contact_first=forbidden[0]if forbidden else None,
       forbidden_contact_pairs=sorted(set((c["geom1"],c["geom2"])for c in forbidden)),
       workspace_clipped_steps=sum(bool(r["policy_action"]and r["policy_action"]["workspace_clipped"])for r in forces),
       selected_previous_rescue=source["id"]in base.SELECTED,protocol_revision="INIT12-A1")
    dump(out/"within-run-qualification.json",within_run(out,rule))
    if condition=="D5":
        native=O/"episodes/N"/source["id"]
        base.array_file_equal(native/"input-audit/initial-prepared.npz",out/"input-audit/initial-prepared.npz")
        assert(native/"boundary-before.pkl").read_bytes()==(out/"boundary-before.pkl").read_bytes()
        base.h.base.noise_gate(out,source["policy_seed"],native)
        audit=pair(native,out,rule)
        audit.update(initial_semantic_state_exact=True,initial_prepared_inputs_exact=True,all48_noise_packets_exact=True,
                     amendment="A1",original_exact_gate_not_retroactively_changed=True)
        dump(out/"paired-gate.json",audit);row["pairing_status"]=audit["status"]
    dump(out/"summary.json",row)
    return row

def main():
    f=read(O/"execution-freeze.json");assert f["private_push_verified"]
    assert sha(ORIGINAL/"STOPPED.json")==f["original_failure_sha256"]
    for p,d in f["original_artifact_sha256"].items():assert sha(ORIGINAL/p)==d,p
    for p,d in f["source_sha256"].items():assert sha(base.W/p)==d,p
    cache=base.W/".cache-xvla/checkpoint"
    assert str(cache.resolve())==f["restored_model_cache"]["resolved_path"]
    for p,d in f["restored_model_cache"]["source_sha256"].items():assert sha(cache/p)==d,p
    assert sha(O/"remaining-cases.json")==f["remaining_cases_sha256"]
    cases=read(O/"remaining-cases.json");assert len(cases)==5
    original_cases=read(ORIGINAL/"evaluation-cases.json")["core"]
    expected=[s for s in original_cases if s["kind"]=="prefix"and not(ORIGINAL/"episodes/N"/s["id"]/"episode.json").exists()]
    assert cases==expected
    assert all(not(ORIGINAL/"episodes"/c/s["id"]/"episode.json").exists()for s in cases for c in ["N","D5"])
    oldclock=read(ORIGINAL/"clock.json");clock=read(O/"clock.json")
    assert clock["activated_epoch"]==oldclock["activated_epoch"] and clock["archive_deadline"]==oldclock["archive_deadline"]
    assert clock["main_deadline"]==oldclock["retention_deadline"]
    base.O=O;base.PHASE="main";base.h.O=O;base.chain.O=O
    base.h.limit=base.limit;base.initial.limit=lambda*a:base.limit()
    base.ev.model_entry=base.model_entry;base.old.model_entry=base.model_entry
    base.h.Policy=base.Policy;base.summaries=summaries
    dump(O/"audit-unit-tests.json",self_check())
    # Reclassify the saved anomaly only as an A1 diagnostic, never overwrite its original failed gate.
    case="init-demo_0003-feeding-912171"
    original_rule=read(ORIGINAL/"episodes/D5"/case/"rule-summary.json")
    audit=pair(ORIGINAL/"episodes/N"/case,ORIGINAL/"episodes/D5"/case,original_rule)
    assert not audit["exact_effective_pretrigger_equivalence"]
    audit.update(original_exact_qualification="FAILED_UNCHANGED",new_physics=False,new_model_calls=0)
    dump(O/"original-failure-amendment-audit.json",audit)
    entry=base.model_entry("Q1");assert sha(entry["checkpoint"])==base.Q1_SHA and sha(entry["probe"])==base.PROBE_SHA
    (O/"worker").mkdir();base.h.base.OUT=O/"worker";base.h.base.native.OUT=O
    w=base.h.base.Worker({"model":entry,"expected_initial_nontext":{}})
    rows=[]
    try:base.run_panel(w,cases,rows)
    finally:w.close()
    assert len(rows)==10
    for p,d in f["original_artifact_sha256"].items():assert sha(ORIGINAL/p)==d,p
    dump(O/"physical-complete.json",{"episodes":10,"epoch":time.time(),"original_outputs_unchanged":True,
          "original_clock_retained":True,"conditional_prior_episodes":0,"original_advancement_gate_passed":False,
          "reason_not_advanced":"Previously observed new forbidden fixture contacts remain disqualifying",
          "no_outcome_retries":True,"training":False})
    dump(O/"retention-not-run.json",{"not_executed":14,"reason":"Original forbidden-contact criterion already failed","no_substitute_panel":True})

if __name__=="__main__":
    try:main()
    except BaseException:
        dump(O/"STOPPED.json",{"epoch":time.time(),"error":traceback.format_exc(),"technical_failure_not_policy_score":True,"outcome_retries":0})
        raise
