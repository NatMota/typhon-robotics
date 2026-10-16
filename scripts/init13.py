"""INIT13: eight paired episodes; unchanged D5 plus a measured-open clearance primitive."""
import os,sys,json,time,gzip,traceback,shutil,hashlib
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path[:0]=[str(W),str(W/"scripts")];os.chdir(W)
os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("PYOPENGL_PLATFORM","egl")
import numpy as np
import init12 as base
from init13_clearance import Geometry,Clearance,self_check
from init12_pair_audit import pair,lines
from init11_contract import history
O=Path("/var/tmp/typhon-init13-20261016")
read,dump,sha=base.read,base.dump,base.sha
CONDITION="D5";PHASE="preparation";LAST=None
def limit():
    assert time.time()<read(O/"clock.json")[PHASE+"_deadline"],("INIT13 deadline",PHASE)
class Policy(base.Policy):
    def __init__(self,*args):
        super().__init__(*args)
        global LAST
        LAST=self;self.acquisition_chunks=[];self.clear_events=[]
    def instrument(self,env,out):
        result=super().instrument(env,out)
        before=base.chain.state_blob(env);m,d=base.diagnostics.rawmodel(env)
        self.geom=Geometry(m,env.env.robots[0].eef_site_id,self.mapping["qpos_ids"],self.mapping["joint_ids"],env.config)
        assert before==base.chain.state_blob(env)
        desc=self.geom.describe()
        ref=read(O/"eligibility-geometry-audit.json")["cases"][0]["geometry"]
        assert len(desc["geoms"])==len(ref["geoms"])
        for a,b in zip(desc["geoms"],ref["geoms"]):
            assert a["name"]==b["name"]and a["vertices"]==b["vertices"]
            np.testing.assert_allclose(a["open_local_min"],b["open_local_min"],rtol=0,atol=1e-10)
            np.testing.assert_allclose(a["open_local_max"],b["open_local_max"],rtol=0,atol=1e-10)
        assert desc["wall_top_m"]==ref["wall_top_m"]
        dump(out/"geometry-qualification.json",{"passed":True,"state_unchanged":True,"static_geometry_matches_eligibility_audit":True,**desc})
        self.clear_log=gzip.open(out/"clearance-trace.jsonl.gz","wt")
        raw=env.step
        def step(action):
            pending=self.pending_action
            raw(action)
            if pending is not None:
                m,d=base.diagnostics.rawmodel(env)
                gap=self.geom.current_min(d)-self.geom.wall_top
                self.clearance.executed(env.last_grip,gap,pending["index_zero_based"])
                row={"index":pending["index_zero_based"],"step":env.steps,"minimum_gripper_wall_clearance_m":gap,
                     "aperture_m":float(sum(abs(d.qpos[i])for i in self.mapping["qpos_ids"])),
                     "active_during_action":pending["clearance_active"],"state":self.clearance.summary()}
                self.clear_events.append(row);self.clear_log.write(json.dumps(row,allow_nan=False)+"\n")
        env.step=step
        return result
    def reset(self,seed):
        super().reset(seed);self.clearance=Clearance(CONDITION=="C")
    def predict(self,obs):
        result=super().predict(obs)
        self.acquisition_chunks.append(np.asarray(result).copy())
        self.issued_chunks[-1]=self.issued_chunks[-1].astype(np.float64)
        return result
    def to_action(self,target,obs,config):
        m,d=base.diagnostics.rawmodel(self.env);prop=obs["proprio"]
        aperture=float(sum(abs(d.qpos[i])for i in self.mapping["qpos_ids"]))
        issued,event=self.clearance.prepare(target,self.rule.executed,self.rule.modified==60,
                    self.env.last_grip,aperture,prop["eef_position_m"],prop["eef_quat_xyzw"],self.geom)
        action=super().to_action(issued,obs,config)
        self.issued_chunks[-1][self.raw_cursor-1]=issued
        self.pending_action.update(clearance_active=event["active"],clearance_triggered=event["triggered"],
                                   acquisition_target=np.asarray(target).tolist(),measured_aperture_before_m=aperture)
        return action
def close():
    base.close_policy()
    if LAST is not None and hasattr(LAST,"clear_log")and not LAST.clear_log.closed:LAST.clear_log.close()

def audit_run(out,rule,clearance):
    raw=np.load(out/"predictions.npz");issued=np.load(out/"issued-predictions.npz")["chunks"]
    acquisition=np.load(out/"acquisition-predictions.npz")["chunks"]
    rows=lines(out/"controller-force-trace.jsonl.gz");acts=[r["policy_action"]for r in rows if r["policy_action"]]
    assert len(rows)==1460 and len(acts)==1436
    flat=issued.reshape(-1,20)[:1436];acq=acquisition.reshape(-1,20)[:1436];original=raw["chunks"].reshape(-1,20)[:1436]
    np.testing.assert_array_equal([a["issued_target"]for a in acts],flat)
    np.testing.assert_array_equal([a["acquisition_target"]for a in acts],acq)
    np.testing.assert_array_equal([a["raw_model_target"]for a in acts],original)
    active=np.array([a["offset_active"]for a in acts]);clear=np.array([a["clearance_active"]for a in acts])
    assert not np.any(active&clear)and int(clear.sum())==clearance["overridden_actions"]
    np.testing.assert_array_equal(acq[:,[i for i in range(20)if i!=2]],original[:,[i for i in range(20)if i!=2]])
    np.testing.assert_array_equal((acq-original)[~active],0)
    np.testing.assert_allclose((acq-original)[active,2],-.005,rtol=0,atol=1e-15)
    np.testing.assert_array_equal(flat[~clear],acq[~clear])
    np.testing.assert_array_equal(flat[:,9:],original[:,9:])
    if clear.any():np.testing.assert_array_equal(flat[clear,:9],np.tile(clearance["trigger"]["target_pose"],(int(clear.sum()),1)))
    hs=raw["histories"];assert np.count_nonzero(hs[:,9:])==0
    for q in range(1,48):np.testing.assert_array_equal(hs[q],np.asarray(history(flat[q*30-1]),np.float32))
    assert int(active.sum())==rule["modified_actions_executed"]
    return {"passed":True,"policy_actions":1436,"D5_actions":int(active.sum()),"clearance_actions":int(clear.sum()),
       "history_all47_transitions_exact":True,"native_gripper_and_inactive_channels_exact":True,"no_clearance_without_D5":not clear.any()or rule["trigger"]is not None}

def finish(row,s,out):
    row=base.old.extra(row,s,"Q1",out);rule=LAST.rule.summary();clear=LAST.clearance.summary()
    np.savez_compressed(out/"issued-predictions.npz",chunks=np.asarray(LAST.issued_chunks))
    np.savez_compressed(out/"acquisition-predictions.npz",chunks=np.asarray(LAST.acquisition_chunks))
    with np.load(out/"policy_io.npz")as z:payload={k:z[k]for k in z.files}
    payload["model_actions"]=np.asarray(LAST.issued_chunks);np.savez_compressed(out/"policy_io.npz",**payload)
    dump(out/"rule-summary.json",rule);dump(out/"clearance-summary.json",clear)
    dump(out/"within-run-qualification.json",audit_run(out,rule,clear))
    ep=read(out/"episode.json");ep.update(experiment="INIT13",condition=CONDITION,harness=True,TTC=False,training=False,
       additional_controller_knowledge="Known robot collision geometry and fixed fixture wall height; no object truth in primitive",
       correction_data_admission=False)
    dump(out/"episode.json",ep)
    forbidden=[]
    for f in lines(out/"controller-force-trace.jsonl.gz"):
        for c in f["force_telemetry"]["contacts"]:
            a,b=c["geom1"],c["geom2"]
            if("gripper"in a and("table"in b or"fixture"in b))or("gripper"in b and("table"in a or"fixture"in a)):
                forbidden.append({"step":f["step"],"action":f["policy_action"]["index_zero_based"]if f["policy_action"]else None,
                   "geom1":a,"geom2":b,"distance_m":c["distance"],"wrench_contact":c["wrench_contact"]})
    row.update(condition=CONDITION,rule=rule,clearance=clear,forbidden_contact_samples=len(forbidden),forbidden_contacts=forbidden)
    if CONDITION=="C":
        ref=O/"episodes/D5"/s["id"]
        assert (ref/"boundary-before.pkl").read_bytes()==(out/"boundary-before.pkl").read_bytes()
        base.array_file_equal(ref/"input-audit/initial-prepared.npz",out/"input-audit/initial-prepared.npz")
        base.h.base.noise_gate(out,s["policy_seed"],ref)
        t=clear["trigger"];pseudo={"trigger":None if t is None else{"query_zero_based":t["policy_action_zero_based"]//30,"policy_action_zero_based":t["policy_action_zero_based"]}}
        p=pair(ref,out,pseudo);p.update(initial_state_and_prepared_exact=True,all48_noise_packets_exact=True,
            intervention="clearance after common D5; not native-vs-D5",RGB_first_limit="INIT12-A1 treatment retained")
        dump(out/"paired-gate.json",p);row["pairing"]=p
    dump(out/"summary.json",row);return row

def main():
    global PHASE,CONDITION,LAST
    freeze=read(O/"execution-freeze.json");assert freeze["private_push_verified"]
    for p,digest in freeze["source_sha256"].items():assert sha(W/p)==digest,p
    cache=W/".cache-xvla/checkpoint";assert str(cache.resolve())==freeze["restored_model_cache"]["resolved_path"]
    for p,digest in freeze["restored_model_cache"]["source_sha256"].items():assert sha(cache/p)==digest,p
    assert read(O/"eligibility-trace-audit.json")["selected_temporal_gate_passed"]
    assert read(O/"eligibility-geometry-audit.json")["selected_eligibility_geometry_passed"]
    dump(O/"unit-tests.json",self_check());limit()
    base.O=O;base.limit=limit;base.h.limit=limit;base.initial.limit=lambda*a:limit()
    base.h.O=O;base.chain.O=O;base.ev.model_entry=base.model_entry;base.old.model_entry=base.model_entry
    base.h.Policy=Policy;base.CONDITION="D5"
    entry=base.model_entry("Q1");assert sha(entry["checkpoint"])==base.Q1_SHA and sha(entry["probe"])==base.PROBE_SHA
    (O/"worker").mkdir();base.h.base.OUT=O/"worker";base.h.base.native.OUT=O
    w=base.h.base.Worker({"model":entry,"expected_initial_nontext":{}})
    rows=[]
    try:
        limit();dump(O/"qualification.json",{"passed":True,"fresh_worker_exact_probe":w.load_info,"epoch":time.time(),"no_training":True})
        PHASE="physical"
        cases=read(O/"cases.json");assert len(cases)==4 and sha(O/"cases.json")==freeze["cases_sha256"]
        for s in cases:
            for CONDITION in ["D5","C"]:
                limit();assert shutil.disk_usage("/var/tmp").free>400_000_000
                out=O/"episodes"/CONDITION/s["id"];assert not out.exists(),"No outcome retries"
                env=None;LAST=None
                try:
                    env,prev=base.ev.initialise(s,O/"prefixes"/CONDITION/s["id"])
                    w.protocol["expected_initial_nontext"]=s.get("expected_initial_nontext",{})
                    row,_,_=base.ev.segment(env,w,s,s["id"],out,False,prev,"Q1")
                    close();row=finish(row,s,out);rows.append(row);dump(O/"results.json",rows)
                    print(json.dumps({k:row[k]for k in ["id","condition","requested_success","preservation_qualified","forbidden_contact_samples","clearance"]}),flush=True)
                finally:
                    close()
                    if env is not None:env.close()
    finally:w.close()
    assert len(rows)==8
    dump(O/"physical-complete.json",{"episodes":8,"epoch":time.time(),"training":False,"outcome_retries":0,"quarantined":True})
if __name__=="__main__":
    try:main()
    except BaseException:
        dump(O/"STOPPED.json",{"epoch":time.time(),"error":traceback.format_exc(),"phase":PHASE,"outcome_retries":0,"technical_failure_not_policy_score":True})
        raise
