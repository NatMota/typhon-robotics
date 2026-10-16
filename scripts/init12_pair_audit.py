"""INIT12-A1 audits captured arrays only; never repairs/substitutes an input."""
import json,gzip
from pathlib import Path
import numpy as np
from init11_contract import history

def read(p): return json.loads(Path(p).read_text())
def lines(p):
    with gzip.open(p,"rt")as f:return[json.loads(x)for x in f]
def first_array_difference(a,b):
    return next((i for i in range(len(a))if not np.array_equal(a[i],b[i])),None)

def classify(a,b,ha,hb,images_a,images_b,robot_a,robot_b,inputs_a,inputs_b,tr_a,tr_b,through_query,through_actions):
    """Hard-fail unexplained inference/nonvisual differences; RGB-origin remains explicitly nonexact."""
    assert len(a)==len(b)==len(inputs_a)==len(inputs_b)
    for q in range(len(a)):
        if inputs_a[q]==inputs_b[q]:
            np.testing.assert_array_equal(a[q],b[q],err_msg="Same effective input/noise, different inference output")
    end=through_query+1
    input_diff=next((i for i in range(end)if inputs_a[i]!=inputs_b[i]),None)
    action_diff=first_array_difference(a[:end],b[:end])
    history_diff=first_array_difference(ha[:end],hb[:end])
    rgb_diffs={k:first_array_difference(images_a[k][:end],images_b[k][:end])for k in images_a}
    robot_diffs={k:first_array_difference(robot_a[k][:end],robot_b[k][:end])for k in robot_a}
    trace_diff=next((i for i,(x,y)in enumerate(zip(tr_a[:through_actions],tr_b[:through_actions]))if x!=y),None)
    record={"first_effective_input_difference_query":input_diff,"first_action_difference_query":action_diff,
            "first_history_difference_query":history_diff,"first_raw_rgb_differences":rgb_diffs,
            "first_measured_robot_differences":robot_diffs,"first_control_trace_difference_index":trace_diff,
            "through_query":through_query,"through_policy_actions":through_actions}
    if input_diff is None:
        assert action_diff is None and history_diff is None and all(v is None for v in robot_diffs.values()) and trace_diff is None, "Unexplained nonvisual divergence"
        return {**record,"passed":True,"exact_effective_pretrigger_equivalence":True,"status":"exact_pretrigger",
                "raw_rgb_exact":all(v is None for v in rgb_diffs.values())}
    q=input_diff
    assert q>0,"Initial effective input remains a hard gate"
    different={k for k in inputs_a[q]if inputs_a[q][k]!=inputs_b[q][k]}
    assert different=={"image_input"},("First effective difference is not RGB-only",different)
    assert any(v is not None and v<=q for v in rgb_diffs.values()),"No captured RGB source for tensor difference"
    assert any(not np.array_equal(images_a[k][q],images_b[k][q])for k in images_a),"No RGB difference at effective-input divergence"
    np.testing.assert_array_equal(a[:q],b[:q])
    np.testing.assert_array_equal(ha[:q+1],hb[:q+1])
    for k in robot_a:np.testing.assert_array_equal(robot_a[k][:q+1],robot_b[k][:q+1])
    assert tr_a[:q*30]==tr_b[:q*30],"Nonvisual execution difference precedes first effective RGB difference"
    return {**record,"passed":True,"exact_effective_pretrigger_equivalence":False,
            "status":"captured_rgb_first_origin_unresolved","raw_rgb_exact":False,
            "same_input_noise_output_check_all_queries":True,
            "does_not_exonerate_complete_wrapper_execution_context":True}

def within_run(root,rule):
    root=Path(root);z=np.load(root/"predictions.npz");raw=z["chunks"];hs=z["histories"]
    issued=np.load(root/"issued-predictions.npz")["chunks"]
    rows=lines(root/"controller-force-trace.jsonl.gz");acts=[r["policy_action"]for r in rows if r["policy_action"] is not None]
    assert len(acts)==1436 and len(rows)==1460
    np.testing.assert_array_equal([a["raw_model_target"]for a in acts],raw.reshape(-1,20)[:1436])
    np.testing.assert_array_equal([a["issued_target"]for a in acts],issued.reshape(-1,20)[:1436])
    assert np.count_nonzero(hs[:,9:])==0
    for q in range(1,len(hs)):np.testing.assert_array_equal(hs[q],np.asarray(history(issued[q-1,-1]),np.float32))
    active=np.array([a["offset_active"]for a in acts])
    delta=issued.reshape(-1,20)[:1436]-raw.reshape(-1,20)[:1436]
    np.testing.assert_array_equal(delta[~active],0)
    np.testing.assert_array_equal(delta[:,[j for j in range(20)if j!=2]],0)
    np.testing.assert_allclose(delta[active,2],-.005,rtol=0,atol=1e-15)
    assert int(active.sum())==rule["modified_actions_executed"]
    return {"passed":True,"unmodified_targets_exact":True,"history_all47_transitions_exact":True,
            "policy_actions":1436,"settle_actions":24,"modified_actions":int(active.sum())}

def pair(native,wrapped,rule):
    native,wrapped=Path(native),Path(wrapped)
    za,zb=[np.load(p/"predictions.npz")for p in [native,wrapped]]
    ia,ib=[np.load(p/"policy_io.npz")for p in [native,wrapped]]
    ha,hb=[[read(p/"input-audit"/f"input-{q:03d}.json")["input_hashes"]for q in range(48)]for p in [native,wrapped]]
    tr_a,tr_b=[lines(p/"controller_trace.jsonl.gz")for p in [native,wrapped]]
    q=rule["trigger"]["query_zero_based"]if rule["trigger"]else 47
    n=rule["trigger"]["policy_action_zero_based"]if rule["trigger"]else 1436
    return classify(za["chunks"],zb["chunks"],za["histories"],zb["histories"],
        {k:ia[k]for k in ["primary","secondary"]},{k:ib[k]for k in ["primary","secondary"]},
        {k:ia[k]for k in ["eef_position_m","eef_quat_xyzw"]},{k:ib[k]for k in ["eef_position_m","eef_quat_xyzw"]},
        ha,hb,tr_a,tr_b,q,n)

def self_check():
    a=np.zeros((3,30,20));h=np.zeros((3,20));im={"secondary":np.zeros((3,2,2,3),np.uint8)}
    rob={"eef":np.zeros((3,3))};inputs=[{"image_input":"x","proprio":"y"}for _ in range(3)];tr=[{"step":i}for i in range(90)]
    def check(b=a,hb=h,ib=im,rb=rob,hashb=inputs,tb=tr):
        return classify(a,b,h,hb,im,ib,rob,rb,inputs,hashb,tr,tb,2,90)
    assert check()["exact_effective_pretrigger_equivalence"]
    b=a.copy();b[1:,0,0]=1.;images={"secondary":im["secondary"].copy()};images["secondary"][1:,0,0,0]=1
    hashes=[dict(x)for x in inputs];hashes[1]["image_input"]="new1";hashes[2]["image_input"]="new2"
    trace=[dict(x)for x in tr];trace[30]["step"]=100
    assert not check(b=b,ib=images,hashb=hashes,tb=trace)["exact_effective_pretrigger_equivalence"]
    try:check(b=b)
    except AssertionError:pass
    else:raise AssertionError("Same input/different output accepted")
    hashes2=[dict(x)for x in hashes];hashes2[1]["proprio"]="changed"
    try:check(b=b,ib=images,hashb=hashes2,tb=trace)
    except AssertionError:pass
    else:raise AssertionError("Nonvisual-first input change accepted")
    early=[dict(x)for x in trace];early[10]["step"]=999
    try:check(b=b,ib=images,hashb=hashes,tb=early)
    except AssertionError:pass
    else:raise AssertionError("Unexplained earlier physical change accepted")
    return {"passed":True,"exact_case":True,"rgb_first_classified_nonexact":True,
            "same_input_output_error_rejected":True,"nonvisual_first_rejected":True,
            "earlier_state_difference_rejected":True}
if __name__=="__main__":print(json.dumps(self_check()))
