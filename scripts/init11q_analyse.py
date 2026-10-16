"""Summarize the six predeclared INIT11-Q calls, no model loading."""
import csv,json,difflib,hashlib,time
from pathlib import Path
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def main():
    o=Path("/var/tmp/typhon-init11q-20261016")
    assert (o/"driver-complete.json").exists()
    rows=[];calls={};runtime={};models={}
    for tag in "ABC":
        complete=read(o/tag/"complete.json");assert complete["calls"]==2 and complete["model_unchanged"]
        calls[tag]=read(o/tag/"calls.json");runtime[tag]=read(o/tag/"runtime-before.json")
        models[tag]=read(o/tag/"model-before.json")
        for row in calls[tag]:
            assert row["exception"]is None
            rows.append({"process":tag,**{k:row[k]for k in ["call","exact","max_abs_error","changed_values","xyz_max_abs_mm","xyz_max_euclidean_mm","rotation_max_degrees","gripper_max_error","gripper_threshold_crossings","inactive_arm_max_error"]},
                         "actual_sha256":row["actual_metadata"]["sha256"],
                         "noise_sha256":row["noise_metadata"][0]["sha256"]})
    for r in rows:assert r["noise_sha256"]==rows[0]["noise_sha256"]
    input_hashes=[{k:v["sha256"]for k,v in r["input_metadata"].items()}for rs in calls.values()for r in rs]
    assert all(x==input_hashes[0]for x in input_hashes)
    with (o/"comparisons.csv").open("w")as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    comparisons={}
    for pair in ["AB","AC"]:
        a,b=pair
        source={k:{"a":runtime[a]["cache_files"].get(k),"b":runtime[b]["cache_files"].get(k)}
                for k in set(runtime[a]["cache_files"])|set(runtime[b]["cache_files"])
                if runtime[a]["cache_files"].get(k)!=runtime[b]["cache_files"].get(k)}
        model_diffs={kind:[k for k in set(models[a][kind])|set(models[b][kind])
                          if models[a][kind].get(k)!=models[b][kind].get(k)]for kind in ["parameters","buffers"]}
        ca=read(o/a/"effective-model-config.json");cb=read(o/b/"effective-model-config.json")
        (o/("effective-config-"+pair+".diff")).write_text("".join(difflib.unified_diff(
            json.dumps(ca,sort_keys=True,indent=2).splitlines(True),
            json.dumps(cb,sort_keys=True,indent=2).splitlines(True),fromfile=a,tofile=b)))
        comparisons[pair]={"different_cache_files":source,"different_model_tensors":model_diffs,
             "attention_classes_equal":runtime[a]["attention_module_classes"]==runtime[b]["attention_module_classes"],
             "call_flags_equal":calls[a][0]["flags_inside"]==calls[b][0]["flags_inside"],
             "first_different_refinement_output":next((i for i,(x,y)in enumerate(zip(calls[a][0]["refinement_outputs"],calls[b][0]["refinement_outputs"]))if x["sha256"]!=y["sha256"]),None)}
    stable={k:len({r["actual_metadata"]["sha256"]for r in rs})==1 for k,rs in calls.items()}
    all_exact={k:all(r["exact"]for r in rs)for k,rs in calls.items()}
    verdict="current_context_stably_wrong_original_context_exact"if stable==dict.fromkeys("ABC",True)and all_exact=={"A":False,"B":False,"C":True}else"requires_review"
    summary={"epoch":time.time(),"calls":6,"physical_rollouts":0,"same_input_hashes":True,"same_initial_noise":True,
             "within_process_stable":stable,"exact_by_process":all_exact,"verdict":verdict,
             "pairwise_comparison":comparisons,"rows":rows,"new_physics_permitted_by_this_diagnostic":False,
             "next_gate":"Restore only the original model-cache context in a new worktree; require exact original probe in two fresh processes before archived-query and physics qualification."}
    dump(o/"validation_summary.json",summary)
    current=Path(runtime["A"]["cache_realpath"]);original=Path(runtime["C"]["cache_realpath"])
    for rel in ["modeling_xvla.py","config.json"]:
        (o/(rel+".diff")).write_text("".join(difflib.unified_diff((current/rel).read_text().splitlines(True),(original/rel).read_text().splitlines(True),fromfile="failed-restoration/"+rel,tofile="original-Q1/"+rel)))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
