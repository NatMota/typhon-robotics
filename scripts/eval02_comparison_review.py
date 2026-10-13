"""Read-only reconciliation of frozen evaluations; no simulation or new significance tests."""
import argparse, ast, collections, csv, datetime, hashlib, itertools, json, random, statistics
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--storage-root",type=Path,default=Path("/media/nm/Fast Drive/Typhon"))
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args(); R=args.storage_root; O=args.output
    O.mkdir(parents=True,exist_ok=True)
    S=R/"research-20261012/ost-sweep-20261012"
    F=R/"evaluations/film-feeding-775-20261011"
    E=R/"research-20261012/eval02-comparison-review"
    W=Path(__file__).resolve().parents[1]
    manifest={}
    def raw(p):
        b=p.read_bytes(); manifest[str(p)]={"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b)}; return b
    def read(p): return json.loads(raw(p))
    def csvread(p): return list(csv.DictReader(raw(p).decode().splitlines()))
    def truth(x): return x is True or x=="True"
    def save(name,obj): (O/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
    def pairs(rows,methods,endpoint="success"):
        out=[]
        by={m:{r["id"]:r for r in rows if r["method"]==m}for m in methods}
        for a,b in itertools.combinations(methods,2):
            assert set(by[a])==set(by[b])
            gains=[];losses=[];ties=collections.Counter()
            for key,x in by[a].items():
                av=truth(x[endpoint]);bv=truth(by[b][key][endpoint])
                if bv and not av:gains.append(key)
                elif av and not bv:losses.append(key)
                else:ties["both_success"if av else"both_fail"]+=1
            out.append({"reference":a,"candidate":b,"endpoint":endpoint,"n":len(by[a]),
                        "gains":len(gains),"losses":len(losses),"gain_ids":gains,"loss_ids":losses,
                        "ties":dict(ties),"inference":"descriptive discordance counts, no new test"})
        return out

    suite=read(F/"suite.json")
    film=csvread(F/"analysis/all-775-episode-metrics.csv")
    post=read(F/"analysis/subgoal-audit-v2-region/summary.json")
    postrows=csvread(F/"analysis/subgoal-audit-v2-region/per-scene.csv")
    bypost={(r["model"],r["scenario_id"]):r for r in postrows}
    assert len(film)==775 and len(bypost)==775
    matched=collections.defaultdict(set)
    totals=[]
    for model in suite["models"]:
        rows=[r for r in film if r["model"]==model];assert len(rows)==155
        for r in rows:
            m=read(F/"benchmark-native"/model/r["scenario_id"]/"metrics.json")
            assert bool(m["success"])==truth(r["success"])
            matched[r["scenario_id"]].add((r["initial_state_sha256"],r["config_sha256"],r["seed"]))
        completed=[r for r in rows if truth(bypost[model,r["scenario_id"]]["initial_full_pose_already_satisfied"])]
        unfinished=[r for r in rows if r not in completed]
        assert len(completed)==18 and len(unfinished)==137
        assert sum(truth(r["success"])for r in unfinished)==0
        summary=next(r for r in post["summary"]if r["model"]==model)
        totals.append({"model":model,"n":155,"strict":sum(truth(r["success"])for r in rows),
            "initially_complete_n":18,"initially_incomplete_n":137,"new_strict_completions":0,
            "pickup_legacy_70mm":sum(truth(r["pickup"])for r in unfinished),
            "new_position_eligible_n":119,"new_region_place_eligible_n":116,"new_region_goal_eligible_n":137,
            "posthoc":summary})
    assert len(matched)==155 and all(len(v)==1 for v in matched.values())
    film_pairs=pairs([dict(r,method=r["model"],id=r["scenario_id"])for r in film],suite["models"])
    original_tests=csvread(F/"analysis/primary-paired-comparisons.csv")
    receipt=read(E/"inputs/775-reproduction-receipt.json")
    assert len(receipt["files"])==5 and all(x["exact_bytes_equal"] and x["sha256"]==x["original_sha256"] for x in receipt["files"])
    human=read(W/"experiment-history/2026-10-13-data07-review/human_observation_ledger.json")
    human_sources={r["id"]:{k:v["recording_id"]for k,v in r.get("original_sources",{}).items()}for r in human}
    film_scenarios=[{k:s.get(k)for k in ["scenario_id","cohort","source_demo_id","parent_family_id","training_exposure","seed"]}for s in suite["scenarios"]]
    for s in film_scenarios:s["original_recordings"]=human_sources.get(s["source_demo_id"])
    film_ledger={"panel":"original_775","models_sha256":suite["checkpoint_sha256"],"results":totals,
        "eligible_definitions":post,"paired_strict_discordances":film_pairs,"original_primary_tests":original_tests,
        "exact_original_analysis_reproduction":receipt,
        "cohorts":dict(collections.Counter(s["cohort"]for s in film_scenarios)),
        "scenarios":film_scenarios,
        "policy_contract":{k:suite[k]for k in ["action_request","annotation_conditioned_models","controls","statistical_limit","physical_scale_limit"]},
        "inference":"Original scenario-weighted primary contrast; 5000 paired family-cluster bootstrap, 50000 cluster sign flips, seed61117, Holm over10 strict contrasts. Repeated near-scenes grouped by parent; 125 clusters assumed. No new posthoc significance test.",
        "exposure":"Includes training-derived, previously inspected and legacy scenes; even novel synthetic scenes use familiar geometry/task. Human take families can share original recordings. Not775 independent demonstrations.",
        "claim":"Strict gains are initially-complete preservation. V3 has4 new correctly oriented released/stable in-square placements on the labelled posthoc region endpoint; original centre criterion not changed."}

    broad=read(E/"inputs/broad-saved-execution-records.json")
    broad_panels=[]
    for panel in ["development","source"]:
        rr=broad[panel]; groups=[]; initial={}
        for x in rr:
            ep=x["episode"];m=x["metrics"];assert ep["evaluation"]["success"]==m["success"]
            assert ep["evaluation"]["criteria"]==m["final_criteria"]
            t=ep["initial_truth"];v=t["object_position_m"]+t["object_quat_wxyz"]+t["eef_position_m"]
            if m["id"]in initial:assert max(abs(a-b)for a,b in zip(initial[m["id"]],v))<1e-9
            initial[m["id"]]=v
        for method in sorted({x["method"]for x in rr}):
            mm=[x["metrics"]for x in rr if x["method"]==method]
            groups.append({"method":method,"n":len(mm),"strict":sum(x["success"]for x in mm),
                "grasp_10mm_bilateral_3steps":sum(x["supported_lift_10mm_3steps"]for x in mm),
                "operational_place":sum(x["operational_placement"]for x in mm),
                "sources":sorted({x["source_demo_id"]for x in mm})})
        broad_panels.append({"panel":panel,"groups":groups,
            "paired_discordances":pairs([x["metrics"]for x in rr],sorted({x["method"]for x in rr})),
            "raw_episode_sha256":{x["episode_path"]:x["episode_sha256"]for x in rr},
            "protocol":broad["evaluation-protocol.json"if panel=="development"else"source-competence/evaluation-protocol.json"],
            "uncertainty":"No new inferential test; development3 previously inspected take families x2seeds; source6 exact training starts x1seed, no source comparator."})

    # Reuse the original functions only. Do not execute its top-level writes.
    script=raw(W/"scripts/exp20261012_sweep_report.py").decode()
    assert hashlib.sha256(script.encode()).hexdigest()=="a92884ea615764bf49260b265f604673db348272b44d31c6b89a8bf3644925cf"
    nodes=[n for n in ast.parse(script).body if isinstance(n,ast.FunctionDef)and n.name in ["quantile","bootstrap"]]
    ns={"random":random,"statistics":statistics};exec(compile(ast.Module(body=nodes,type_ignores=[]),"original_bootstrap","exec"),ns)
    boot=ns["bootstrap"]
    rows=read(S/"report/evaluation-rows.json"); published=read(S/"report/evaluation-groups.json")
    assert len(rows)==296
    plans={p:read(S/r)for p,r in [
        ("goal","goal-evaluation/evaluation-protocol.json"),
        ("recovery","recovery-evaluation/evaluation-protocol.json"),
        ("baselines","confirmatory-v2/protocol.json")]}
    cases={p:{s["scenario_id"]:s for s in v["scenarios"]}for p,v in plans.items()}
    initial={}; identities=[]; starts=collections.defaultdict(list)
    for r in rows:
        spec=cases[r["panel"]][r["id"]]
        assert r["seed"]==spec["seed"]and r["source_demo_id"]==spec["source_demo_id"]
        rel=Path(r["evidence"]).relative_to("/var/tmp/typhon-ost-sweep-20261012")
        ep=read(S/rel/"episode.json")
        assert ep["evaluation"]["success"]==r["success"]
        assert ep["evaluation"]["criteria"]==r["final_criteria"]
        if "evaluation_criteria" in r: assert r["evaluation_criteria"]==r["final_criteria"]
        assert ep["seed"]==r["seed"]
        assert not ep["artifact_errors"]and not ep["evaluation_errors"]
        identity=ep["provenance"]["model"]; plan=plans[r["panel"]]
        if r["panel"]=="baselines":
            expected=plan["vanilla_base_sha256"]if r["method"]=="vanilla-Pt-native"else plan["methods"]["broad-pt-12000"]["expected_sha256"]
        else:expected=plan["methods"][r["method"]]["expected_sha256"]
        assert identity["checkpoint_sha256"]==expected
        t=ep["initial_truth"];v=t["object_position_m"]+t["object_quat_wxyz"]+t["eef_position_m"]
        key=(r["panel"],r["id"])
        if key in initial:assert max(abs(a-b)for a,b in zip(initial[key],v))<1e-9
        initial[key]=v
        starts[r["panel"]].append(datetime.datetime.fromisoformat(ep["started_at_utc"]).timestamp())
        identities.append({"panel":r["panel"],"id":r["id"],"method":r["method"],"checkpoint_sha256":expected,"seed":r["seed"],"source_demo_id":r["source_demo_id"],"group":r["group"],"original_recordings":human_sources.get(r["source_demo_id"])})
    reproduced=[]
    for g in published:
        rr=[r for r in rows if (r["panel"],r["method"],r["group"])==(g["panel"],g["method"],g["group"])]
        fam=collections.defaultdict(list)
        for r in rr:fam[r["source_demo_id"]].append(int(r["success"]))
        got={"n":len(rr),"strict":sum(r["success"]for r in rr),
             "supported_grasp_10mm":sum(r["supported_lift_10mm_3steps"]for r in rr),
             "operational_place":sum(r["operational_placement"]for r in rr),
             "runtime_errors":sum(bool(r["errors"])for r in rr),
             "strict_family_bootstrap":boot([statistics.mean(v)for v in fam.values()])}
        assert all(g[k]==v for k,v in got.items()),(g,got)
        reproduced.append(dict(g,pooled_success_fraction=g["strict"]/g["n"]))
    paired_boot={}
    for group in sorted({r["group"]for r in rows if r["panel"]=="goal"}):
        by={m:{r["id"]:r for r in rows if r["panel"]=="goal"and r["group"]==group and r["method"]==m}for m in ["fixed","varied"]}
        fam=collections.defaultdict(list)
        for key,r in by["varied"].items():fam[r["source_demo_id"]].append(int(r["success"])-int(by["fixed"][key]["success"]))
        paired_boot[group]=boot([statistics.mean(v)for v in fam.values()])
    assert paired_boot==read(S/"report/paired-varied-minus-fixed.json")
    paired=[]
    for panel,group in sorted({(r["panel"],r["group"])for r in rows}):
        rr=[r for r in rows if r["panel"]==panel and r["group"]==group]
        paired.append({"panel":panel,"group":group,"contrasts":pairs(rr,sorted({r["method"]for r in rr}))})
    timing={}
    for p,v in plans.items():
        frozen=v.get("frozen_before_calls",v.get("created_before_evaluation",v.get("created_before_new_model_evaluation")))
        assert min(starts[p])>frozen
        timing[p]={"protocol_frozen_unix":frozen,"first_saved_episode_unix":min(starts[p]),"precedes_calls":True,
                   "claim":"Procedural freeze, not proof of pristine holdout after earlier research."}
    training=[]
    for model,rel in [("fixed","goal-grounding/train-fixed"),("varied","goal-grounding/train-varied"),("recovery","recovery-learning/train-recovery")]:
        p=S/rel;config=read(p/"config.json");audit=read(p/"data-audit.json");complete=read(p/"complete.json")
        logs=[json.loads(x)for x in raw(p/"losses.jsonl").decode().splitlines()]
        counts=collections.Counter(s.split(":")[0]for x in logs for s in x["sample_ids"])
        assert len(logs)==3000 and sum(counts.values())==12000
        assert set(counts)==set(audit["episodes"])
        training.append({"model":model,"parent_sha256":config["base_sha256"],"checkpoint_sha256":complete["checkpoint_sha256"],
                         "updates":3000,"batch":config["batch"],"seed":config["seed"],"windows":audit["windows"],
                         "actual_trajectory_draws":dict(counts),"family_exposures":audit["source_counts"],"config":config,
                         "warning":"Shared config training_sources is not actual per-variant membership; draw log and data audit reconciled."})
    registry=read(S/"audits/model-registry.json")
    historical_data=read(S/"audits/training-data-audit.json")
    broad_config=read(R/"research-20261011/broad-pt/artifacts/config.json")
    broad_exposure=read(R/"research-20261011/broad-pt/artifacts/actual-training-exposure.json")
    relevant_sha=set(suite["checkpoint_sha256"].values())|{"03a55fa46088368da0b8d92091ec3a4a5a539ae2772bf2cfd4c592b92b745a85","970bb00faaa19c80715df5a3b37c827dda2300efedcb6d49f7d26069d27de4a1","59eb9e502c057d639e278e7f84d06c8fa8e3e513e52522dae47089a97b5c3736"}
    lineage={"verified_saved_headers":[m for m in registry["models"]if m["sha256"]in relevant_sha],
             "historical_variant_configs_and_exposure":{k:v for k,v in historical_data.items()if k in ["v1-m6","v2-m6","v3-m6","v4-m6"]},
             "broad_config":broad_config,"broad_exposure":broad_exposure,
             "limits":"V1/V2 M6 identities come from frozen suite; their configs/exposure from saved audit, not a new checkpoint-header reread. Missing narrow initialization is not guessed. Original V4 recovery source arrays remain unresolved by DATA07."}
    recording_dependence={name:{d:human_sources.get(d)for d in demos}for name,demos in [
        ("P_train",["demo_0001","demo_0002","demo_0003","demo_0005","demo_0012","demo_0013"]),
        ("development",["demo_0006","demo_0014","demo_0016"])]}
    ledger={"schema":"eval02-comparison-ledger-v1","review_ticket":11036,"parent_ticket":10996,
            "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scope":"Existing evidence only; original endpoint, grouping, resampling and multiplicity unchanged.",
            "original775":film_ledger,"model_lineage_and_data":lineage,"recording_dependence":recording_dependence,"broad_development_and_source":broad_panels,
            "later_panels":{"groups":reproduced,"paired_descriptive_gains_losses":paired,"original_paired_varied_minus_fixed":paired_boot,
                "timing_checks":timing,"episode_identities":identities,"protocols":{k:{kk:vv for kk,vv in v.items()if kk!="scenarios"}for k,v in plans.items()},
                "uncertainty":"Original family-balanced bootstrap reproduced exactly. Its estimand can differ from pooled success; null-source groups collapse to one family and have no interval. All-zero empirical bootstrap interval is not a zero population risk bound. One fitted seed; repeated policy seeds/layouts are not independent human recordings."},
            "later_training":training,
            "gaps":[
                "Original EVAL02:136/204 executed; qualified intervention68 not run, not zero successes.",
                "No narrow/broad3k source-competence comparator.",
                "Original narrow-M6 header does not declare initialization lineage; verified identity is not proof of matched initialization.",
                "Original V4 three declared recovery arrays remain missing in audited backups; behavior evaluable, data effect not fully reproducible.",
                "V4 versus V3 adds500 updates and recovery data; no matched additional-update-only control.",
                "No newly trained seed replication or recording-level independence inference.",
                "VIDEO09-11 observer-assisted control is excluded from learned-policy superiority claims.",
                "No direct frame-aligned human video to robot-action training established."]}
    save("comparison_ledger.json",ledger)
    save("input_manifest.json",manifest)
    save("validation_summary.json",{"passed":True,"frozen_raw_775_metrics_checked":775,"later_raw_episodes_checked":296,
        "broad_saved_raw_record_extracts_checked":24,"original_775_exact_reproduction_receipt":receipt,
        "later_original_group_estimates_exact":len(reproduced),"later_original_paired_intervals_exact":len(paired_boot),
        "matched_initial_states":True,"later_checkpoint_identities":296,"training_draws_recounted":36000,
        "new_physics_rollouts":0,"new_training_updates":0,"new_significance_tests":0,"parent_204_complete":False,
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({"passed":True,"film":totals,"later_groups":[{k:g[k]for k in ["panel","method","group","n","strict","supported_grasp_10mm","operational_place"]}for g in reproduced],"files":len(manifest)}))

if __name__=="__main__":main()
