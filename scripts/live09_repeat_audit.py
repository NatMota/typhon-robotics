#!/usr/bin/env python3
"""Compare the already-run R10 batch with LIVE09 without sending API requests."""
import argparse, datetime, hashlib, json, pathlib, statistics

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--earlier",type=pathlib.Path,default=pathlib.Path("/var/tmp/typhon-ost-sweep-20261012/luna"))
    parser.add_argument("--repeat",type=pathlib.Path,default=pathlib.Path("/var/tmp/typhon-live09-ghost-decisions-20261012"))
    parser.add_argument("--out",type=pathlib.Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    records=[]
    for name,original in [("primary","ghost-critic-v2"),("supplement","ghost-critic-supplement-v2")]:
        newer=args.repeat/name
        earlier=args.earlier/original
        paths=sorted((newer/"slates").glob("*/ */order*/luna/choice.json".replace(" ","")))
        assert len(paths)==(24 if name=="primary" else 8),(name,len(paths))
        for path in paths:
            rel=path.relative_to(newer)
            old=earlier/rel
            a,b=read(old),read(path)
            row={"panel":name,"slate":str(rel.parent.parent),
                 "earlier_choice":a["choice"],"repeat_choice":b["choice"],
                 "choice_equal":a["choice"]==b["choice"],
                 "choice_probabilities_equal":a["probabilities"]==b["probabilities"],
                 "earlier_at":a["at_utc"],"repeat_at":b["at_utc"],
                 "earlier_choice_wall_s":a["wall_seconds"],"repeat_choice_wall_s":b["wall_seconds"],
                 "earlier_result_sha256":sha(old),"repeat_result_sha256":sha(path)}
            for kind in ["choice","reason"]:
                q1=old.with_name(kind+"-request.json")
                q2=path.with_name(kind+"-request.json")
                p1,p2=read(q1),read(q2)
                row[kind+"_request_equal"]=p1==p2
                row[kind+"_request_canonical_sha256"]=canonical(p2)
                r1,r2=read(old.with_name(kind+".json")),read(path.with_name(kind+".json"))
                row[kind+"_semantic_response_equal"]=all(r1.get(k)==r2.get(k) for k in ["choice","status","confidence","probabilities","model"])
                assert r1["at_utc"] < r2["at_utc"]
            records.append(row)
    result={
        "created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope":"Post-run provenance and repeatability audit; no new API calls, scenes, simulator actions, training or independent sample-size increase.",
        "earlier_root":str(args.earlier),"repeat_root":str(args.repeat),
        "matched_slates":len(records),
        "identical_choice_requests":sum(r["choice_request_equal"] for r in records),
        "identical_reason_requests":sum(r["reason_request_equal"] for r in records),
        "identical_choices":sum(r["choice_equal"] for r in records),
        "identical_choice_semantics":sum(r["choice_semantic_response_equal"] for r in records),
        "identical_reason_semantics":sum(r["reason_semantic_response_equal"] for r in records),
        "earlier_choice_median_s":statistics.median(r["earlier_choice_wall_s"] for r in records),
        "repeat_choice_median_s":statistics.median(r["repeat_choice_wall_s"] for r in records),
        "records":records,
    }
    (args.out/"repeat_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))

if __name__=="__main__": main()
