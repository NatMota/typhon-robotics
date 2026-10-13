#!/usr/bin/env python3
"""Collect DATA07 existing artifact evidence on Powerglide; read-only to originals."""
import gzip, hashlib, json, pathlib
H=pathlib.Path.home()/"Projects"; out=pathlib.Path("/var/tmp/typhon-data07-existing-audit");out.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
p=H/"typhon-harness-20261010/results/personal-m7-data-prep-20261009/training-catalog-v2-complete/temporal/source_snapshot.json.gz"
d=json.loads(gzip.decompress(p.read_bytes()))
a={"source_path":str(p),"source_sha256":sha(p),"episodes":[],"merged_sources":d.get("merged_source_sha256")}
for e in d["episodes"]:
 row={}
 for k,v in e.items():
  if k=="config":continue
  if isinstance(v,(list,dict)) and len(str(v))>500:
   row[k]={"type":type(v).__name__,"length":len(v),"first":str(v[0])[:100] if isinstance(v,list) and v else list(v)[:15] if isinstance(v,dict) else None}
  else:row[k]=v
 row["source_provenance"]=e.get("source_provenance")
 candidates=[pathlib.Path(e["source_npz_uri"]),H/"typhon-harness-20261010/frozen_results/personal-m3-20261009/data"/e["path"]]
 q=next((x for x in candidates if x.is_file()),candidates[0])
 row["npz_probe"]={"path":str(q),"exists":q.is_file(),"hash_match":False}
 if q.is_file():row["npz_probe"].update(bytes=q.stat().st_size,sha256=sha(q));row["npz_probe"]["hash_match"]=row["npz_probe"]["sha256"]==e["sha256"]
 src=pathlib.Path(e.get("source_path","/nonexistent"))
 if not src.is_absolute():src=pathlib.Path(str(q).split("/results/")[0])/src
 row["resolved_source_folder"]=str(src);row["source_folder_exists"]=src.is_dir()
 if src.is_dir():
  row["source_folder_files"]=[x.name for x in src.iterdir()]
  for n in ["episode.json","summary.json"]:
   x=src/n
   if x.is_file():
    v=json.loads(x.read_text());row[n]={k:vv for k,vv in v.items() if k in ["success","evaluation","source_demo_id","status"]}
    row[n+"_sha256"]=sha(x);row[n+"_matches_original"]=row[n+"_sha256"]==e.get("source_episode_sha256")
  x=src/"robot_actions.jsonl"
  if x.is_file():row["source_actions_current_sha256"]=sha(x);row["source_actions_hash_match"]=row["source_actions_current_sha256"]==e.get("source_action_sha256")
 a["episodes"].append(row)
(out/"catalog-evidence.json").write_text(json.dumps(a,indent=2))
(out/"source_snapshot.json.gz").write_bytes(p.read_bytes())
coverage=H/"humanoid-m6-joint/results/personal-m7-training-20261009/coverage-v1"
for n in ["curated_exclusions.json","personal_examples.json","new_completed_protocol.json","source_assets.json","validation_summary.json"]:
 (out/n).write_bytes((coverage/n).read_bytes())
print(json.dumps({"npz_verified":sum(x["npz_probe"]["hash_match"] for x in a["episodes"]),"episode_summaries_verified":sum(x.get("episode.json_matches_original",False) for x in a["episodes"]),"robot_action_files_verified":sum(x.get("source_actions_hash_match",False) for x in a["episodes"]),"total":len(a["episodes"])}))
