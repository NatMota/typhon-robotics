"""Recompute observer state from archived evidence without a simulator."""
import argparse,ast,gzip,hashlib,json
from pathlib import Path
def main(old,new,out):
 root=Path(__file__).resolve().parents[1]
 namespace={"MODES":("legacy","aperture","opening","both")}
 hashes={}
 for filename,class_name in [("obs05_feature_motion.py","Latch"),("obs06_aperture.py","Observer")]:
  p=root/"scripts"/filename;source=p.read_text();hashes[filename]=hashlib.sha256(p.read_bytes()).hexdigest()
  node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name==class_name)
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),"exec"),namespace)
 total=0;legacy=0
 for p in sorted(old.glob("*/observations.jsonl.gz")):
  observer=namespace["Observer"]()
  for line in gzip.open(p,"rt"):
   r=json.loads(line);pred,_=observer.update(r["evidence"],r["step"])
   assert pred["legacy"]==r["predictions"]["real"]["contour_legacy"]
   legacy+=1
 for p in sorted(new.glob("*/observations.jsonl.gz")):
  observer=namespace["Observer"]()
  for line in gzip.open(p,"rt"):
   r=json.loads(line);pred,rate=observer.update(r["evidence"],r["step"])
   assert pred==r["predictions"],(p,r["step"])
   assert rate==r["aperture_rate_m_s"]
   total+=1
 assert legacy==1219 and total==8784
 result={"legacy_per_sample_parity":legacy,"additional_panel_full_state_recomputations":total,"all_four_variants_match":True,"opening_rate_values_match":True,"source_hashes":hashes,"method":"AST-isolated actual Latch/Observer classes on archived observable evidence; no simulator, truth or model invocation."}
 out.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--old",type=Path,required=True);p.add_argument("--new",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();main(a.old,a.new,a.out)
