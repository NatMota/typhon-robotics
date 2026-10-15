"""Reproduce INIT09 read-only analysis into a new directory; no simulator or model calls."""
import argparse,time,json,sys
from pathlib import Path
import init09_prepare as p
import init09_analyse as a
import init09_retry as r
ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
out=Path(args.output).resolve();original=p.O.resolve()
assert out!=original and original not in out.parents and not out.exists(),'Use a fresh directory outside the frozen original'
seal=p.read(original/'selection-freeze.json')
for path,h in seal['source_sha256'].items():assert p.sha(Path(path))==h,path
out.mkdir(parents=True)
start=time.time()
p.dump(out/'clock.json',{'t0':start,'inventory_deadline':start+480,'analysis_deadline':start+1500,'archive_deadline':start+2100,'mode':'separate read-only reproduction','original_clock':p.read(original/'clock.json')})
for module in [p,a,r]:module.O=out
p.main();a.main();r.main()
again=p.read(out/'selection-freeze.json')
assert again['source_sha256']==seal['source_sha256']
assert again['attempts']==seal['attempts']
p.dump(out/'reproduction-validation.json',{'passed':True,'mode':'read-only reproduction, not new research execution','source_manifest':str(original/'selection-freeze.json'),'original_unchanged':all(p.sha(Path(path))==h for path,h in seal['source_sha256'].items()),'new_model_calls':0,'new_physics':0,'new_training':0,'completed_epoch':time.time()})
