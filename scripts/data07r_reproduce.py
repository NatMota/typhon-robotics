"""Re-run DATA07-R into a fresh directory without altering frozen outputs."""
import argparse,json,pathlib,time,sys,traceback
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import data07r_run as runner
p=argparse.ArgumentParser();p.add_argument("--output",type=pathlib.Path,required=True);args=p.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
source=runner.R/"experiment-history/2026-10-14-data07r"
for name in ["protocol.json","protocol-amendment-01.json"]:
 (out/name).write_bytes((source/name).read_bytes())
now=time.time()
runner.OUT=out
runner.dump(out/"clock.json",{"start":now,"execution_deadline":now+35*60,"final_deadline":now+45*60,"reproduction":True})
try:runner.main()
except BaseException:
 runner.dump(out/"STOPPED.json",{"time":time.time(),"error":traceback.format_exc()})
 raise
