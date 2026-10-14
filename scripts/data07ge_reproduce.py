"""Fresh bounded reproduction; requires the exact qualified DATA07-R interpreter alias."""
import pathlib,sys,argparse,time,json,traceback
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import data07ge_run as runner
p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=pathlib.Path);a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);runner.O=out
d=runner.R/"experiment-history/2026-10-14-data07ge";(out/"protocol.json").write_bytes((d/"protocol.json").read_bytes())
t=time.time();runner.dump(out/"clock.json",{"start":t,"commands_deadline":t+300,"qualification_deadline":t+900,"execution_deadline":t+1500,"final_deadline":t+1800,"reproduction":True})
try:runner.main()
except BaseException:runner.dump(out/"STOPPED.json",{"error":traceback.format_exc(),"time":time.time()});raise
