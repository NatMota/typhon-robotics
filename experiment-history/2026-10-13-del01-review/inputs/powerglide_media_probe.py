from pathlib import Path
import json,subprocess,hashlib,concurrent.futures,datetime
spec=json.loads(Path('/var/tmp/del01-selected-records-20261013.json').read_text())
jobs=[(r['slot'],'original',str(Path(r['original_pg_directory'])/r['video_filename']))for r in spec]+[(r['slot'],'derived_comparison',r['comparison_pg_path'])for r in spec if 'comparison_pg_path'in r]
def probe(t):
 slot,kind,path=t;p=Path(path);out={'slot':slot,'kind':kind,'path':path,'exists':p.is_file()}
 if not p.is_file():return out
 out.update({'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 cmd=['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_frames,duration:format=duration','-of','json',path]
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=90);out['returncode']=r.returncode;out['stderr']=r.stderr
 if r.returncode==0:out['probe']=json.loads(r.stdout)
 return out
with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:results=list(pool.map(probe,jobs))
p=Path('/var/tmp/del01-media-probes-20261013.json');p.write_text(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':results},indent=2)+'\n')
print(json.dumps(results))
