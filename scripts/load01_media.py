"""Offline matched videos and endpoint sheets from recorded rollouts; no simulator run."""
from pathlib import Path
import json,sys,os
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/"scripts")];os.chdir(R)
from load01_prepare import OUT,read,dump,sha
methods=["P","F","FL"];labels={"P":"Existing varied-goal model","F":"More feeding only","FL":"Feeding + loading"}
font=ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",18) if Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf").exists() else ImageFont.load_default()
qa=OUT/"qa";qa.mkdir(exist_ok=True)
p=read(OUT/"protocol.json");rows={}
for m in methods:
 for phase in ["source","remainder"]:
  for r in read(OUT/"evaluation"/m/phase/"summary.json"):rows[m,r["id"]]=r
for panel in ["loading_source","loading_development","feeding_retention"]:
 cases=[s for s in p["scenarios"]if s["group"]==panel]
 sheet=Image.new("RGB",(3*384,len(cases)*426),(15,20,27));draw=ImageDraw.Draw(sheet)
 for i,s in enumerate(cases):
  phase="source"if panel=="loading_source"else"remainder"
  for j,m in enumerate(methods):
   r=rows[m,s["scenario_id"]];d=OUT/"evaluation"/m/phase/s["scenario_id"]
   im=Image.open(d/"final.png").crop((0,0,512,512)).resize((384,384))
   sheet.paste(im,(j*384,i*426+42))
   draw.text((j*384+6,i*426+3),labels[m]+" / "+s["source_demo_id"],font=font,fill="white")
   draw.text((j*384+6,i*426+23),("PASS"if r["success"]else"FAIL")+"  %.1f mm / %.2f deg"%(r["position_error_mm"],r["yaw_error_deg"]),font=font,fill=(110,240,160)if r["success"]else(255,165,125))
 sheet.save(qa/(panel+"-endpoints.jpg"),quality=90)
selected=[("loading_development","load01-development-demo_0012"),("loading_source","load01-source-demo_0001"),("feeding_retention","load01-retention-demo_0003")]
manifest=[]
for panel,id in selected:
 phase="source"if panel=="loading_source"else"remainder"
 sources=[OUT/"evaluation"/m/phase/id/"rollout.mp4"for m in methods]
 readers=[imageio.get_reader(s)for s in sources];fps=[x.get_meta_data()["fps"]for x in readers];assert len(set(fps))==1
 name=id+"-matched-overhead.mp4";writer=imageio.get_writer(qa/name,fps=fps[0],macro_block_size=16,codec="libx264",quality=8)
 count=0
 try:
  for frames in zip(*readers):
   canvas=Image.new("RGB",(1536,560),(15,20,27));draw=ImageDraw.Draw(canvas)
   for j,(m,frame)in enumerate(zip(methods,frames)):
    canvas.paste(Image.fromarray(frame[:,:512,:]),(j*512,48))
    r=rows[m,id];draw.text((j*512+8,5),labels[m],font=font,fill="white")
    draw.text((j*512+8,27),("PASS"if r["success"]else"FAIL")+" | final %.1f mm, %.2f deg"%(r["position_error_mm"],r["yaw_error_deg"]),font=font,fill=(110,240,160)if r["success"]else(255,165,125))
   writer.append_data(np.asarray(canvas));count+=1
 finally:
  writer.close()
  for reader in readers:reader.close()
 # Fully decode the derivative, and compare frame counts with all originals.
 counts=[]
 for path in sources+[qa/name]:
  rd=imageio.get_reader(path);counts.append(sum(1 for _ in rd));rd.close()
 assert counts==[count]*4,counts
 manifest.append({"id":id,"panel":panel,"file":name,"frames":count,"fps":fps[0],"duration_s":count/fps[0],"sha256":sha(qa/name),"bytes":(qa/name).stat().st_size,
 "source_files":[{"path":str(s),"sha256":sha(s)}for s in sources],"selection":"post-hoc representative: first loading development; shared source failure; single newly failed feeding-retention case","visualization_only":True})
dump(qa/"media-manifest.json",manifest);print(json.dumps(manifest,indent=2))
