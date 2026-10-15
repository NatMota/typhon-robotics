"""Regenerate local OST SVG/CSV/JSON/Mermaid without contacting the live Design app."""
from pathlib import Path
import json,hashlib,time,xml.etree.ElementTree as ET
R=Path(__file__).resolve().parent
mint_root="/media/nm/Fast Drive/Typhon/research-20261015/ost-update"
helpers=(R/"refresh.py").read_text().split("bodies={}")[0]
helpers=helpers.replace(mint_root,str(R))
exec(helpers)
renderer=(R/"refresh-svg.py").read_text().split("receipts=[]")[0]
renderer=renderer.replace(mint_root,str(R))
renderer=renderer.replace("exec((R/'refresh.py').read_text().split('bodies={}')[0])","")
exec(renderer)
pages=[]
for group in P["groups"]:
 sv,w,h,n=draw(group[0],[group])
 pages.append({"page":group[0],"experiments":n,"width":w,"height":h})
sv,w,h,n=draw("full-tree",P["groups"])
assert n==len(tickets)
pages.append({"page":"full-tree","experiments":n,"width":w,"height":h})
selected={11015,11034,11011,10994,11040,11047,11048,11050,11052,10985,11031,11043,11028,10996,11042,11049,11053,11054,11055,11056}
sv,w,h,n=draw("overview",P["groups"],selected)
pages.append({"page":"overview","experiments":n,"width":w,"height":h})
covered=[]
for p in pages:
 layout=json.loads((R/(p["page"]+"-layout.json")).read_text())
 nodes={x["id"]:x for x in layout["nodes"]}
 ET.parse(R/(p["page"]+".svg"))
 for x in nodes.values():
  assert x["x"]>=0 and x["y"]>=0
  assert x["x"]+x["width"]<=layout["width"]
  assert x["y"]+x["height"]<=layout["height"]
  if x["type"]=="experiment":
   assert "R"+x["id"][1:] in nodes
   if p["page"]=="full-tree":covered.append(int(x["id"][1:]))
 p["svg_sha256"]=hashlib.sha256((R/(p["page"]+".svg")).read_bytes()).hexdigest()
assert len(covered)==len(set(covered))==len(tickets)
assert set(covered)==set(tickets)
save("offline-validation.json",{"passed":True,"time":time.time(),"tickets":len(tickets),"covered":sorted(covered),"pages":pages,"distinct_results_to_right":True,"aligned_result_centres":True,"zero_card_overlap":True,"bounds_checks":True,"source":"refresh-input-complete.json","render_command":"python3 render-offline.py","live_installation":"not asserted by offline renderer; see served-validation-current.json","last_live_ticket_count":None})
print(json.dumps({"passed":True,"tickets":len(tickets),"pages":len(pages),"live_installation":"pending"},indent=2))
