"""Saved-byte analysis only; no simulator or model imports."""
import sys,json,pickle,gzip,hashlib,csv
from pathlib import Path
import numpy as np
import cv2
R=Path(sys.argv[1])
def digest(a):return hashlib.sha256(str(a.dtype).encode()+json.dumps(list(a.shape)).encode()+a.tobytes()).hexdigest()
def dump(n,x):(R/n).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")
def comparison(a,b):
 d=b.astype(np.int16)-a.astype(np.int16);ys,xs=np.nonzero(np.any(d!=0,axis=2));coords=np.column_stack((xs,ys))
 return {"changed_pixels":len(ys),"changed_channels":int(np.count_nonzero(d)),"max_channel_difference":int(np.abs(d).max()),"xy_bbox":None if not len(xs) else [int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],"changed_xy":coords.tolist(),"pixel_24_417_changed":bool(np.any(d[417,24]))}
def main():
 with gzip.open(R/"captures.pkl.gz","rb") as f:bundles=pickle.load(f)
 expected=["N","N","P","P","P","P","N","N"]*3
 complete=len(bundles)==24 and [b["mode"] for b in bundles]==expected
 q=json.loads((R/"qualification.json").read_text())
 first=bundles[0];matrix=[];changes=[];tensor_changes=[];graphics_changes=[]
 native=Path("/var/tmp/typhon-live01-p-20261015")
 with np.load(native/"monitoring/demo_0002/policy_io.npz") as f:hist_n={c:f[c][5].copy() for c in ["primary","secondary"]}
 with np.load(native/"lineage/demo_0002/policy_io.npz") as f:hist_p={c:f[c][5].copy() for c in ["primary","secondary"]}
 array_dir=R/"arrays";array_dir.mkdir(exist_ok=True)
 for i,b in enumerate(bundles):
  assert b["read_only"] and not b["state_differences"],"Non-read-only capture"
  row={"bundle":i,"mode":b["mode"],"tick":190,"read_only":True,"secondary_pixel_24_417":b["rgb"]["secondary"][417,24].tolist(),"image_mask":b["prepared"]["arrays"]["image_mask"].tolist(),"crop":b["prepared"]["crop"]}
  if b["lineage"] is not None:
   assert all(x["known"] for x in b["lineage"].values())
   row["capture_ids"]={c:x["capture_id"] for c,x in b["lineage"].items()}
  else:row["capture_ids"]=None
  for c in ["primary","secondary"]:
   a=b["rgb"][c];row[c+"_sha256"]=digest(a)
   for label,ref in [("first_bundle",first["rgb"][c]),("historical_monitoring_query5",hist_n[c]),("historical_lineage_query5",hist_p[c])]:
    d=comparison(ref,a);changes.append({"bundle":i,"mode":b["mode"],"camera":c,"reference":label,**d})
   g=next(x["graphics"] for x in b["graphics"] if x["camera"]==c);g0=next(x["graphics"] for x in first["graphics"] if x["camera"]==c)
   gs=json.dumps(g,sort_keys=True);gs0=json.dumps(g0,sort_keys=True);row[c+"_graphics_sha256"]=hashlib.sha256(gs.encode()).hexdigest()
   graphics_changes.append({"bundle":i,"camera":c,"active_graphics_exact_first":gs==gs0})
  for k,a in b["prepared"]["arrays"].items():
   eq=np.array_equal(first["prepared"]["arrays"][k],a)
   tensor_changes.append({"bundle":i,"mode":b["mode"],"key":k,"exact_first":bool(eq),"sha256":digest(a)})
  np.savez_compressed(array_dir/("%02d-%s.npz"%(i,b["mode"])),**b["rgb"],**{"prepared_"+k:v for k,v in b["prepared"]["arrays"].items()})
  matrix.append(row)
 adjacent=[]
 for i in range(1,len(bundles)):
  for c in ["primary","secondary"]:
   a,b=bundles[i-1],bundles[i];adjacent.append({"left_bundle":i-1,"right_bundle":i,"left_mode":a["mode"],"right_mode":b["mode"],"camera":c,**comparison(a["rgb"][c],b["rgb"][c])})
 dump("capture-matrix.json",matrix);dump("pixel-comparisons.json",changes);dump("adjacent-comparisons.json",adjacent);dump("tensor-comparisons.json",tensor_changes);dump("graphics-comparisons.json",graphics_changes)
 dump("graphics-settings.json",[{"bundle":b["bundle"],"mode":b["mode"],"graphics":b["graphics"]} for b in bundles])
 with (R/"capture-matrix.csv").open("w",newline="") as f:
  keys=["bundle","mode","tick","read_only","secondary_pixel_24_417","primary_sha256","secondary_sha256","primary_graphics_sha256","secondary_graphics_sha256","image_mask","crop","capture_ids"]
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(matrix)
 raw_exact=all(x["changed_pixels"]==0 for x in changes if x["reference"]=="first_bundle")
 prepared_exact=all(x["exact_first"] for x in tensor_changes);graphics_exact=all(x["active_graphics_exact_first"] for x in graphics_changes)
 nn_diff=any(x["changed_pixels"] for x in adjacent if x["left_mode"]==x["right_mode"]=="N")
 if not complete:conclusion="incomplete implementation attempt; no repeatability conclusion"
 elif raw_exact and prepared_exact:conclusion="not reproduced: all24 capture bundles and processed inputs are exact"
 elif raw_exact and not prepared_exact:conclusion="downstream processing discrepancy"
 elif nn_diff and graphics_exact:conclusion="variation reproduced during consecutive ordinary captures without sidecars"
 else:conclusion="variation observed; inspect mode/order matrix before attributing cause"
 summary={"complete_preregistered_matrix":complete,"bundles":len(bundles),"native_images":2*len(bundles),"mode_counts":{m:sum(b["mode"]==m for b in bundles) for m in ["N","P"]},"reconstruction_qualified":q["passed"],"all_retained_physical_controller_states_exact":all(b["read_only"] for b in bundles),"all_native_images_exact_first":raw_exact,"all_prepared_tensors_exact_first":prepared_exact,"all_available_active_render_inputs_exact_first_per_camera":graphics_exact,"physics_steps_after_boundary":0,"model_inference_calls":0,"conclusion":conclusion,"live01_p_parity_status":"still failed","historical_render_context_reconstructed":False,"max_changed_pixels_vs_first":max(x["changed_pixels"] for x in changes if x["reference"]=="first_bundle"),"max_channel_difference_vs_first":max(x["max_channel_difference"] for x in changes if x["reference"]=="first_bundle"),"secondary_pixel_24_417_values":sorted({tuple(b["rgb"]["secondary"][417,24]) for b in bundles}),"not_claimed":["live end-to-end transparency","absence of dynamic/context/timing effects","sidecar harmlessness in all executions"]}
 summary["secondary_pixel_24_417_values"]=[list(map(int,v)) for v in summary["secondary_pixel_24_417_values"]]
 dump("validation-summary.json",summary)
 # Figure generated entirely from retained arrays after all captures.
 chosen=[first,next((b for b in bundles if b["mode"]=="P"),first)]
 sheet=np.full((1184,1024,3),22,np.uint8)
 for r,b in enumerate(chosen):
  y=r*592;label=f"Bundle {b['bundle']:02d} | {b['mode']} | fixed tick190 | native512 RGB"
  cv2.putText(sheet,label,(16,y+30),cv2.FONT_HERSHEY_SIMPLEX,.75,(230,230,230),2)
  for j,c in enumerate(["primary","secondary"]):
   sheet[y+56:y+568,j*512:(j+1)*512]=b["rgb"][c]
   cv2.putText(sheet,c,(16+j*512,y+53),cv2.FONT_HERSHEY_SIMPLEX,.55,(230,230,230),1)
  cv2.rectangle(sheet,(512+24-6,y+56+417-6),(512+24+6,y+56+417+6),(255,80,80),1)
 cv2.imwrite(str(R/"contact-sheet.png"),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
 print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
