"""Audit first policy-input arrays and initial PNGs without rerunning simulations."""
from pathlib import Path
import json,zipfile
import numpy as np
from PIL import Image
OUT=Path("/var/tmp/typhon-load01-20261013");p=json.loads((OUT/"protocol.json").read_text())
def first(path,key):
 with zipfile.ZipFile(path)as z,z.open(key+".npy")as f:
  v=np.lib.format.read_magic(f)
  shape,order,dtype=(np.lib.format.read_array_header_1_0(f)if v==(1,0)else np.lib.format.read_array_header_2_0(f))
  assert not order and dtype==np.uint8
  return np.frombuffer(f.read(int(np.prod(shape[1:]))*dtype.itemsize),dtype=dtype).reshape(shape[1:])
def diff(a,b):
 d=np.abs(a.astype(np.int16)-b.astype(np.int16));mask=np.any(d!=0,axis=-1);coords=np.argwhere(mask)
 return {"array_equal":bool(np.array_equal(a,b)),"changed_channels":int((d!=0).sum()),"changed_pixels":int(mask.sum()),"total_pixels":int(mask.size),"max_channel_difference_255":int(d.max()),"mean_channel_difference_255":float(d.mean()),"changed_bbox_yx":([coords.min(0).tolist(),coords.max(0).tolist()]if len(coords)else None)}
rows=[]
for s in p["scenarios"]:
 phase="source"if s["group"]=="loading_source"else"remainder"
 paths={m:OUT/"evaluation"/m/phase/s["scenario_id"]for m in p["models"]}
 data={m:{k:first(d/"policy_io.npz",k)for k in ["primary","secondary"]}for m,d in paths.items()}
 png={m:np.asarray(Image.open(d/"initial.png"))for m,d in paths.items()}
 rows.append({"id":s["scenario_id"],"comparisons":{m:{"policy_primary":diff(data["P"]["primary"],data[m]["primary"]),"policy_secondary":diff(data["P"]["secondary"],data[m]["secondary"]),"initial_png":diff(png["P"],png[m])}for m in ["F","FL"]}})
summary={"all_first_policy_arrays_exact":all(c[k]["array_equal"]for r in rows for c in r["comparisons"].values()for k in ["policy_primary","policy_secondary"]),"all_initial_png_pixels_exact":all(c["initial_png"]["array_equal"]for r in rows for c in r["comparisons"].values()),"cases":rows,"new_rollouts":0}
(OUT/"analysis/initial-pixel-audit.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps({**{k:v for k,v in summary.items()if k!="cases"},"nonidentical_cases":[r for r in rows if any(not d["array_equal"]for c in r["comparisons"].values()for d in c.values())]},indent=2))
