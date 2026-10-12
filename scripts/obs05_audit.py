"""Independent OBS05 trace audit; does not run or change a controller."""
import argparse, csv, gzip, hashlib, json, math
from pathlib import Path
import cv2
import numpy as np
MODES=("contour_legacy","contour_fresh","glyph_legacy","glyph_fresh")
def save(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False)+"\n")
def audit(out):
 summary=json.loads((out/"summary.json").read_text())
 allrows=[]; bycase={}; percase=[]; events=[]
 for p in sorted(out.glob("*/observations.jsonl.gz")):
  rows=[json.loads(l) for l in gzip.open(p,"rt")];case=p.parent.name;bycase[case]=rows
  allrows += rows
  item={"case":case,"samples":len(rows),"contact_samples":sum(r["scorer_only"]["bilateral_contact"] for r in rows),"lift_samples":sum(r["scorer_only"]["supported_lift"] for r in rows)}
  for mode in MODES:
   item[mode]={}
   for label in ("bilateral_contact","supported_lift"):
    tp=sum(r["predictions"]["real"][mode]["held"] and r["scorer_only"][label] for r in rows)
    fp=sum(r["predictions"]["real"][mode]["held"] and not r["scorer_only"][label] for r in rows)
    fn=sum(not r["predictions"]["real"][mode]["held"] and r["scorer_only"][label] for r in rows)
    item[mode][label]={"tp":tp,"fp":fp,"fn":fn}
  percase.append(item)
  for i,r in enumerate(rows):
   if not (r["predictions"]["real"]["contour_legacy"]["held"] and not r["scorer_only"]["bilateral_contact"]): continue
   a=max(0,i-2);b=min(len(rows),i+3)
   contexts=[]
   for j in range(a,b):
    z=rows[j];t=z["scorer_only"];ev=z["evidence"]
    contexts.append({"index":j,"step":z["step"],"sim_time_s":t["truth"]["sim_time_s"],"contact":t["bilateral_contact"],"supported_lift":t["supported_lift"],"lift_mm":t["lift_m"]*1000,"span_mm":ev["span_m"]*1000,"object_position_m":t["truth"]["object_position_m"],"held":{m:z["predictions"]["real"][m]["held"] for m in MODES},"motion_positive":{"contour":ev["contour"][0],"glyph":ev["glyph"][0]}})
   events.append({"case":case,"index":i,"step":r["step"],"previous_sample_contact":i>0 and rows[i-1]["scorer_only"]["bilateral_contact"],"any_contact_in_case":item["contact_samples"]>0,"context":contexts})
 # Recalculate metrics without invoking experiment analyzer.
 for mode in MODES:
  for label in ("bilateral_contact","supported_lift"):
   ref=summary["metrics"][mode][label]
   tp=sum(x[mode][label]["tp"] for x in percase);fp=sum(x[mode][label]["fp"] for x in percase);fn=sum(x[mode][label]["fn"] for x in percase)
   assert (tp,fp,tp+fn)==(ref["true_positive"],ref["false_positive"],ref["positive_labels"])
 for r in allrows:
  truth=r["scorer_only"]
  assert truth["supported_lift"]==(truth["bilateral_contact"] and truth["lift_m"]>=.01)
  for key in ("frozen_rgb","fixed_eef"):
   assert not any(v["held"] for v in r["predictions"][key].values())
 held=[r for r in allrows if r["predictions"]["real"]["contour_legacy"]["held"]]
 lowcontact=sum(r["scorer_only"]["bilateral_contact"] and not r["scorer_only"]["supported_lift"] for r in held)
 assert len(events)==8 and lowcontact==26
 assert sum(e["previous_sample_contact"] for e in events)==5
 assert sum(not e["any_contact_in_case"] for e in events)==3
 diagnostic={"sample_count":len(allrows),"independent_metric_recalculation_passed":True,"label_formula_checked":True,"legacy_supported_false_positives":{"bilateral_contact_below_lift_threshold":lowcontact,"no_bilateral_contact":len(events)},"false_contact_events":events,"per_case":percase,"units":"Samples at 5 Hz; correlated observations, not independent trials.","visuals":[]}
 # Deterministically select three evidence examples. Decode videos sequentially.
 picks=[
  ("empty-grasp","broad-local-demo_0003-yawplus15deg-seed7417",60),
  ("release-lag","expert-personal_demo_0002",52)]
 for case,rows in bycase.items():
  missed=[i for i,r in enumerate(rows) if r["scorer_only"]["supported_lift"] and r["predictions"]["real"]["contour_legacy"]["held"] and not r["predictions"]["real"]["glyph_fresh"]["held"]]
  if missed:
   picks.append(("missed-lift",case,missed[len(missed)//2]));break
 cards=[]
 for label,case,index in picks:
  rows=bycase[case];r=rows[index]
  cap=cv2.VideoCapture(str(out/case/"preview.mp4"));frame=None
  for k in range(index+1):
   ok,frame=cap.read();assert ok
  cap.release()
  # Header+two unmodified camera views from a sequentially decoded preview frame.
  title=label+" | "+case
  panel=cv2.copyMakeBorder(frame,86,0,0,0,cv2.BORDER_CONSTANT,value=(22,22,22))
  cv2.putText(panel,title,(12,26),cv2.FONT_HERSHEY_SIMPLEX,.53,(245,245,245),1,cv2.LINE_AA)
  s=r["scorer_only"]
  cv2.putText(panel,f"sample {index}; step {r['step']}; contact={s['bilateral_contact']}; lift={s['lift_m']*1000:.2f} mm",(12,52),cv2.FONT_HERSHEY_SIMPLEX,.51,(245,245,245),1,cv2.LINE_AA)
  cv2.putText(panel,"Scorer labels below are diagnostic only; never detector inputs.",(12,74),cv2.FONT_HERSHEY_SIMPLEX,.46,(180,190,220),1,cv2.LINE_AA)
  name=label+".jpg";assert cv2.imwrite(str(out/name),panel,[cv2.IMWRITE_JPEG_QUALITY,93])
  cards.append(panel);diagnostic["visuals"].append({"file":name,"case":case,"sample_index":index,"step":r["step"],"selection":"post-hoc diagnostic example"})
 assert cv2.imwrite(str(out/"diagnostic-contact-sheet.jpg"),np.concatenate(cards,axis=0),[cv2.IMWRITE_JPEG_QUALITY,92])
 save(out/"independent_audit.json",diagnostic)
 with (out/"metrics.csv").open("w",newline="") as f:
  w=csv.writer(f,lineterminator="\n");w.writerow(["method","label","true_positive","false_positive","false_negative","precision","recall"])
  for mode in MODES:
   for label in ("bilateral_contact","supported_lift"):
    m=summary["metrics"][mode][label];w.writerow([mode,label,m["true_positive"],m["false_positive"],m["positive_labels"]-m["true_positive"],m["precision"],m["recall"]])
 print(json.dumps({"samples":len(allrows),"false_contact_events":len(events),"after_contact_loss":5,"empty_grasp_samples":3,"low_height_contact":26,"visuals":diagnostic["visuals"]},indent=2))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);audit(p.parse_args().out)
