import pathlib,json,gzip,cv2,numpy as np
root=pathlib.Path('/var/tmp/typhon-obs06-aperture-20261013')
out=pathlib.Path(__file__).resolve().parents[1]/'experiment-history/2026-10-13-obs06';out.mkdir(parents=True,exist_ok=True)
summary=json.loads((root/'summary.json').read_text());allrows=[];bad=[];bycase={}
for f in sorted(root.glob('*/observations.jsonl.gz')):
 rows=[json.loads(l) for l in gzip.open(f,'rt')];bycase[f.parent.name]=rows;allrows+=rows
 for i,r in enumerate(rows):
  if r['predictions']['both']['held'] and not r['scorer_only']['bilateral_contact']:bad.append({'case':f.parent.name,'index':i,'step':r['step'],'any_grasp_contact':r['scorer_only']['truth']['grasp_contact'],'lift_mm':r['scorer_only']['lift_m']*1000})
for mode,m in summary['metrics'].items():
 for label,expected in m.items():
  tp=sum(r['predictions'][mode]['held'] and r['scorer_only'][label] for r in allrows)
  fp=sum(r['predictions'][mode]['held'] and not r['scorer_only'][label] for r in allrows)
  fn=sum(not r['predictions'][mode]['held'] and r['scorer_only'][label] for r in allrows)
  assert (tp,fp,fn)==(expected['tp'],expected['fp'],expected['fn'])
result={'samples':len(allrows),'independent_counts_match':True,'residual_false_bilateral_contact':bad,'residual_any_contact':sum(r['any_grasp_contact'] for r in bad),'residual_no_contact':sum(not r['any_grasp_contact'] for r in bad),'false_contact_cases':len(set(r['case'] for r in bad))}
(out/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
picks=[bad[0],next(r for r in bad if r['any_grasp_contact'])]
images=[]
for j,p in enumerate(picks):
 cap=cv2.VideoCapture(str(root/p['case']/'preview.mp4'))
 for i in range(p['index']+1):ok,im=cap.read();assert ok
 cap.release();cv2.putText(im,'Scorer: bilateral contact FALSE; any finger contact '+str(p['any_grasp_contact']),(8,67),cv2.FONT_HERSHEY_SIMPLEX,.42,(100,180,255),1)
 images.append(im)
cv2.imwrite(str(out/'residual-errors.jpg'),np.concatenate(images,axis=0),[cv2.IMWRITE_JPEG_QUALITY,92])
print(json.dumps({k:v for k,v in result.items() if k!='residual_false_bilateral_contact'}))
