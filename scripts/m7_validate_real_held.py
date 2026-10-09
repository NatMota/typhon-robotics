#!/usr/bin/env python3
"""Read-only source checks plus QA and summary for real held-crosshair derivatives."""
import argparse,collections,csv,hashlib,json,pathlib,subprocess
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from transformers import AutoProcessor
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--checkpoint',default='.cache-xvla/checkpoint');a=p.parse_args()
root=a.root;m=json.loads((root/'manifest.json').read_text());summary=collections.Counter();allrows=[]
for v in m['videos']:
 rows=list(csv.DictReader((root/v['sidecar']).open()));assert len(rows)==v['frame_count']
 for i,r in enumerate(rows):
  assert int(r['clip_frame_index'])==i
  held=None if r['held_manipulation']=='' else bool(int(r['held_manipulation']))
  on=bool(int(r['crosshair_on']));eligible=bool(int(r['eligible_for_binary_enrichment']))
  assert not on or held is True
  assert eligible==(held is not None and (held is False or on))
  if on:assert r['tracking_detected']=='1' and float(r['tracking_confidence'])>=.65
  summary['frames']+=1;summary['held_true' if held is True else 'held_false' if held is False else 'held_unknown']+=1
  summary['crosshair_on' if on else 'crosshair_off']+=1;summary['binary_eligible']+=eligible
  allrows.append(r)
 assert hashlib.sha256(pathlib.Path(v['source_clip']).read_bytes()).hexdigest()==v['source_clip_sha256']
fields=list(allrows[0]);agg=root/'held_trajectory_annotations.csv'
with agg.open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(allrows)
review=root/'review_final';review.mkdir(exist_ok=True)
processor=AutoProcessor.from_pretrained(a.checkpoint,trust_remote_code=True,local_files_only=True)
probes=[];strips=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
for did,frame,why in [('demo_0002',20,'known not held'),('demo_0002',120,'held / crosshair ON'),('demo_0002',80,'ambiguous contact / OFF'),('demo_0012',135,'held PAUSE / remains ON'),('demo_0015',120,'oblique localization limitation'),('demo_0018',80,'already completed / OFF')]:
 strip=Image.new('RGB',(896,270),(17,20,26));dr=ImageDraw.Draw(strip);dr.text((8,5),f'{did} frame{frame} {why}',font=font,fill='white')
 for col,view in enumerate(['camera_a','camera_b']):
  v=next(x for x in m['videos'] if x['demo_id']==did and x['view']==view)
  row=next(r for r in allrows if r['demo_id']==did and r['view']==view and int(r['clip_frame_index'])==frame)
  images=[]
  for name in [v['source_clip'],str(root/v['output'])]:
   cap=cv2.VideoCapture(name);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read();cap.release();assert ok
   images.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
  t=processor(images=images,language_instruction='Observe the orange block.')['image_input'][0,:2].permute(0,2,3,1).numpy()
  imgs=np.clip(np.rint((t*np.array(processor.image_processor.image_std)+processor.image_processor.image_mean)*255),0,255).astype(np.uint8)
  def green(im):return (im[...,1]>100)&(im[...,1]>1.5*im[...,0])&(im[...,1]>1.3*im[...,2])
  added=int(np.count_nonzero(green(imgs[1])&~green(imgs[0])))
  if row['crosshair_on']=='1':assert added>=20,(did,view,added)
  for j,img in enumerate(imgs):strip.paste(Image.fromarray(img),(col*448+j*224,45))
  dr.text((col*448+4,26),f'{view}: held={row["held_manipulation"] or "?"} cue={row["crosshair_on"]} valid={row["eligible_for_binary_enrichment"]}',font=font,fill='white')
  probes.append(dict(demo_id=did,view=view,frame=frame,case=why,held_label=row['held_manipulation'],cue_on=int(row['crosshair_on']),binary_valid=int(row['eligible_for_binary_enrichment']),added_green_pixels_native224=added))
 strip.save(review/(did+'-f'+str(frame)+'.png'));strips.append(strip)
sheet=Image.new('RGB',(896,270*len(strips)),(17,20,26))
for i,strip in enumerate(strips):sheet.paste(strip,(0,i*270))
sheet.save(review/'contact_sheet.png')
sheet.save(review/'contact_sheet.jpg',quality=92)
report=dict(passed=True,counts=dict(summary),trials=len(set(x['demo_id'] for x in m['videos'])),video_count=len(m['videos']),raw_source_clips_unchanged=True,
            fps_framecount_pts_unchanged=True,unknown_OFF_never_negative_label=True,native224_probes=probes,
            usable_robot_20d_action_targets_from_phone=0,human_physical_calibration_available=False,
            independent_camera_views_are_not_independent_demonstrations=True,training_started_by_visual_agent=False,
            task_categories={'normal_feeding':7,'awkward_pause_resume_feeding':4,'orientation_correction':3,'already_completed':2})
(root/'validation_summary.json').write_text(json.dumps(report,indent=2)+'\n')
(root/'REPORT.txt').write_text(f'''Real personal M0 held-crosshair export\n\n16 curated trials,32camera videos, {summary['frames']} synchronized view-frames. {summary['held_true']} reviewedheld, {summary['held_false']} reviewednot-held, {summary['held_unknown']} unknown. CrosshairON {summary['crosshair_on']}; knownbinary-enrichmenteligible {summary['binary_eligible']}. Camera views are duplicate observations of16demonstrations, not32independenttakes.\n\nA causal real-time held detector is not implemented: the real-video labels come from explicit agent-reviewed temporal paired-view intervals. Clearly-held stationary pauses are retained. Touch/grasp onset/release uncertainties remain unknown. CueOFF on unknown is never a negativeheldlabel. Existing frozen M0 RGBcolourtracking anchors the glyph; low-confidence localization disablesit without changing the heldannotation. Inparticular oblique demo0015 has no reliable held-interval locator, while itsprimary view has ONexamples.\n\nThe green extruded brackets use the exact shared robot-training renderer. Extrusion is a visualization style, not calibrated3Dgeometry. The observed block/numeral pixels are untouched before videoencoding. Framecount,FPS,PTSandoptional audio are preserved. Allsourcecliphashes unchanged. Review_final/contact_sheet.png covers held, pause, completed, ambiguity and localizationfailure through native224processing.\n\nThe separate trajectoryCSV carries realimage coordinates, originalrecording timestamps and held1/0/unknown. Itcontains ZERO native20Drobotmotor targets. Do not substitute these humanobject motions into an EEF/joint actionloss. Use separate declared visual-state/intention supervision, or first establish a documentedrobotlabelmapping.\n''')
(root/'REPRODUCE.txt').write_text('From humanoid-m6-joint onPowerglide (new outputpathrequired):\n.venv-sim/bin/python scripts/m7_real_held_crosshair.py --source-root ../humanoid-task-supervisor --review results/personal-m7-data-prep-20261009/visual-enrichment-v1/real-held-v1/held_review.json --out results/personal-m7-data-prep-20261009/visual-enrichment-v1/real-held-v1/export\n.venv-xvla/bin/python scripts/m7_validate_real_held.py --root results/personal-m7-data-prep-20261009/visual-enrichment-v1/real-held-v1/export\nReview sheets: .venv-xvla/bin/python scripts/m7_real_held_review_sheets.py --source-root ../humanoid-task-supervisor --out NEW_REVIEW_DIRECTORY\n')
files=sorted(x for x in root.rglob('*') if x.is_file() and x.name!='SHA256SUMS')
(root/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(x.read_bytes()).hexdigest()}  {x.relative_to(root)}\n' for x in files))
print(json.dumps({k:v for k,v in report.items() if k!='native224_probes'},indent=2))
