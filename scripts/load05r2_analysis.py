import pathlib,json,gzip,csv,hashlib
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=pathlib.Path('/var/tmp/typhon-load05r2-20261014');D=OUT/'analysis';D.mkdir(exist_ok=False)
roots={'C→C':pathlib.Path('/var/tmp/typhon-load05r-20261014/continuations/C-to-C'),'B→B':pathlib.Path('/var/tmp/typhon-load05r-20261014/continuations/B-to-B'),'C→B':OUT/'continuations/C-to-B','B→C':OUT/'continuations/B-to-C'}
rows=[];traces={};media={}
for label,p in roots.items():
 tr=[json.loads(x) for x in gzip.open(p/'trace.jsonl.gz','rt')];traces[label]=tr
 first=next(x for x in tr if x['object_truth']['grasp_contact']);pre=next(x for x in tr if x['step']==first['step']-1)
 cs=[x for x in first['contacts_full'] if any('gripper' in x[k] for k in ['geom1','geom2']) and any('cube' in x[k] for k in ['geom1','geom2'])]
 finger=next(x for x in pre['finger_geoms'] if x['name']=='gripper0_finger1_collision')
 truth=pre['object_truth'];vel=np.array(finger['body_velocity'])-np.array(truth['object_linear_velocity_m_s'])
 r={'cell':label,**json.loads((p/'summary.json').read_text()),'first_close_tick':next((x['step'] for x in tr if x['command'][-1]>0),None),'command_at_first_contact':'close' if first['command'][-1]>0 else 'open','model_grip_probability':first['target'][9],'finger1_vertical_relative_speed_before_contact_mm_s':float(vel[2]*1000),'finger_joint_opening_at_contact_mm':float(np.abs(first['object_truth']['gripper_joint_position_m']).sum()*1000),'first_contact_robot_geometries':sorted({x[k] for x in cs for k in ['geom1','geom2'] if 'gripper' in x[k]}),'first_contact_object_local_points_mm':(np.array([x['object_local_position'] for x in cs])*1000).tolist(),'initial_state_source':str(p),'evidence':'reused R native control' if label[0]==label[-1] else 'new R2 crossed continuation'}
 rows.append(r);media[label]={'path':str(p/'rollout.mp4'),'sha256':hashlib.sha256((p/'rollout.mp4').read_bytes()).hexdigest()}
(D/'acquisition-details.json').write_text(json.dumps(rows,indent=2))
keys=['cell','evidence','first_contact','command_at_first_contact','first_bilateral','first_supported_lift','supported_lift_tilt_deg','max_object_tilt_deg','finger1_vertical_relative_speed_before_contact_mm_s','first_close_tick']
with (D/'acquisition-table.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
colors={'C→C':'#4361ee','B→B':'#d62828','C→B':'#2a9d8f','B→C':'#9c36b5'}
fig,ax=plt.subplots(2,1,figsize=(9,6),sharex=True,layout='constrained')
for label,tr in traces.items():
 x=[r['step'] for r in tr];ax[0].plot(x,[r['object_tilt_deg'] for r in tr],label=label,color=colors[label]);ax[1].plot(x,[abs(r['object_truth']['gripper_joint_position_m'][0])+abs(r['object_truth']['gripper_joint_position_m'][1]) for r in tr],label=label,color=colors[label])
ax[0].axhline(15,color='grey',ls=':',lw=.8);ax[0].set_ylabel('Brick tilt from upright (degrees)');ax[0].legend(title='Prefix → continuation')
ax[1].set_ylabel('Sum of finger joint travel (m)');ax[1].set_xlabel('Control tick (20 Hz; branch starts after tick 100)')
fig.suptitle('LOAD05-R/R2: only B continuing from its own prefix tips the brick\nOne development scene and noise seed; 90-step acquisition diagnostic')
fig.savefig(D/'acquisition-traces.png',dpi=160);plt.close(fig)
# Four source videos, unchanged frames, with explicit condition/evidence labels.
order=['C→C','C→B','B→C','B→B'];videos={k:[frame for frame in imageio.get_reader(media[k]['path'])] for k in order}
lengths={k:len(v) for k,v in videos.items()};assert len(set(lengths.values()))==1,lengths
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
assembled=[]
for i in range(next(iter(lengths.values()))):
 tiles=[]
 for k in order:
  frame=videos[k][i];h,w=frame.shape[:2];im=Image.new('RGB',(w,h+48),'#101722');im.paste(Image.fromarray(frame),(0,48));draw=ImageDraw.Draw(im)
  draw.text((10,7),k+'  |  '+('reused native control' if k[0]==k[-1] else 'new crossover'),font=font,fill='white');tiles.append(np.asarray(im))
 assembled.append(np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]))
imageio.mimwrite(str(D/'four-cell-acquisition.mp4'),assembled,fps=5,codec='libx264',macro_block_size=1)
imageio.imwrite(D/'four-cell-final-frame.png',assembled[-1])
n=sum(1 for _ in imageio.get_reader(D/'four-cell-acquisition.mp4'));assert n==len(assembled)
(D/'media-manifest.json').write_text(json.dumps({'inputs':media,'source_frame_counts':lengths,'output_frames':n,'fps':5,'all_output_frames_decoded':True,'no_simulator_render_or_inference':True},indent=2))
print(json.dumps([{k:r[k] for k in keys} for r in rows],indent=2))
