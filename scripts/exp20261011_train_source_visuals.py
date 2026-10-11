"""Descriptive frozen-state coverage and post-hoc source examples, no model calls."""
import gzip,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,'/var/tmp/typhon-research-20261011/plot-deps')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/var/tmp/typhon-research-20261011/broad-pt')
DEST=OUT/'source-competence/figures';DEST.mkdir(exist_ok=True)
def main():
 m=json.loads((ROOT/'frozen_results/personal-m3-20261009/data-canonical/dataset.json').read_text());train=[e for e in m['episodes'] if e['split']=='P_train'];val=[e for e in m['episodes'] if e['split']=='P_validation'];assert len(train)==6 and len(val)==3
 xy=np.array([e['config']['initial_state']['position_xy_m'] for e in train]);yaw=np.array([e['config']['initial_state']['yaw_rad'] for e in train]);h=ConvexHull(xy);rows=[]
 for e in train+val:
  pos=np.array(e['config']['initial_state']['position_xy_m']);a=e['config']['initial_state']['yaw_rad'];distance=np.linalg.norm(xy-pos,axis=1);yawdistance=np.abs(np.degrees(np.arctan2(np.sin(yaw-a),np.cos(yaw-a))))
  rows.append({'id':e['id'],'split':e['split'],'initial_xy_m':pos.tolist(),'initial_yaw_deg':float(np.degrees(a)),'inside_source_position_convex_hull':bool(np.all(h.equations[:,:2]@pos+h.equations[:,2]<=1e-9)),'nearest_source_position_mm':float(distance.min()*1000),'nearest_source_position_id':train[int(distance.argmin())]['id'],'nearest_source_yaw_difference_deg':float(yawdistance.min()),'nearest_source_yaw_id':train[int(yawdistance.argmin())]['id']})
 (OUT/'state-coverage.json').write_text(json.dumps({'scope':'Descriptive exact configuration geometry, not a causal explanation or combined pose similarity metric. Position and yaw nearest neighbours may be different takes; no new data or evaluation.','rows':rows},indent=2)+'\n')
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
 fig,ax=plt.subplots(figsize=(7,5.2));verts=xy[h.vertices];ax.fill(verts[:,0]*1000,verts[:,1]*1000,color='#3E8E80',alpha=.13,label='Six-source position convex hull')
 for split,color,marker in [('P_train','#236A61','o'),('P_validation','#C36725','X')]:
  rs=[r for r in rows if r['split']==split];coords=np.array([r['initial_xy_m'] for r in rs]);ax.scatter(coords[:,0]*1000,coords[:,1]*1000,c=color,marker=marker,s=70,label='Source starts (6)' if split=='P_train' else 'Development holdouts (3)')
  for r in rs:
   p=np.array(r['initial_xy_m'])*1000;ax.annotate(r['id'].replace('personal_demo_','')+f" ({r['initial_yaw_deg']:.0f}°)",p,xytext=(5,6),textcoords='offset points',fontsize=8,color=color)
 ax.scatter([125],[-120],marker='s',facecolor='none',edgecolor='#333333',s=160,label='Common goal centre')
 ax.set(xlabel='Simulator world X (mm)',ylabel='Simulator world Y (mm)',title='All three holdout starts require position extrapolation');ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(loc='lower left',fontsize=8)
 fig.text(.02,.025,'Exact supplied simulator starts; labels show signed initial yaw. Hull membership is descriptive, not causal.',fontsize=8);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(DEST/'initial-state-coverage.png',dpi=180);fig.savefig(DEST/'initial-state-coverage.svg')
 canvas=Image.new('RGB',(1590,980),'white');d=ImageDraw.Draw(canvas)
 try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
 except OSError:font=ImageFont.load_default()
 d.text((12,10),'Post-hoc illustrative source cases: initial / observation near maximum lift / final (both cameras)',fill='black',font=font);provenance=[]
 labels={'0001':'Holds away from goal; strict fail','0002':'Strict success','0005':'In square; 15.80 mm centre error misses 15 mm tolerance'}
 for ri,(take,label) in enumerate(labels.items()):
  folder=OUT/'source-competence/evaluation/broad-pt-12000'/f'ptrain-{take}-seed7417'
  trace=[json.loads(s) for s in gzip.open(folder/'trajectory.jsonl.gz','rt')];peak=max(trace,key=lambda r:r['truth']['object_position_m'][2]);a=np.load(folder/'policy_io.npz');query_steps=40+30*np.arange(len(a['primary']));idx=int(np.argmin(abs(query_steps-peak['step'])));middle=Image.fromarray(np.concatenate([a['primary'][idx],a['secondary'][idx]],axis=1))
  images=[Image.open(folder/'initial.png').convert('RGB'),middle,Image.open(folder/'final.png').convert('RGB')]
  y=49+ri*300;d.text((12,y),f'Take {take}: {label}',fill='black',font=font)
  for col,im in enumerate(images):im.thumbnail((514,257));canvas.paste(im,(12+col*525,y+28))
  provenance.append({'take':take,'seed':7417,'maximum_lift_step':peak['step'],'middle_policy_query_step':int(query_steps[idx]),'middle_policy_io_index':idx,'selection':'Post-hoc examples: held-stall, strict success, centre-tolerance-only failure. Mid-frame is nearest saved model observation, not exact peak contact image.'})
 d.text((12,956),'Illustrative selections only; all six rollouts and outcomes remain in the report.',fill='black',font=font)
 canvas.save(DEST/'representative-source-cases.jpg',quality=92)
 (DEST/'visual-provenance.json').write_text(json.dumps({'cases':provenance,'state_plot_source':'frozen canonical dataset.json exact initial configs','all_primary_outcomes_unchanged':True},indent=2)+'\n')
 print('Saved state coverage and source-case contact sheet')
if __name__=='__main__':main()
