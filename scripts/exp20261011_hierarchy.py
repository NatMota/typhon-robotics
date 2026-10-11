"""Observable phase diagnostics and explicit primitive adapter. No simulator import.

Training labels are archived geometric-controller phases. Online functions accept
RGB, measured robot telemetry, supplied goal/camera calibration; never object truth.
The phase head is a classifier, not token reasoning or native X-VLA tool calling.
"""
import argparse, copy, gzip, hashlib, json, math, sys
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
OUT=Path('/var/tmp/typhon-research-20261011/hierarchy')
DATA=ROOT/'frozen_results/personal-m3-20261009/data-canonical'
PHASES=['approach','grasp','lift','orient','place','release','retreat']
PHASE_MAP={'approach_and_align':'approach','descend_to_grasp':'approach','grasp_close':'grasp','lift':'lift','transport_and_reorient':'orient','place_lower':'place','release_open':'release','retreat':'retreat'}
TRAIN={'personal_demo_'+s for s in ['0001','0002','0003','0005','0012','0013']}
VAL={'personal_demo_'+s for s in ['0006','0014','0016']}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def serial(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,Path):return str(v)
 raise TypeError(type(v).__name__)
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,default=serial,allow_nan=False)+'\n')
def color_measure(rgb):
 """Transparent color baseline; missing/ambiguous objects explicitly unknown."""
 hsv=cv2.cvtColor(np.asarray(rgb),cv2.COLOR_RGB2HSV)
 mask=cv2.inRange(hsv,(3,95,55),(29,255,255))
 n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
 candidates=sorted([(int(s[4]),i,s) for i,s in enumerate(stats) if i and 25<=s[4]<=rgb.shape[0]*rgb.shape[1]*.08],reverse=True)
 empty=np.zeros(265,np.float64)
 if not candidates or (len(candidates)>1 and candidates[1][0]>.6*candidates[0][0]):return empty,{'visible':False,'reason':'orange_missing_or_ambiguous'}
 area,i,(x,y,w,h,_)=candidates[0];x,y,w,h=map(int,(x,y,w,h));cx,cy=centers[i]
 patch=hsv[max(0,y-5):min(rgb.shape[0],y+h+5),max(0,x-5):min(rgb.shape[1],x+w+5)]
 blue=cv2.inRange(patch,(88,70,25),(137,255,255));ys,xs=np.where(blue>0)
 glyph=np.zeros((16,16),np.float64)
 if len(xs)>=6:glyph=cv2.resize(blue[ys.min():ys.max()+1,xs.min():xs.max()+1],(16,16),interpolation=cv2.INTER_AREA)/255.
 basic=np.array([1.,cx/rgb.shape[1],cy/rgb.shape[0],w/rgb.shape[1],h/rgb.shape[0],area/(rgb.shape[0]*rgb.shape[1]),area/(w*h),len(xs)/max(area,1),float(len(xs)>=6)])
 info={'visible':True,'bbox_xywh':[x,y,w,h],'center_px':[float(cx),float(cy)],'orange_pixels':area,'fill':float(area/(w*h)),'blue_pixels':int(len(xs)),'orientation_observable':len(xs)>=6,'glyph':glyph}
 return np.r_[basic,glyph.ravel()],info

def observed_features(rgb,proprio,task,previous=None,dt=.2):
 allowed={'eef_position_m','joint_position_rad','gripper_joint_position_m','eef_quat_xyzw','joint_velocity_rad_s'}
 if set(proprio)-allowed:raise ValueError('unexpected proprio channel')
 p=np.asarray(proprio['eef_position_m'],float);q=np.asarray(proprio['joint_position_rad'],float);g=np.asarray(proprio['gripper_joint_position_m'],float)
 if p.shape!=(3,) or q.shape!=(7,) or g.shape!=(2,) or not all(np.isfinite(x).all() for x in [p,q,g]):raise ValueError('invalid measured telemetry')
 views=[color_measure(rgb[k]) for k in ['primary','secondary']]
 goal=np.asarray(task['target_position_xy_m']);yaw=float(task['target_yaw_rad'])
 state=np.r_[p,q,np.abs(g),goal,[math.sin(yaw),math.cos(yaw)]]
 # The previous measured state is causal. No timestamp/episode fraction enters the fit.
 temporal=np.zeros(7)
 if previous is not None:
  temporal[:3]=(p-previous['eef'])/dt
  for i,(_,info) in enumerate(views):
   if info['visible'] and previous['views'][i]['visible']:
    temporal[3+i*2:5+i*2]=(np.asarray(info['center_px'])-previous['views'][i]['center_px'])/512/dt
 vector=np.r_[state,*(v[0] for v in views),temporal]
 return vector,{'eef':p.copy(),'views':[v[1] for v in views]}

def make_dataset(out):
 from m7_export_catalog_v2 import derive_primitives
 episodes=json.loads((DATA/'dataset.json').read_text())['episodes']
 xs=[];ys=[];meta=[];evidence=[]
 selected=[e for e in episodes if e['id'] in TRAIN|VAL]
 assert {e['id'] for e in selected}==TRAIN|VAL
 for e in selected:
  expected='P_train' if e['id'] in TRAIN else 'P_validation';assert e['split']==expected
  path=DATA/e['path'];assert sha(path)==e['sha256']
  folder=DATA/e['source_path'].split('data-canonical/',1)[1]
  ep=folder/'episode.json';act=folder/'robot_actions.jsonl'
  assert sha(ep)==e['source_episode_sha256'] and sha(act)==e['source_action_sha256']
  episode=json.loads(ep.read_text());actions=[json.loads(l) for l in act.read_text().splitlines()]
  with np.load(path,allow_pickle=False) as z:
   parts,contexts=derive_primitives(dict(e,targets=z['targets'].tolist()),{'episode':episode,'actions':actions})
   previous=None;frames=z['frame_action_indices'];primary=z['primary'];secondary=z['secondary']
   for fi,idx0 in enumerate(frames):
    idx=int(idx0)
    # Dataset contains current camera before action idx. Retrieve only pre-action measured robot fields.
    if idx==0:continue  # Initial joint vector was not separately archived; no future substitution.
    before=actions[idx-1];truth=before['post_action_truth']
    proprio={'eef_position_m':truth['eef_position_m'],'joint_position_rad':before['joint_position_rad'],'gripper_joint_position_m':truth['gripper_joint_position_m']}
    x,previous=observed_features({'primary':primary[fi],'secondary':secondary[fi]},proprio,e['config']['task'],previous)
    part=next((p for p in parts if p['action_start']<=idx<p['action_end_exclusive']),None)
    if part is None:continue
    phase=PHASE_MAP[part['label']];xs.append(x);ys.append(PHASES.index(phase))
    meta.append({'take':e['id'],'split':e['split'],'frame_index':fi,'action_index':idx,'phase':phase,'teacher_label':part['label'],'teacher_phase_start':part['action_start'],'teacher_phase_end':part['action_end_exclusive']})
   evidence.append({'take':e['id'],'split':e['split'],'npz_sha256':e['sha256'],'episode_sha256':e['source_episode_sha256'],'robot_actions_sha256':e['source_action_sha256'],'primitives':parts,'excluded_context':contexts})
  print('extracted',e['id'],sum(m['take']==e['id'] for m in meta),flush=True)
 np.savez_compressed(out/'phase-features.npz',x=np.array(xs),y=np.array(ys))
 save(out/'phase-rows.json',meta);save(out/'source-evidence.json',evidence)
 return np.array(xs),np.array(ys),meta

def fit_head(x,y):
 # Small class-balanced ridge classifier. Fixed regularization, no tuning on held-outs.
 mean=x.mean(0);scale=np.maximum(x.std(0),.03);z=np.c_[(x-mean)/scale,np.ones(len(x))]
 n=np.bincount(y,minlength=len(PHASES));weight=1/np.maximum(n[y],1);weight*=len(x)/weight.sum()
 zz=z*np.sqrt(weight[:,None]);yy=np.eye(len(PHASES))[y]*np.sqrt(weight[:,None])
 reg=np.eye(z.shape[1])*10.;reg[-1,-1]=0.
 w=np.linalg.solve(zz.T@zz+reg,zz.T@yy)
 return {'mean':mean,'scale':scale,'w':w}
def predict(head,x):
 scores=np.c_[(x-head['mean'])/head['scale'],np.ones(len(x))]@head['w']
 order=np.argsort(scores,axis=1);margin=scores[np.arange(len(x)),order[:,-1]]-scores[np.arange(len(x)),order[:,-2]]
 return scores.argmax(1),margin,scores

def metrics(y,p):
 cm=np.zeros((len(PHASES),len(PHASES)),int)
 for a,b in zip(y,p):cm[a,b]+=1
 f1=[]
 for i in range(len(PHASES)):
  den=cm[i].sum()+cm[:,i].sum();f1.append(2*cm[i,i]/den if den else 0.)
 return {'n':len(y),'accuracy':float(np.mean(y==p)) if len(y) else None,'macro_f1':float(np.mean(f1)),'per_phase_f1':dict(zip(PHASES,f1)),'confusion_rows_truth_columns_predicted':cm.tolist(),'labels':PHASES}

def fit(out):
 out.mkdir(parents=True,exist_ok=True)
 if (out/'phase-model.npz').exists():raise FileExistsError('create-only model output')
 protocol={'schema':'hierarchy-observable-phase-v1','training_takes':sorted(TRAIN),'untouched_heldout_takes':sorted(VAL),'phases':PHASES,'label_origin':'Exact archived geometric expert command boundaries; orient label includes transport/reorient composite. No human phase labels implied.','features':'Current primary/secondary RGB color/glyph; pre-action measured EEF xyz,7jointpositions,2gripperpositions; causal difference from previous 5Hzframe; supplied goal. No future actions, objecttruth, phase, index, time or fraction.','model':'class-balanced ridge onehot classifier, fixed lambda10; CPU NumPy','abstention':'margin threshold selected on leave-one-take-out P_train for >=90% accuracy if any threshold qualifies, else disabled online. Heldout not used for tuning.','online_phase_head_enabled':False,'online_acceptance_gate':'heldout selective accuracy>=0.90 and coverage>=0.50, every phase represented; otherwise phase prediction advisory only','note':'A learned phase classifier is not a language reasoning head or native VLA tool caller.'}
 save(out/'protocol.json',protocol)
 x,y,rows=make_dataset(out);train=np.array([r['split']=='P_train' for r in rows]);val=~train
 folds=[];op=np.full(len(y),-1);om=np.full(len(y),np.nan)
 for take in sorted(TRAIN):
  test=np.array([r['take']==take for r in rows]);head=fit_head(x[train&~test],y[train&~test]);p,m,_=predict(head,x[test]);op[test]=p;om[test]=m
  folds.append({'take':take,**metrics(y[test],p)})
 options=[]
 for threshold in [0.,.05,.1,.15,.2,.3,.4,.5,.75,1.]:
  keep=train&(om>=threshold)
  if keep.any():options.append({'threshold':threshold,'coverage':float(keep.sum()/train.sum()),'accuracy':float(np.mean(op[keep]==y[keep]))})
 valid=[o for o in options if o['accuracy']>=.9]
 threshold=max(valid,key=lambda o:o['coverage'])['threshold'] if valid else 1e9
 head=fit_head(x[train],y[train]);p,margin,scores=predict(head,x);keep=val&(margin>=threshold)
 np.savez_compressed(out/'phase-model.npz',**head,threshold=np.array(threshold),classes=np.array(PHASES))
 preds=[dict(r,predicted=PHASES[int(a)],margin=float(m),accepted=bool(m>=threshold)) for r,a,m in zip(rows,p,margin)]
 save(out/'predictions.json',preds)
 timing=[]
 for take in sorted(VAL):
  rr=[r for r in preds if r['take']==take]
  for phase in PHASES:
   actual=next((r['action_index'] for r in rr if r['phase']==phase),None)
   predicted=next((r['action_index'] for r in rr if r['predicted']==phase and r['accepted']),None)
   timing.append({'take':take,'phase':phase,'first_teacher_sample':actual,'first_accepted_prediction':predicted,'offset_seconds':None if actual is None or predicted is None else (predicted-actual)/20})
 result={'P_train':metrics(y[train],p[train]),'leave_one_take_out':metrics(y[train],op[train]),'folds':folds,'threshold_candidates':options,'selected_margin_threshold':threshold,'P_validation':metrics(y[val],p[val]),'P_validation_selective':metrics(y[keep],p[keep]),'heldout_coverage':float(keep.sum()/val.sum()),'transitions':timing,'release_before_teacher_phase_heldout_frames':sum(r['split']=='P_validation' and r['accepted'] and r['predicted']=='release' and r['phase'] not in ['release','retreat'] for r in preds),'raw_take_count_train':len(TRAIN),'raw_take_count_validation':len(VAL)}
 result['online_gate_passed']=bool(keep.any() and result['P_validation_selective']['accuracy']>=.9 and result['heldout_coverage']>=.5 and {r['phase'] for r in preds if r['split']=='P_validation' and r['accepted']}==set(PHASES))
 save(out/'phase-validation.json',result)
 print(json.dumps({k:result[k] for k in ['leave_one_take_out','P_validation','P_validation_selective','heldout_coverage','online_gate_passed']},default=serial),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['fit']);ap.add_argument('--output',type=Path,default=OUT/'phase-head-v1');a=ap.parse_args();fit(a.output)
