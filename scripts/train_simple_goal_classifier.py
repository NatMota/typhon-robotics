"""Generate independent rendered supervision, fit a tiny CPU MLP, then test once."""
import argparse,copy,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import cv2
from supervisor.planning.simple_goal_classifier import rgb_features,GoalSuccessClassifier
OUT=Path('/var/tmp/typhon-goal-classifier-20261009')
SEEDS={'train':2026100911,'validation':2026100922,'test':2026100933}
COUNTS={'train':720,'validation':240,'test':360}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')

def generate():
    from supervisor.feeding_loading.config import DEFAULTS
    from supervisor.feeding_loading.environment import FeedingLoadingEnv
    from scipy.spatial.transform import Rotation
    OUT.mkdir(exist_ok=True);(OUT/'images').mkdir(exist_ok=True)
    protocol={'seeds':SEEDS,'counts':COUNTS,'fit_uses_existing_pilot_frames':False,
      'label_source':'independent offline commanded simulator geometry plus declared synthetic occlusion, never classifier predictions',
      'goal':'declared 15mm target-center tolerance, entire footprint inside, directed yaw <=10deg, marked face up',
      'test_not_used_for_threshold_or_fitting':True,'render_engine':'isolated EGL slot; training and inference CPU',
      'split_camera_fovy':{'train':[33,36,39],'validation':[34,38],'test':[35,37]},
      'occlusion':'Opaque rectangles over 40-85% of blue glyph (unknown), or small orange-corner occlusion (readable).',
      'release_stability':'not trained labels; required external measured gates'}
    dump(OUT/'protocol_before_generation.json',protocol)
    rows=[];features=[];labels=[];splits=[];start=time.monotonic()
    for geom in range(2):
        cfg=copy.deepcopy(DEFAULTS);cfg['object']['visual_version']='duplo_31110pb074_v2'
        if geom:cfg['object'].update(width_m=.032,depth_m=.043,height_m=.032)
        cfg['cameras']['overhead'].update(position_m=[0,0,2.2],fovy_deg=36,width=512,height=512)
        cfg['cameras']['oblique'].update(width=512,height=512)
        cfg['initial_state'].update(position_jitter_m=0,yaw_choices_rad=[0])
        env=FeedingLoadingEnv(cfg);env.reset(990100+geom)
        sim=env.env.sim;camid=sim.model.camera_name2id('m1_overhead')
        stageid=sim.model.body_mocapid[sim.model.body_name2id('m1_stage')]
        for split,count in COUNTS.items():
            rng=np.random.default_rng(SEEDS[split]+geom*100000)
            fovs=protocol['split_camera_fovy'][split]
            for i in range(count//2):
                category=i%12
                target=rng.uniform([-.15,-.2],[.16,.05]);targetyaw=float(rng.uniform(-.25,.25))
                angle=float(rng.uniform(-8,8));roll=pitch=0.;r=float(rng.uniform(0,.0105))
                direction=float(rng.uniform(-np.pi,np.pi));offset=r*np.array([np.cos(direction),np.sin(direction)])
                orientation_label=0;reason='correct'
                if category==3:offset=rng.uniform(-.055,.055,2);reason='varied_placement'
                if category==4:offset=rng.choice([-1.,1.])*np.array([rng.uniform(.034,.060),rng.uniform(-.01,.01)]);reason='outside'
                if category==5:angle=180+rng.uniform(-8,8);orientation_label=1;reason='wrong180'
                if category==6:angle=float(rng.choice([90,270])+rng.uniform(-8,8));orientation_label=1;reason='wrong90'
                if category==7:angle=float(rng.choice([-1,1])*rng.uniform(17,70));orientation_label=1;reason='wrong_angle'
                if category==8:roll=float(rng.choice([-1,1])*rng.uniform(35,70));orientation_label=1;reason='tilted'
                if category==9:orientation_label=2;reason='occluded'
                if category==10:reason='partial_readable'
                if category==11:r=float(rng.uniform(.012,.018));offset=r*np.array([np.cos(direction),np.sin(direction)]);reason='position_boundary'
                xy=target+offset;yaw=targetyaw+np.deg2rad(angle)
                quat=Rotation.from_euler('xyz',[np.deg2rad(roll),np.deg2rad(pitch),yaw]).as_quat()[[3,0,1,2]]
                z=cfg['workspace']['table_z_m']+cfg['object']['height_m']/2+.0002
                sim.data.set_joint_qpos(env.env.cube.joints[0],np.r_[xy,z,quat]);sim.data.set_joint_qvel(env.env.cube.joints[0],np.zeros(6))
                sim.data.mocap_pos[stageid]=[*target,.8];sim.data.mocap_quat[stageid]=[np.cos(targetyaw/2),0,0,np.sin(targetyaw/2)]
                sim.model.cam_fovy[camid]=float(rng.choice(fovs));sim.model.cam_pos[camid]=[rng.uniform(-.025,.025),rng.uniform(-.025,.025),2.2]
                sim.forward();rgb=env.frame()
                # Photometric changes use independent RNG draws per split.
                gain=rng.uniform(.80,1.16);bias=rng.uniform(-8,8)
                rgb=np.clip(rgb.astype(np.float32)*gain+bias+rng.normal(0,.5,rgb.shape),0,255).astype(np.uint8)
                if category in (9,10):
                    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);mask=cv2.inRange(hsv,(88,70,25),(137,255,255));ys,xs=np.where(mask>0)
                    if len(xs):
                        bx,by,bw,bh=int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)
                        if category==9:
                            side=int(rng.integers(0,4));fraction=float(rng.uniform(.4,.85));color=tuple(int(v) for v in rng.integers(45,190,3))
                            if side==0:box=(bx-3,by-3,bx+max(1,int(bw*fraction)),by+bh+3)
                            elif side==1:box=(bx+int(bw*(1-fraction)),by-3,bx+bw+3,by+bh+3)
                            elif side==2:box=(bx-3,by-3,bx+bw+3,by+max(1,int(bh*fraction)))
                            else:box=(bx-3,by+int(bh*(1-fraction)),bx+bw+3,by+bh+3)
                            cv2.rectangle(rgb,box[:2],box[2:],color,-1)
                        else:cv2.rectangle(rgb,(bx-8,by-7),(bx-4,by-3),(105,108,111),-1)
                name=f'{split}-g{geom}-{i:04d}.png';path=OUT/'images'/name
                cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                f,info=rgb_features(rgb,targetyaw)
                effective_label=orientation_label
                featrow=-1
                if f is not None:featrow=len(features);features.append(f);labels.append(effective_label);splits.append(split)
                distance=float(np.linalg.norm(offset));yawerror=abs(float(np.arctan2(np.sin(np.deg2rad(angle)),np.cos(np.deg2rad(angle)))))
                truth_visual=distance<=.015 and yawerror<=np.deg2rad(10) and abs(roll)<=10
                context={'task_family':'feeding','target_yaw_rad':targetyaw,'position_tolerance_m':.015,'target_region_size_m':.1,'object_height_m':cfg['object']['height_m']}
                rows.append({'id':name,'split':split,'seed':SEEDS[split]+geom*100000,'config_group':f'g{geom}-{split}',
                    'sha256':digest(path),'image':str(path),'feature_row':featrow,'orientation_label':effective_label,
                    'category':reason,'geometry':cfg['object'],'fovy_deg':float(sim.model.cam_fovy[camid]),'camera_position_m':sim.model.cam_pos[camid].tolist(),
                    'declared_goal_context':context,'label_truth':{'position_error_m':distance,'yaw_error_rad':yawerror,'roll_deg':roll,'visual_geometry_satisfied':bool(truth_visual),'observability_unknown':category==9},
                    'offline_object_xy_m':xy.tolist(),'offline_target_xy_m':target.tolist()})
                if len(rows)%100==0:print('rendered',len(rows),'elapsed',round(time.monotonic()-start,1),flush=True)
        env.close()
    np.savez_compressed(OUT/'features.npz',x=np.asarray(features,np.float32),y=np.asarray(labels,np.int64),split=np.asarray(splits))
    (OUT/'data_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    localize_occlusion_labels()
    print('RENDERING_FINISHED',len(rows),round(time.monotonic()-start,1),flush=True)



def localize_occlusion_labels():
    """Rebuild only independently declared occluders from object-local blue pixels.

    Retain the original generation manifest for audit; split RNGs are independent.
    """
    out=OUT;original=[json.loads(l) for l in (out/'data_manifest.jsonl').read_text().splitlines()]
    (out/'original_data_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in original))
    rows=[r for r in original if r['category']!='occluded'];x=[];y=[];splits=[];aug=[];rngs={k:np.random.default_rng(v) for k,v in {'train':810929,'validation':820929,'test':830929}.items()}
    for row in rows.copy():
     rgb=cv2.cvtColor(cv2.imread(row['image']),cv2.COLOR_BGR2RGB);angle=row['declared_goal_context']['target_yaw_rad'];f,info=rgb_features(rgb,angle)
     if f is not None:row['feature_row']=len(x);x.append(f);y.append(row['orientation_label']);splits.append(row['split'])
     if row['orientation_label']!=0 or row['category']!='correct':continue
     hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);mask=cv2.inRange(hsv,(88,70,25),(137,255,255));ox,oy,ow,oh=info['object_bbox_xywh'];local=np.zeros_like(mask);local[oy:oy+oh,ox:ox+ow]=mask[oy:oy+oh,ox:ox+ow];ys,xs=np.where(local>0)
     if len(xs)<8:continue
     bx,by,bw,bh=int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1);rng=rngs[row['split']]
     for index in range(4 if row['split']=='train' else 1):
      image=rgb.copy();side=int(rng.integers(4));fraction=float(rng.uniform(.4,.90));gray=int(rng.integers(55,180));color=(gray,gray,gray)
      if side==0:box=(bx-3,by-3,bx+max(1,int(bw*fraction)),by+bh+3)
      elif side==1:box=(bx+int(bw*(1-fraction)),by-3,bx+bw+3,by+bh+3)
      elif side==2:box=(bx-3,by-3,bx+bw+3,by+max(1,int(bh*fraction)))
      else:box=(bx-3,by+int(bh*(1-fraction)),bx+bw+3,by+bh+3)
      cv2.rectangle(image,box[:2],box[2:],color,-1);f,_=rgb_features(image,angle)
      name=row['id'].replace('.png',f'-occlusion2-{index}.png');path=out/'images'/name;cv2.imwrite(str(path),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
      nr=dict(row,id=name,image=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),orientation_label=2,category='occluded',augmentation={'source':row['id'],'box_xyxy':list(box),'color':color,'scope':'blue glyph localized within orange bounding box'})
      nr['label_truth']=dict(row['label_truth'],observability_unknown=True);nr['feature_row']=-1
      if f is not None:nr['feature_row']=len(x);x.append(f);y.append(2);splits.append(row['split'])
      rows.append(nr);aug.append(nr['augmentation'])
    np.savez_compressed(out/'features.npz',x=np.asarray(x,np.float32),y=np.asarray(y,np.int64),split=np.asarray(splits));(out/'data_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(out/'augmentation_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in aug));print('fixed',len(rows),'features',len(x))

def fit():
    import torch
    torch.set_num_threads(4);torch.manual_seed(401091);np.random.seed(401091)
    data=np.load(OUT/'features.npz');x=data['x'];y=data['y'];split=data['split']
    train=split=='train';val=split=='validation';mean=x[train].mean(0);scale=x[train].std(0).clip(.08)
    xt=torch.from_numpy((x-mean)/scale);yt=torch.from_numpy(y)
    net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.ReLU(),torch.nn.Linear(64,3))
    opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.003)
    counts=np.bincount(y[train],minlength=3);classweights=torch.tensor(np.sqrt(counts.sum()/counts),dtype=torch.float32)
    best=None;bestloss=float('inf');history=[]
    for epoch in range(260):
        net.train();indices=np.random.permutation(np.where(train)[0])
        for j in range(0,len(indices),128):
            ids=indices[j:j+128];opt.zero_grad();loss=torch.nn.functional.cross_entropy(net(xt[ids]),yt[ids],weight=classweights);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():vloss=float(torch.nn.functional.cross_entropy(net(xt[val]),yt[val]))
        if vloss<bestloss:bestloss=vloss;best={k:v.detach().clone() for k,v in net.state_dict().items()};bestepoch=epoch
        if epoch%40==0:print('epoch',epoch,'validation_loss',round(vloss,5),flush=True)
        history.append({'epoch':epoch,'validation_loss':vloss})
    net.load_state_dict(best);net.eval()
    with torch.no_grad():p=torch.softmax(net(xt),1).numpy()
    # Predeclared grid; select smallest threshold with no wrong/occluded validation accepted upright.
    grid=[.80,.85,.90,.93,.95,.97,.98,.99,.995,.999]
    threshold=None
    for t in grid:
        accepted=(p[val].argmax(1)==0)&(p[val].max(1)>=t)
        if not np.any(accepted & (y[val]!=0)) and np.sum(accepted & (y[val]==0))>=5:threshold=t;break
    if threshold is None:raise RuntimeError('No usable validation abstention threshold; do not publish model')
    np.savez_compressed(OUT/'model.npz',mean=mean,scale=scale,w1=best['0.weight'].numpy().T,b1=best['0.bias'].numpy(),w2=best['2.weight'].numpy().T,b2=best['2.bias'].numpy(),threshold=np.array(threshold))
    dump(OUT/'training.json',{'architecture':f'{x.shape[1]} RGB mask/statistic features ->64 ReLU->3 classes','parameter_count':sum(v.numel() for v in net.parameters()),'train_seed':401091,'best_validation_epoch':bestepoch,'validation_loss':bestloss,'threshold':threshold,'threshold_rule':'smallest predeclared grid threshold with zero wrong/unknown-to-upright validation errors and >=5 upright accepted','threshold_grid':grid,'train_counts':counts.tolist(),'history':history,'cpu_only':True})
    print('MODEL_READY',OUT/'model.npz','threshold',threshold,flush=True)
    evaluate()


def evaluate():
    model=GoalSuccessClassifier(OUT/'model.npz');rows=[json.loads(line) for line in (OUT/'data_manifest.jsonl').read_text().splitlines()]
    report={'model_sha256':model.model_sha256,'threshold':model.threshold,'splits':{},'limitations':['Synthetic native overhead DUPLO v2 only; no real-camera calibration claim.','No current pose/contact truth at inference.','Single-view image cannot establish physical release/contact or angular/3D stability.','Known gray partial occluders; arbitrary out-of-distribution occlusion may fail.','Near visual position boundary abstains. Loading returns unknown.']}
    predictions=[];selected=[]
    for split in ['validation','test']:
        matrix={label:{p:0 for p in ['complete','incomplete','unknown']} for label in ['complete','incomplete','unknown']};cat={}
        for row in [r for r in rows if r['split']==split]:
            rgb=cv2.cvtColor(cv2.imread(row['image']),cv2.COLOR_BGR2RGB);r=model(rgb,row['declared_goal_context']);truth=row['label_truth']
            label='unknown' if truth['observability_unknown'] else ('complete' if truth['visual_geometry_satisfied'] else 'incomplete')
            matrix[label][r['status']]+=1;cat.setdefault(row['category'],{});key=f'{label}->{r["status"]}';cat[row['category']][key]=cat[row['category']].get(key,0)+1
            predictions.append({'id':row['id'],'split':split,'label':label,'prediction':r['status'],'reason':r['reason'],'score':r['score'],'position_error_rgb_m':r['evidence'].get('position_error_m'),'truth_position_error_m':truth['position_error_m']})
            if split=='test' and len([a for a in selected if a[0]['category']==row['category']])<2:selected.append((row,r,rgb))
        report['splits'][split]={'count':sum(sum(v.values()) for v in matrix.values()),'confusion':matrix,'by_category':cat}
    tilew=256;tileh=285;grid=np.full((6*tileh,4*tilew,3),242,np.uint8)
    for i,(row,r,rgb) in enumerate(selected[:24]):
        tile=cv2.resize(rgb,(256,256));y=(i//4)*tileh;x=(i%4)*tilew;grid[y:y+256,x:x+256]=tile
        cv2.putText(grid,f'{row["category"]}: {r["status"]}',(x+4,y+274),cv2.FONT_HERSHEY_SIMPLEX,.39,(10,10,10),1,cv2.LINE_AA)
    cv2.imwrite(str(OUT/'heldout-grid.png'),cv2.cvtColor(grid,cv2.COLOR_RGB2BGR))
    dump(OUT/'evaluation.json',report);(OUT/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    for rgb in [np.zeros((512,512,3),np.uint8),np.full((512,512,3),120,np.uint8)]:assert model(rgb,{'task_family':'feeding'})['status']=='unknown'
    assert model(np.zeros((512,512,3),np.uint8),{'task_family':'loading'})['status']=='unknown'
    dump(OUT/'self_check.json',{'pass':True,'checks':['blank images abstain','unsupported loading abstains','CPU numpy artifact reload','no input pose/contact/state fields'], 'model_sha256':model.model_sha256})
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS']
    (OUT/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(OUT)}\n' for p in sorted(paths)))
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','fit','evaluate']);args=p.parse_args();globals()[args.action]()
