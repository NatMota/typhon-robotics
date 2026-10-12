"""Causal held-state probes on unmarked personal footage; annotation agreement only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,collections,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def label_at(review,t,margin):
    found=[]
    for value,key in [(1,'held_intervals_clip_s'),(0,'not_held_intervals_clip_s')]:
        for a,b in review[key]:
            if a+margin<=t<=b-margin:found.append(value)
    assert len(set(found))<=1
    return found[0] if found else -1

def features(rows,visual,history=.5,max_gap=.25):
    """All histories end at the current sample; no interpolation across gaps."""
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[(r['demo_id'],r['view'])].append(i)
    basic=np.zeros((len(rows),21),np.float32)
    hist_mean=np.zeros_like(visual);hist_delta=np.zeros_like(visual);history_ids=[]
    for ids in groups.values():
        last_gap=0
        for j,i in enumerate(ids):
            r=rows[i];t=r['clip_time_s'];xy=r['xy_normalized'];hasxy=r['valid_xy']
            if j and t-rows[ids[j-1]]['clip_time_s']>max_gap:last_gap=j
            past=[k for k in ids[last_gap:j+1] if t-rows[k]['clip_time_s']<=history+1e-8]
            assert past and all(rows[k]['clip_time_s']<=t for k in past)
            hist_mean[i]=visual[past].mean(0);hist_delta[i]=visual[i]-visual[past[0]]
            history_ids.append({'sample_id':r['sample_id'],'input_ids':[rows[k]['sample_id']for k in past]})
            hand=None
            if r['hands']:
                hand=min(r['hands'],key=lambda h:np.linalg.norm(np.array(h['pinch_midpoint_image_xy'])-xy)) if hasxy else r['hands'][0]
            rel=np.array(hand['pinch_midpoint_image_xy'])-xy if hand is not None and hasxy else np.zeros(2)
            yaw=r['yaw_clockwise_deg'];wh=np.zeros(2) if not hasxy else np.array(r['bbox_xywh'][2:])/r['image_wh']
            motion=np.zeros(2);angle_speed=0.;motion_valid=0.
            # Motion requires every intervening object observation, not just two visible endpoints.
            if len(past)>1 and all(rows[k]['valid_xy'] for k in past):
                dt=t-rows[past[0]]['clip_time_s']
                if dt>0:
                    motion=(np.array(xy)-rows[past[0]]['xy_normalized'])/dt;motion_valid=1.
                    y0=rows[past[0]]['yaw_clockwise_deg']
                    if yaw is not None and y0 is not None:angle_speed=np.deg2rad((yaw-y0+180)%360-180)/dt
            basic[i]=[bool(hand),hasxy,r['crop_xyxy']is not None,yaw is not None,
                *rel,float(np.linalg.norm(rel))if hand is not None and hasxy else 0.,
                hand['aperture_diagonal_fraction']if hand is not None else 0.,
                *wh,*(xy if hasxy else [0.,0.]),
                np.sin(np.deg2rad(yaw))if yaw is not None else 0.,np.cos(np.deg2rad(yaw))if yaw is not None else 0.,
                len(r['hands']),*motion,angle_speed,motion_valid,float(len(past)>1),t-rows[past[0]]['clip_time_s']]
    return basic,hist_mean,hist_delta,history_ids

def weights(rows,ids,y):
    cells=collections.defaultdict(list)
    for j,i in enumerate(ids):cells[(rows[i]['demo_id'],int(y[j]))].append(j)
    take_count=len({d for d,c in cells})
    w=np.zeros(len(ids),np.float32)
    for (d,c),jj in cells.items():
        classes=len({cc for dd,cc in cells if dd==d})
        views=collections.defaultdict(list)
        for j in jj:views[rows[ids[j]]['view']].append(j)
        for v,kk in views.items():w[kk]=1/(take_count*classes*len(views)*len(kk))
    assert abs(w.sum()-1)<1e-5
    return w

def fit_predict(x,ids,y,rows,cfg,out,name):
    train=x[ids];mean=train.mean(0);std=np.maximum(train.std(0),cfg['train_feature_standard_deviation_floor'])
    z=((x-mean)/std).astype(np.float32)
    xt=torch.tensor(z[ids],device='cuda');yt=torch.tensor(y,dtype=torch.float32,device='cuda')
    wt=torch.tensor(weights(rows,ids,y),device='cuda')
    coef=torch.zeros(x.shape[1],device='cuda',requires_grad=True);bias=torch.zeros((),device='cuda',requires_grad=True)
    optimizer=torch.optim.LBFGS([coef,bias],max_iter=cfg['max_iterations'],line_search_fn='strong_wolfe',tolerance_grad=1e-7,tolerance_change=1e-9)
    calls=0
    def closure():
        nonlocal calls
        optimizer.zero_grad()
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(xt@coef+bias,yt,reduction='none')*wt).sum()+cfg['logistic_l2']*.5*coef.square().sum()
        assert torch.isfinite(loss)
        loss.backward();calls+=1
        return loss
    t=time.monotonic();optimizer.step(closure)
    with torch.no_grad():pred=torch.sigmoid(torch.tensor(z,device='cuda')@coef+bias).cpu().numpy()
    assert np.isfinite(pred).all()
    np.savez_compressed(out/(name+'.npz'),coef=coef.detach().cpu().numpy(),bias=bias.detach().cpu().numpy(),mean=mean,std=std,train_ids=np.array(ids),train_labels=y)
    return pred,{'model':name,'dimension':x.shape[1],'closure_calls':calls,'seconds':time.monotonic()-t,'train_samples':len(ids),'train_takes':sorted({rows[i]['demo_id']for i in ids}),'train_views':sorted({rows[i]['view']for i in ids})}

def metrics(y,p,threshold=.5):
    known=y>=0;y=y[known];p=p[known]>=threshold
    tp=int(((y==1)&p).sum());fp=int(((y==0)&p).sum());tn=int(((y==0)&~p).sum());fn=int(((y==1)&~p).sum())
    f1p=2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else None
    f1n=2*tn/(2*tn+fp+fn)if 2*tn+fp+fn else None
    return {'known':len(y),'positive':int(sum(y==1)),'negative':int(sum(y==0)),'tp':tp,'fp':fp,'tn':tn,'fn':fn,
      'precision':tp/(tp+fp)if tp+fp else None,'recall':tp/(tp+fn)if tp+fn else None,
      'f1':f1p,'macro_f1':(f1p+f1n)/2 if f1p is not None and f1n is not None else None}

def main(a):
    t0=time.monotonic();cfg=json.loads(a.protocol.read_text());d=Path(cfg['input_data'])
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'models').mkdir()
    idx=json.loads((d/'source-index.json').read_text());rows=idx['samples'];assert sha(d/'source-index.json')==cfg['source_index_sha256']
    inp=ROOT/'experiment-history/2026-10-12-video03/inputs'
    assert sha(inp/'held_review.json')==cfg['held_review_sha256'] and sha(inp/'manifest.json')==cfg['held_manifest_sha256']
    review=json.loads((inp/'held_review.json').read_text());manifest=json.loads((inp/'manifest.json').read_text())
    assert json.loads((d/'sequential-decode-verification.json').read_text())['mismatch_count']==0
    original={r['recording_id']:r for r in json.loads((d/'input-manifest.json').read_text())['raw']}
    byview={(v['demo_id'],v['view']):v for v in manifest['videos']}
    for r in rows:
        source=byview[(r['demo_id'],r['view'])]['source_recording']
        assert source['recording_id']==r['recording_id'] and source['source_sha256']==original[r['recording_id']]['sha256']
        assert abs(source['start_s']-r['source_start_s'])<1e-7 and abs(source['end_s']-r['source_end_s'])<1e-7
        assert sha(d/r['frame_path'])==r['frame_sha256']
        assert r['demo_id']!='demo_0017'
    ff=np.load(d/'features/features.npz');assert list(ff['sample_ids'])==[r['sample_id']for r in rows]
    full=ff['full_rgb'];crop=ff['current_observed_object_square_crop'];visual=np.concatenate([full,crop],1)
    kin,mean,delta,hist=features(rows,visual,cfg['history_s'],cfg['maximum_gap_s'])
    # Prefix causality regression: every group's output at its midpoint must reproduce with the future removed.
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[(r['demo_id'],r['view'])].append(i)
    for ids in groups.values():
        cut=ids[:len(ids)//2+1];k,m,de,_=features([rows[i]for i in cut],visual[cut],cfg['history_s'],cfg['maximum_gap_s'])
        assert np.array_equal(k,kin[cut])and np.array_equal(m,mean[cut])and np.array_equal(de,delta[cut])
    y=np.array([label_at(review['demonstrations'][r['demo_id']],r['clip_time_s'],cfg['label_boundary_margin_s'])for r in rows],np.int8)
    y0=np.array([label_at(review['demonstrations'][r['demo_id']],r['clip_time_s'],0)for r in rows],np.int8)
    arrays={'full_current':full,'full_crop_current':visual,'hand_object':kin,
      'fusion_current':np.concatenate([visual,kin],1),'fusion_causal_history':np.concatenate([visual,mean,delta,kin],1),
      'clock_only':np.array([[r['clip_time_s'],r['clip_time_s']**2]for r in rows],np.float32)}
    train_ids=[i for i,r in enumerate(rows)if r['demo_id']in cfg['train_takes']and y[i]>=0]
    panels={'both_views':train_ids,'train_primary_only':[i for i in train_ids if rows[i]['view']=='camera_a'],
      'train_secondary_only':[i for i in train_ids if rows[i]['view']=='camera_b'],
      'train_normal_session_only':[i for i in train_ids if rows[i]['demo_id']in cfg['train_takes'][:4]],
      'train_awkward_session_only':[i for i in train_ids if rows[i]['demo_id']in cfg['train_takes'][4:]]}
    assert not(set(cfg['train_takes'])&set(cfg['validation_takes']))
    torch.manual_seed(cfg['seed']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    predictions={};fits=[];rng=np.random.default_rng(cfg['seed'])
    for panel,ids in panels.items():
        for method in cfg['models']:
            x=arrays['fusion_causal_history'if method=='shuffled_fusion'else method]
            yy=y[ids].copy()
            if method=='shuffled_fusion':
                for did,view in groups:
                    jj=[j for j,i in enumerate(ids)if rows[i]['demo_id']==did and rows[i]['view']==view]
                    if jj:yy[jj]=rng.permutation(yy[jj])
            name=panel+'__'+method
            pred,fit=fit_predict(x,ids,yy,rows,cfg,a.output/'models',name);predictions[name]=pred;fits.append(fit)
        print('FITTED',panel,flush=True)
    predictions.update(always_not_held=np.zeros(len(rows)),hand_present=kin[:,0],near_hand=(kin[:,0]>0)&(kin[:,1]>0)&(kin[:,6]<.08))
    scopes={'P_validation':[i for i,r in enumerate(rows)if r['demo_id']in cfg['validation_takes']],
      'normal_validation':[i for i,r in enumerate(rows)if r['demo_id']=='demo_0006'],
      'awkward_session_validation':[i for i,r in enumerate(rows)if r['demo_id']in ['demo_0014','demo_0016']],
      'other_observation_only':[i for i,r in enumerate(rows)if r['split']=='other_observation_only'],
      'corrections':[i for i,r in enumerate(rows)if r['manual_category']=='orientation_correction'],
      'already_completed':[i for i,r in enumerate(rows)if r['manual_category']=='already_completed']}
    for did,view in groups:scopes[did+'/'+view]=groups[(did,view)]
    results=[]
    for name,p in predictions.items():
        for scope,ids in scopes.items():
            for view in ['both','camera_a','camera_b']:
                ii=[i for i in ids if view=='both'or rows[i]['view']==view]
                if not ii:continue
                m=metrics(y[ii],p[ii],cfg['threshold']);m.update(model=name,scope=scope,view=view,
                  unknown=int(sum(y[ii]<0)),unknown_predicted_held=int(sum((y[ii]<0)&(p[ii]>=cfg['threshold']))))
                results.append(m)
    save(a.output/'metrics.json',results);save(a.output/'fits.json',fits);save(a.output/'protocol.json',cfg)
    np.savez_compressed(a.output/'predictions.npz',sample_ids=ff['sample_ids'],labels=y,labels_without_margin=y0,**predictions)
    (a.output/'samples.jsonl').write_text(''.join(json.dumps({'sample_id':r['sample_id'],'demo_id':r['demo_id'],'view':r['view'],
      'recording_id':r['recording_id'],'source_time_s':r['source_time_s'],'clip_time_s':r['clip_time_s'],
      'frame_path':r['frame_path'],'frame_sha256':r['frame_sha256'],'split':r['split'],'label':int(y[i]),
      'label_without_margin':int(y0[i]),'label_evidence_id':review['demonstrations'][r['demo_id']]['evidence_id']})+'\n'for i,r in enumerate(rows)))
    (a.output/'causal-history-inputs.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in hist))
    selected=[x for x in results if x['scope']=='P_validation'and x['view']=='both'and(x['model'].startswith('both_views')or'__'not in x['model'])]
    counts={split:dict(collections.Counter(int(y[i])for i,r in enumerate(rows)if r['split']==split))for split in sorted({r['split']for r in rows})}
    summary={'samples':len(rows),'known_after_margin':int(sum(y>=0)),'held_after_margin':int(sum(y==1)),'counts_by_original_split':counts,
      'known_before_margin':int(sum(y0>=0)),'fits':len(fits),'primary_validation':selected,
      'wall_seconds':time.monotonic()-t0,'causal_prefix_tests':len(groups),'all_source_frame_hashes_verified':len(rows),
      'no_green_marker_or_action_labels':True,'unknown_excluded_from_fitting_and_scoring':True,
      'robot_control_qualified':False,'label_provenance':review['reviewer'],'input_manifest':{str(p):sha(p)for p in
        [d/'source-index.json',d/'features/features.npz',inp/'held_review.json',inp/'manifest.json',a.protocol,Path(__file__)]}}
    save(a.output/'summary.json',summary);print(json.dumps({'done':True,'known':int(sum(y>=0)),'fits':len(fits),'seconds':summary['wall_seconds']}),flush=True)

if __name__=='__main__':
    assert label_at({'held_intervals_clip_s':[[2,4]],'not_held_intervals_clip_s':[[0,1]]},1.5,.25)==-1
    assert label_at({'held_intervals_clip_s':[[2,4]],'not_held_intervals_clip_s':[[0,1]]},2.1,.25)==-1
    assert label_at({'held_intervals_clip_s':[[2,4]],'not_held_intervals_clip_s':[[0,1]]},3,.25)==1
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
