"""Test training-only transition negatives on frozen phone-image descriptors."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,collections,json,hashlib,time,shutil
from pathlib import Path
import numpy as np
import torch
import video03_held_probe as base
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load_events(selection,labels,rows):
    byid={v['review_id']:v for v in labels['samples']};events=[]
    for s in selection:
        if 'context'not in s:continue
        label=byid[s['review_id']]['held_label']
        for c in s['context']:
            if c['delta_requested_s']!=0:continue
            i=c['row_index'];r=rows[i];assert c['sample_id']==r['sample_id']
            events.append({'review_id':s['review_id'],'row_index':i,'sample_id':r['sample_id'],'demo_id':r['demo_id'],'view':r['view'],'label':-1 if label is None else label,'role':s.get('role','reused_validation')})
    return events
def main(a):
    start=time.monotonic();cfg=json.loads(a.protocol.read_text());d=Path(cfg['input_data']);out=a.output
    out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir()
    doc=ROOT/'experiment-history/2026-10-12-video04';prior=ROOT/'experiment-history/2026-10-12-video03'
    assert sha(d/'source-index.json')==cfg['source_index_sha256']
    rows=json.loads((d/'source-index.json').read_text())['samples'];ff=np.load(d/'features/features.npz')
    assert list(ff['sample_ids'])==[r['sample_id']for r in rows]
    for r in rows:assert sha(d/r['frame_path'])==r['frame_sha256']
    review=json.loads((prior/'inputs/held_review.json').read_text())['demonstrations']
    y=np.array([base.label_at(review[r['demo_id']],r['clip_time_s'],cfg['label_boundary_margin_s'])for r in rows],np.int8)
    visual=np.concatenate([ff['full_rgb'],ff['current_observed_object_square_crop']],1)
    kin,mean,delta,hist=base.features(rows,visual,cfg['history_s'],cfg['maximum_gap_s'])
    xs={'full_crop_current':visual,'fusion_current':np.concatenate([visual,kin],1),'fusion_causal_history':np.concatenate([visual,mean,delta,kin],1)}
    events=load_events(json.loads((doc/'selection.json').read_text()),json.loads((doc/'labels.json').read_text()),rows)
    oldevents=load_events(json.loads((prior/'challenge-selection.json').read_text()),json.loads((prior/'blinded-challenge-labels.json').read_text()),rows)
    hard={e['row_index']for e in events if e['role']=='train'and e['label']==0}
    assert hard and all(rows[i]['demo_id']in cfg['train_takes']and y[i]==-1 for i in hard)
    updated_y=y.copy()
    for i in hard:updated_y[i]=0
    train=[i for i,r in enumerate(rows)if r['demo_id']in cfg['train_takes']and updated_y[i]>=0]
    assert set(cfg['train_takes']).isdisjoint(cfg['validation_takes']+cfg['extra_validation_takes'])
    assert not(set(train)&{e['row_index']for e in events if e['role']!='train'})
    original_weight=base.weights
    def balanced_weight(rows_,ids,labels):
        ww=original_weight(rows_,ids,labels);cells=collections.defaultdict(list)
        for j,i in enumerate(ids):cells[(rows_[i]['demo_id'],int(labels[j]),rows_[i]['view'])].append(j)
        for (did,label,view),jj in cells.items():
            if label!=0:continue
            groups=[[j for j in jj if ids[j]not in hard],[j for j in jj if ids[j]in hard]]
            if all(groups):
                total=float(ww[jj].sum())
                for group in groups:ww[group]=total/(2*len(group))
        assert abs(ww.sum()-1)<1e-5
        return ww
    torch.manual_seed(cfg['seed']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    predictions={};fitinfo=[];independent_errors={};weightlog=[]
    for method,x in xs.items():
        saved=prior/'models'/('both_views__'+method+'.npz')
        if not saved.exists():saved=Path('/var/tmp/typhon-video-held-20261012/probe-v1/models')/saved.name
        old=np.load(saved);z=(x-old['mean'])/old['std'];scores=1/(1+np.exp(-np.clip(z@old['coef']+old['bias'],-80,80)))
        predictions['frozen03__'+method]=scores
        shutil.copy2(saved,out/'models'/('frozen03__'+method+'.npz'))
    for arm in cfg['training_arms']:
        base.weights=original_weight if arm=='added_negatives_original_weighting'else balanced_weight
        ww=base.weights(rows,train,updated_y[train])
        weightlog.append({'arm':arm,'sum':float(ww.sum()),'new_hard_negative_total':float(sum(ww[j]for j,i in enumerate(train)if i in hard)),'train_ids':train,'weights':ww.tolist(),'train_labels':updated_y[train].tolist()})
        for method,x in xs.items():
            name=arm+'__'+method
            scores,fit=base.fit_predict(x,train,updated_y[train],rows,cfg,out/'models',name)
            predictions[name]=scores;fitinfo.append(fit)
            model=np.load(out/'models'/(name+'.npz'))
            independent=1/(1+np.exp(-np.clip(((x-model['mean'])/model['std'])@model['coef']+model['bias'],-80,80)))
            error=float(np.abs(independent-scores).max());assert error<1e-5;independent_errors[name]=error
            assert np.array_equal(model['train_ids'],train)and np.array_equal(model['train_labels'],updated_y[train])
            assert np.array_equal(model['mean'],x[train].mean(0))
        print('FITTED',arm,flush=True)
    base.weights=original_weight
    scopes={}
    for name,takes in [('old_validation_interiors',cfg['validation_takes']),('extra_takes_interiors',cfg['extra_validation_takes'])]:
        ids=[i for i,r in enumerate(rows)if r['demo_id']in takes];scopes[name]=(ids,y[ids])
    for name,ee in [('train_transition_diagnostic',[e for e in events if e['role']=='train']),('extra_transition_panel',[e for e in events if e['role']=='extra_validation']),('reused03_transition_panel',oldevents)]:
        ids=[e['row_index']for e in ee];scopes[name]=(ids,np.array([e['label']for e in ee],np.int8))
    metrics=[];per_event=[]
    for name,pred in predictions.items():
        for scope,(ids,labels)in scopes.items():
            m=base.metrics(labels,pred[ids],cfg['threshold']);metrics.append({'model':name,'scope':scope,'unknown':int(sum(labels<0)),**m})
        for e in events+oldevents:per_event.append({'model':name,**e,'score':float(pred[e['row_index']]),'predicted_held':bool(pred[e['row_index']]>=cfg['threshold'])})
    np.savez_compressed(out/'predictions.npz',sample_ids=ff['sample_ids'],original_labels=y,updated_labels=updated_y,**predictions)
    for n,v in [('metrics.json',metrics),('event-scores.json',per_event),('fits.json',fitinfo),('training-weights.json',weightlog),('protocol.json',cfg)]:save(out/n,v)
    counts=collections.Counter((e['role'],e['label'])for e in events)
    summary={'fits':len(fitinfo),'source_frames':len(rows),'new_negative_frame_views':len(hard),'new_negative_events':len({e['review_id']for e in events if e['role']=='train'and e['label']==0}),'old_train_count':len(train)-len(hard),'new_train_count':len(train),'event_counts':{str(k):v for k,v in counts.items()},'max_independent_reload_error':max(independent_errors.values()),'seconds':time.monotonic()-start,'control_authority':False,'VLA_motor_weights_changed':False,'source_hashes':{str(p):sha(p)for p in [a.protocol,Path(__file__),doc/'selection.json',doc/'labels.json',d/'source-index.json',d/'features/features.npz']}}
    save(out/'summary.json',summary)
    save(out/'validation.json',{'train_takes_disjoint':True,'new_labels_only_training':True,'source_frames_checked':len(rows),'scalers_fit_on_training_only':True,'independent_probability_errors':independent_errors,'old_unknowns_stay_unknown_except_reviewed_train_negatives':True,'no_future_context_in_features':True})
    print(json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
