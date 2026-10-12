"""Dense phone QA and temporal diagnostics. No GE labels, model fit or robot actions."""
import argparse,collections,csv,hashlib,json,math,subprocess
from pathlib import Path
import cv2,numpy as np
import imageio_ffmpeg

def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def circ_diff(a,b):return float((a-b+180)%360-180)
def valid_pairs(rr):
    return [(a,b)for a,b in zip(rr,rr[1:])if a['valid_xy']and b['valid_xy']and 0<b['clip_time_s']-a['clip_time_s']<=.25]
def distance(a,b):return math.dist(a['xy_normalized'],b['xy_normalized'])
def unit(x):return x/max(float(np.linalg.norm(x)),1e-8)
def angular_extent(v):
    if not v:return None
    a=np.sort(np.array(v)%360);return float(360-max(np.diff(np.r_[a,a[0]+360])))
def angle_average(v):
    if not v:return None
    z=np.exp(1j*np.deg2rad(v)).mean()
    return None if abs(z)<.25 else float(np.rad2deg(np.angle(z))%360)
def anchors(rr,start,stop,count):
    times=np.array([r['clip_time_s']for r in rr])
    selected=[int(np.abs(times-t).argmin())for t in np.linspace(start,stop,count)]
    if len(set(selected))!=count:return None
    if any(abs(times[i]-t)>.075 for i,t in zip(selected,np.linspace(start,stop,count))):return None
    return [rr[i]for i in selected]
def tile(im,size=(400,260)):
    w,h=size;s=min(w/im.shape[1],h/im.shape[0]);res=cv2.resize(im,(round(im.shape[1]*s),round(im.shape[0]*s)));out=np.full((h,w,3),18,np.uint8);y=(h-res.shape[0])//2;x=(w-res.shape[1])//2;out[y:y+res.shape[0],x:x+res.shape[1]]=res;return out

def annotated(data,r):
    im=cv2.imread(str(data/r['frame_path']))
    if r['valid_xy']and r['bbox_xywh']:
        x,y,w,h=r['bbox_xywh'];cv2.rectangle(im,(x,y),(x+w,y+h),(20,210,245),2)
    for hand in r['hands']:
        arr=np.array(hand['image_landmarks_normalized'])
        for q in arr:cv2.circle(im,(int(q[0]*im.shape[1]),int(q[1]*im.shape[0])),2,(255,190,80),-1)
    shown=tile(im,size=(400,220));im=np.full((260,400,3),18,np.uint8);im[40:]=shown;text=f"{r['demo_id']} {r['view']} t={r['clip_time_s']:.2f}"
    cv2.rectangle(im,(0,0),(400,39),(18,18,18),-1)
    cv2.putText(im,text,(6,16),cv2.FONT_HERSHEY_SIMPLEX,.42,(240,240,240),1)
    cv2.putText(im,f"yaw={r['yaw_clockwise_deg']} / contact UNKNOWN",(6,35),cv2.FONT_HERSHEY_SIMPLEX,.40,(235,235,235),1)
    return im

def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    idx=json.loads((a.data/'source-index.json').read_text());rows=idx['samples'];groups=collections.defaultdict(list)
    for r in rows:groups[(r['demo_id'],r['view'])].append(r)
    config=json.loads((a.data/'protocol.json').read_text());rng=np.random.default_rng(config['seed'])
    stats=[];windows=[];events=[]
    for (did,view),rr in groups.items():
        rr.sort(key=lambda r:r['clip_time_s']);pairs=valid_pairs(rr);start=rr[0]['clip_time_s'];stop=rr[-1]['clip_time_s']
        valid=[r for r in rr if r['valid_xy']];yy=[r['yaw_clockwise_deg']for r in rr if r['yaw_clockwise_deg']is not None]
        initial=[r for r in rr if r['clip_time_s']<=start+.6];final=[r for r in rr if r['clip_time_s']>=stop-.6]
        yi=angle_average([r['yaw_clockwise_deg']for r in initial if r['yaw_clockwise_deg']is not None]);yf=angle_average([r['yaw_clockwise_deg']for r in final if r['yaw_clockwise_deg']is not None])
        ii=[r['xy_normalized']for r in initial if r['valid_xy']];ff=[r['xy_normalized']for r in final if r['valid_xy']]
        xy_i=np.median(ii,axis=0).tolist()if ii else None;xy_f=np.median(ff,axis=0).tolist()if ff else None
        lengths=[]
        for _ in range(100):
            positions=[r['xy_normalized']for r in valid];rng.shuffle(positions);it=iter(positions)
            trial=[dict(r,xy_normalized=next(it))if r['valid_xy']else r for r in rr]
            lengths.append(sum(distance(x,y)for x,y in valid_pairs(trial)))
        co_motion=0;near=0
        for prev,r in pairs:
            # Observation hypothesis only. 2D overlap is not grasp, contact or lift.
            prox=bool(r['hands'])and min(math.dist(h['pinch_midpoint_image_xy'],r['xy_normalized'])for h in r['hands'])<=.08
            near+=int(prox);moving=distance(prev,r)>.003
            co_motion+=int(prox and moving)
        for st in np.arange(0,max(0,stop-4)+1e-7,config['window_stride_s']):
            aa=anchors(rr,float(st),float(st+4),config['anchors'])
            if aa is None:continue
            windows.append({'demo_id':did,'view':view,'split':rr[0]['split'],'manual_category':rr[0]['manual_category'],'window_start_s':float(st),'requested_horizon_s':4,'anchors':[{k:r[k]for k in ['sample_id','source_time_s','clip_time_s','xy_normalized','yaw_clockwise_deg','valid_xy','is_grasped','metric_xyz']}for r in aa],'source_type':'observed human object trajectory; NOT Franka/EEF actions'})
        # Local observable motion segments only. Do not relabel them grasp/lift/release.
        raw_events=[]
        for prev,r in zip(rr,rr[1:]):
            if not prev['valid_xy']or not r['valid_xy']or r['clip_time_s']-prev['clip_time_s']>.25:kind='unknown'
            elif distance(prev,r)>.006:kind='image_translation'
            elif prev['yaw_clockwise_deg']is not None and r['yaw_clockwise_deg']is not None and abs(circ_diff(r['yaw_clockwise_deg'],prev['yaw_clockwise_deg']))>=10:kind='visible_yaw_change'
            else:kind='no_resolved_image_motion'
            if raw_events and raw_events[-1]['kind']==kind:raw_events[-1]['end_s']=r['clip_time_s'];raw_events[-1]['last_sample_id']=r['sample_id']
            else:raw_events.append({'kind':kind,'start_s':prev['clip_time_s'],'end_s':r['clip_time_s'],'first_sample_id':prev['sample_id'],'last_sample_id':r['sample_id']})
        events.extend(dict(x,demo_id=did,view=view,split=rr[0]['split'],is_physical_contact_annotation=False)for x in raw_events)
        stats.append({'demo_id':did,'view':view,'manual_category':rr[0]['manual_category'],'split':rr[0]['split'],'samples':len(rr),'valid_xy':len(valid),'known_yaw':len(yy),'samples_with_hand':sum(bool(r['hands'])for r in rr),'valid_adjacent_edges':len(pairs),'path_length_image_units':sum(distance(x,y)for x,y in pairs),'shuffle_path_length_median':float(np.median(lengths)),'static_path_length':0.,'initial_xy':xy_i,'final_xy':xy_f,'endpoint_displacement':math.dist(xy_i,xy_f)if xy_i and xy_f else None,'observed_yaw_extent_deg':angular_extent(yy),'initial_yaw_deg':yi,'final_yaw_deg':yf,'endpoint_directed_yaw_change_deg':circ_diff(yf,yi)if yi is not None and yf is not None else None,'hand_near_object_edges':near,'hand_near_moving_object_edges':co_motion,'contact_confirmed':None,'metric_depth':None})
    save(a.output/'trajectory-summary.json',{'sequences':stats,'frames':len(rows),'semantic_grasp_labels_generated':0,'robot_actions_generated':0})
    for name,x in [('four-second-anchors.jsonl',windows),('observed-motion-segments.jsonl',events)]:
        (a.output/name).write_text(''.join(json.dumps(r,allow_nan=False)+'\n'for r in x))
    # Paired camera artifacts at matched observation times (known source offset, uncertainty unchanged).
    previews=a.output/'previews';previews.mkdir();sheets=[];preview_meta=[]
    for did in ['demo_0001','demo_0006','demo_0007','demo_0011','demo_0018','demo_0019']:
        aa=groups[(did,'camera_a')];bb=groups[(did,'camera_b')];times=np.array([r['clip_time_s']for r in bb])
        panels=[]
        for r in [aa[0],aa[len(aa)//2],aa[-1]]:panels.append(annotated(a.data,r))
        sheets.append(np.hstack(panels))
        path=previews/(did+'-paired.mp4')
        ff=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','800x260','-r','10','-i','-','-an','-c:v','libx264','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
        deltas=[]
        for r in aa:
            j=int(np.abs(times-r['clip_time_s']).argmin());other=bb[j];deltas.append(abs(other['clip_time_s']-r['clip_time_s']))
            ff.stdin.write(np.hstack([annotated(a.data,r),annotated(a.data,other)]).tobytes())
        ff.stdin.close();assert ff.wait()==0
        preview_meta.append({'demo_id':did,'path':str(path.relative_to(a.output)),'paired_frames':len(aa),'max_sampling_delta_s':max(deltas),'sync_uncertainty_s':aa[0]['sync_uncertainty_s'],'labels':'observed image box/hand keypoints; physical contact unknown','playback':'10Hz nearest-PTS proxy, no audio; raw media preserved'})
    cv2.imwrite(str(a.output/'initial-middle-final.jpg'),np.vstack(sheets),[cv2.IMWRITE_JPEG_QUALITY,88])
    save(a.output/'previews.json',preview_meta)
    if a.features:
        features=np.load(a.features);assert list(features['sample_ids'])==[r['sample_id']for r in rows]
        positions={r['sample_id']:i for i,r in enumerate(rows)};results=[];retrieval=[]
        for method in config['feature_methods']:
            x=features[method];pool={};ordered={};endpoint={};first={}
            for key,rr in groups.items():
                ids=[positions[r['sample_id']]for r in rr];times=np.array([r['clip_time_s']for r in rr]);norm=(times-times[0])/max(times[-1]-times[0],1e-8);f=x[ids]
                bins=[unit(f[(norm>=i/8)&(norm<(i+1)/8+1e-9)].mean(0))for i in range(8)]
                ordered[key]=unit(np.concatenate(bins));pool[key]=unit(f.mean(0));endpoint[key]=unit(np.r_[f[0],f[-1]]);first[key]=f[0]
            for kind,table in [('pooled',pool),('ordered_8_bins',ordered),('endpoints',endpoint),('first_frame',first)]:
                matches=[]
                for d in idx['demonstrations']:
                    did=d['demo_id'];best=max(idx['demonstrations'],key=lambda other:table[(did,'camera_a')]@table[(other['demo_id'],'camera_b')])['demo_id'];matches.append({'query':did,'matched':best,'correct':best==did})
                retrieval.append({'method':method,'summary':kind,'top1':sum(r['correct']for r in matches),'total':len(matches),'matches':matches})
            for d in idx['demonstrations']:
                did=d['demo_id'];aa=groups[(did,'camera_a')];bb=groups[(did,'camera_b')]
                for scope in ['all_frames','both_object_visible']:
                    arows=[r for r in aa if scope=='all_frames'or r['crop_xyxy']is not None];brows=[r for r in bb if scope=='all_frames'or r['crop_xyxy']is not None]
                    if not arows or not brows:continue
                    ia=[positions[r['sample_id']]for r in arows];ib=[positions[r['sample_id']]for r in brows];ta=np.array([r['clip_time_s']for r in arows]);tb=np.array([r['clip_time_s']for r in brows]);best=(x[ia]@x[ib].T).argmax(1);err=np.abs(ta-tb[best])
                    shuffled=[float(np.mean(np.abs(ta-rng.permutation(tb)[best])))for _ in range(100)]
                    results.append({'demo_id':did,'method':method,'scope':scope,'primary_queries':len(ia),'secondary_candidates':len(ib),'mean_time_error_s':float(err.mean()),'within_075_s':int(sum(err<=.75)),'shuffle_mean_s':float(np.mean(shuffled)),'reversed_time_mean_s':float(np.abs(ta-tb[::-1][best]).mean()),'static_first_frame_mean_s':float(np.abs(ta-tb[0]).mean())})
        save(a.output/'feature-comparison.json',{'within_known_pair':results,'cross_take_retrieval':retrieval,'limits':['No fitted temporal model. Eight-bin summary retains rough order but duration-normalizes complete clips.','Whole clip/future frames are available in offline matching; not causal control.','Crop falls back to full frame when object missing. Both-visible scope compares exact same queries/candidates for both methods.','Recording session confounds categories. No independent session-level category accuracy claim.']})
    summary={'frames':len(rows),'four_second_windows':len(windows),'windows_by_split':dict(collections.Counter(r['split']for r in windows)),'unique_anchor_samples':len({r['sample_id']for w in windows for r in w['anchors']}),'source_human_takes':16,'physical_contact_labels':0,'robot_action_labels':0,'previews':len(preview_meta),'synchronization_uncertainty_s':.25,'time_resolution_note':'10Hz preserves more observations; does not improve the original synchronization uncertainty','adapter_gate_passed':False,'adapter_missing':['measured physical scale/camera pose','qualified lift/metric depth','independent contact and gripper event timing']}
    save(a.output/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    assert circ_diff(5,355)==10
    rr=[{'valid_xy':True,'clip_time_s':0.,'xy_normalized':[0,0]},{'valid_xy':False,'clip_time_s':.1,'xy_normalized':None},{'valid_xy':True,'clip_time_s':.2,'xy_normalized':[1,0]}]
    assert not valid_pairs(rr)
    assert anchors([{'clip_time_s':t}for t in [0.,1.,2.,3.,4.]],0,4,30)is None
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
