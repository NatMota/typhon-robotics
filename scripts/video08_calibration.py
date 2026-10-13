"""VIDEO08: camera-geometry sensitivity, not calibrated physical reconstruction."""
import argparse, csv, hashlib, json, math, platform, sys
from pathlib import Path
import cv2
import numpy as np

XYZ = np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
XY = XYZ[:,:2].astype(np.float32)

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def project(points, K, rvec, tvec):
    R = cv2.Rodrigues(rvec)[0]
    camera = (R @ np.asarray(points).T).T + np.asarray(tvec).reshape(1,3)
    pixels = (K @ camera.T).T
    return pixels[:,:2] / pixels[:,2:], camera[:,2]

def focal_constraints(corners, principal):
    H = cv2.getPerspectiveTransform(XY, np.asarray(corners,np.float32))
    B = H.copy()
    B[0] -= principal[0] * B[2]
    B[1] -= principal[1] * B[2]
    n = [-np.dot(B[:2,0],B[:2,1]),
         -(np.dot(B[:2,0],B[:2,0])-np.dot(B[:2,1],B[:2,1]))]
    d = [B[2,0]*B[2,1], B[2,0]**2-B[2,1]**2]
    return [float(a/b) if abs(b)>1e-12 else None for a,b in zip(n,d)]

def solve(corners, K):
    result = cv2.solvePnPGeneric(XYZ, np.asarray(corners,np.float64), K,
                                np.zeros(5), flags=cv2.SOLVEPNP_IPPE)
    out = []
    for i,(r,t) in enumerate(zip(result[1],result[2])):
        r,t = cv2.solvePnPRefineLM(XYZ, np.asarray(corners,np.float64), K,
                                 np.zeros(5), r.copy(), t.copy())
        xy,depth = project(XYZ,K,r,t)
        # Independent direct projection agrees with OpenCV.
        cvxy = cv2.projectPoints(XYZ,r,t,K,np.zeros(5))[0].reshape(-1,2)
        assert np.max(np.abs(xy-cvxy)) < 1e-7
        R = cv2.Rodrigues(r)[0]
        centre = (-R.T@t).reshape(3)
        tilt = math.degrees(math.acos(min(1.,abs(float(R[2,2])))))
        error = np.linalg.norm(xy-corners,axis=1)
        probe = np.array([[.5,.5,math.copysign(.5,float(centre[2]))]])
        probe_xy, probe_depth = project(probe,K,r,t)
        out.append(dict(branch=i,rvec=r.ravel().tolist(),tvec=t.ravel().tolist(),
            camera_centre_target_units=centre.tolist(),height_target_units=abs(float(centre[2])),
            tilt_deg=tilt,rms_px=float(np.sqrt(np.mean(error**2))),max_corner_error_px=float(error.max()),
            corner_errors_px=error.tolist(),reprojected_corners_px=xy.tolist(),
            all_positive_depth=bool(np.all(depth>0)),
            hypothetical_half_side_probe_px=probe_xy[0].tolist(),
            probe_positive_depth=bool(probe_depth[0]>0)))
    return out

def selfcheck():
    K=np.array([[700.,0.,239.],[0.,700.,425.],[0.,0.,1.]])
    r=np.array([.25,-.30,.12]); t=np.array([-.4,-.3,6.])
    corners,depth=project(XYZ,K,r,t)
    f2=focal_constraints(corners,[239.,425.])
    assert all(abs(math.sqrt(v)-700.) < .03 for v in f2), f2
    models=solve(corners,K)
    best=min(models,key=lambda x:x['max_corner_error_px'])
    assert best['max_corner_error_px'] < 1e-6
    assert np.max(np.abs(np.asarray(best['rvec'])-r)) < 1e-6
    assert np.max(np.abs(np.asarray(best['tvec'])-t)) < 1e-6
    # Rotating primary image CCW and back is exact, including coordinate convention.
    a=np.arange(478*850,dtype=np.int32).reshape(850,478)
    b=cv2.rotate(a,cv2.ROTATE_90_COUNTERCLOCKWISE)
    assert np.array_equal(a,cv2.rotate(b,cv2.ROTATE_90_CLOCKWISE))
    for x,y in [(237,435),(316,431),(323,518),(245,532)]:
        assert a[y,x] == b[478-1-x,y]
    return dict(exact_camera_recovery=True,analytic_focal_error_px=[abs(math.sqrt(v)-700.) for v in f2],
                independent_projection_matches_opencv=True,primary_rotation_roundtrip=True)

def distribution(values):
    a=np.asarray(values,float)
    return dict(min=float(a.min()),p05=float(np.quantile(a,.05)),median=float(np.median(a)),
                p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else None

def run(protocol, output):
    p=json.loads(protocol.read_text()); output.mkdir(parents=True,exist_ok=False)
    check=selfcheck(); write_json(output/'selfcheck.json',check)
    rng=np.random.default_rng(p['sensitivity']['seed'])
    root=Path(p['source_root']); source=json.loads((root/'source-index.json').read_text())
    summary={}; all_candidates=[]; focal_draws=[]; references=[]
    for cam in p['cameras']:
        w,h=cam['raw_wh']; long=max(w,h); corners=np.array(cam['corners_raw_px'],float)
        principal=np.array([(w-1)/2,(h-1)/2])
        f2=focal_constraints(corners,principal)
        draws=[]
        for i in range(p['sensitivity']['corner_draws']):
            perturbed=corners+rng.uniform(-p['uncertainty_px'],p['uncertainty_px'],(4,2))
            q=focal_constraints(perturbed,principal)
            draws.append(q);focal_draws.append(dict(camera=cam['id'],draw=i,corner_px=perturbed.tolist(),focal_squared=q))
        candidates=[]
        for fx in np.geomspace(*p['sensitivity']['focal_over_long_side'],p['sensitivity']['focal_samples']):
            for dx in p['sensitivity']['principal_offsets_fraction']:
                for dy in p['sensitivity']['principal_offsets_fraction']:
                    f=float(fx*long); cx=float(principal[0]+dx*w); cy=float(principal[1]+dy*h)
                    K=np.array([[f,0.,cx],[0.,f,cy],[0.,0.,1.]])
                    for result in solve(corners,K):
                        result.update(camera=cam['id'],focal_px=f,
                            focal_over_long_side=float(fx),principal_px=[cx,cy],
                            principal_offset_fraction=[dx,dy],
                            long_side_fov_deg=math.degrees(2*math.atan(long/(2*f))))
                        result['compatible_with_corner_bound']=bool(result['all_positive_depth'] and result['max_corner_error_px']<=p['pose_fit']['accepted_max_corner_error_px'])
                        candidates.append(result)
        accepted=[x for x in candidates if x['compatible_with_corner_bound']]
        centred=[x for x in accepted if x['principal_offset_fraction']==[0,0]]
        cond=[q for q in draws if all(v is not None and v>0 for v in q)]
        ratio=[max(q)/min(q) for q in cond]
        probe=np.array([x['hypothetical_half_side_probe_px'] for x in accepted if x['probe_positive_depth']])
        max_spread=0.
        if len(probe):
            max_spread=float(np.sqrt(((probe[:,None,:]-probe[None,:,:])**2).sum(-1)).max())
        summary[cam['id']]=dict(raw_wh=[w,h],nominal_focal_squared=f2,
            nominal_real_focal_px=[math.sqrt(v) if v is not None and v>0 else None for v in f2],
            corner_perturbations=len(draws),positive_both_focal_constraints=len(cond),
            squared_focal_constraint_ratio=distribution(ratio),
            positive_consistent_within_10percent_focal=sum(abs(math.sqrt(q[0]/q[1])-1)<=.10 for q in cond),
            candidate_count=len(candidates),accepted_count=len(accepted),centred_accepted_count=len(centred),
            accepted_focal_px=distribution([x['focal_px'] for x in accepted]),
            accepted_height_target_units=distribution([x['height_target_units'] for x in accepted]),
            centred_height_target_units=distribution([x['height_target_units'] for x in centred]),
            accepted_tilt_deg=distribution([x['tilt_deg'] for x in accepted]),
            minimum_corner_fit_error_px=min(x['max_corner_error_px'] for x in candidates),
            hypothetical_probe_max_prediction_disagreement_px=max_spread,
            hypothetical_probe_count=len(probe))
        all_candidates.extend(candidates)
        sample=next(x for x in source['samples'] if x['sample_id']==cam['sample_id'])
        frame=root/cam['frame_path']; assert sample['frame_path']==cam['frame_path']
        assert sha(frame)==sample['frame_sha256']
        im=cv2.imread(str(frame)); assert im is not None
        if cam['canonical_rotate_ccw']:im=cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE)
        assert im.shape[:2]==(h,w)
        im=cv2.copyMakeBorder(im,42,0,0,0,cv2.BORDER_CONSTANT,value=(16,16,16))
        pts=np.rint(corners+[0,42]).astype(int)
        cv2.polylines(im,[pts],True,(0,255,255),2)
        for i,(x,y) in enumerate(pts):cv2.putText(im,str(i),(x+4,y-4),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
        cv2.putText(im,cam['id']+' | inherited M0 anchors',(10,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.putText(im,'RAW pixels; not measured calibration',(10,36),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        cv2.imwrite(str(output/(cam['id']+'-reference.jpg')),im)
        references.append(dict(camera=cam['id'],source_sample=sample,frame_sha256=sha(frame)))
    write_json(output/'summary.json',summary)
    write_json(output/'candidates.json',all_candidates)
    write_json(output/'corner-sensitivity.json',focal_draws)
    write_json(output/'source-references.json',references)
    write_json(output/'environment.json',dict(python=sys.version,opencv=cv2.__version__,numpy=np.__version__,
        platform=platform.platform(),compute='CPU audit; no GPU, no model updates'))
    write_json(output/'inputs.json',dict(protocol_sha256=sha(protocol),script_sha256=sha(__file__),
        source_index_sha256=sha(root/'source-index.json')))
    fields=['camera','branch','focal_px','long_side_fov_deg','height_target_units','tilt_deg','rms_px','max_corner_error_px','compatible_with_corner_bound']
    with (output/'candidates.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(all_candidates)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();run(a.protocol,a.output)
