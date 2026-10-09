"""Independent source-clock checks for the four-second Run3 export."""
import argparse,csv,json,sys,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from supervisor.adaptation.m7_catalog_data import M7Dataset
from supervisor.adaptation.m7_timing import reconstruct_20hz

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(args):
    cfg=json.loads((args.temporal/'training_data_config.json').read_text());export=args.temporal/'export'
    ds=M7Dataset(export,Path(cfg['source_snapshot']));parent=defaultdict(list)
    qa={r['sample_id']:r for r in map(json.loads,(args.temporal/'trajectory_anchor_pairs.jsonl').read_text().splitlines())}
    candidates=list(csv.DictReader((export/'v3_candidates.csv').open()));counts=defaultdict(set);rows=[]
    for c in candidates:
        assert c['history_mode']=='sparse' and c['green_held_marker']=='True'
        counts[c['sample_id']].add(c['camera_variant'])
    maximum_xyz_error=0.;maxlag=0.;totalanchors=0
    for row in ds.rows:
        sid=row['sample_id'];p=ds.pack(sid);s=p['supervision'];pr=p['provenance'];source=ds.sources[row['recording_id']]
        start,end=int(row['action_start']),int(row['action_end_exclusive']);frame=int(row['frame_index']);ix=list(map(int,row['anchor_action_indices'].split()))
        assert 0<end-start<=80 and len(ix)<=30 and min(ix)==start and max(ix)==end-1
        assert counts[sid]=={'overhead_main','oblique_main'}
        h=p['observation']['history']['sparse'];assert h['frame_indices'][-1]==frame and len(h['frame_indices'])<=5
        expected=list(range(int(source['frame_action_indices'][frame]),max(-1,start-81),-20))[::-1]
        assert h['action_indices']==expected,(sid,h,expected)
        assert all(-4<=t<=0 for t in h['relative_timestamps_s'])
        q=qa[sid];assert q['future_anchor_action_indices']==ix
        for i,t,f,af,lag in zip(ix,q['future_anchor_timestamps_s'],q['paired_preceding_rgb_frame_indices'],q['paired_preceding_rgb_action_indices'],q['rgb_to_anchor_lag_s']):
            assert source['frame_action_indices'][f]==af and af<=i<af+4 and abs(t-i/20)<1e-9
            assert abs(lag-(i-af)/20)<1e-9 and lag<=.15;maxlag=max(maxlag,lag)
        request=dict(source='training_demonstration_supervision',scope=s['scope'],instruction=s['semantic_instruction'],anchor_times_s=s['timestamps_s'][:s['true_length']],aligned_duration_s=s['aligned_duration_seconds'])
        payload=dict(request=request,waypoints=[dict(time_s=t,target_generic_interleaved=a) for t,a in zip(request['anchor_times_s'],s['action'][:s['true_length']])])
        replay=reconstruct_20hz(payload,oracle_replay=True)['targets'];native=np.array(source['targets'][start:end]);assert np.array_equal(replay[np.array(ix)-start],native[np.array(ix)-start])
        assert np.array_equal(replay[:,[9,19]],native[:,[9,19]])
        error=float(np.max(np.linalg.norm(replay[:,:3]-native[:,:3],axis=1)));maximum_xyz_error=max(maximum_xyz_error,error);totalanchors+=len(ix)
        parent[row['parent_sample_id']].append((start,end,int(row['parent_action_start']),int(row['parent_action_end_exclusive'])))
        rows.append(dict(sample_id=sid,input_frame=frame,input_action_start=start,input_history_count=len(h['frame_indices']),output_anchor_count=len(ix),seconds=(end-start)/20,maximum_xyz_interpolation_error_m=error))
    for pid,windows in parent.items():
        w=sorted(windows);assert w[0][0]==w[0][2] and w[-1][1]==w[-1][3]
        assert all(a[1]==b[0] for a,b in zip(w,w[1:])),pid
    assert len(parent)==316 and len(rows)==388 and len(candidates)==776 and len(ds.sources)==18
    hashes={str(p):sha(p) for p in [args.temporal/'training_data_config.json',export/'SHA256SUMS',args.temporal/'trajectory_anchor_pairs.jsonl',args.temporal/'history_image_validation.json',Path(cfg['source_snapshot']),Path(__file__)]}
    result=dict(pass_=True,temporal_export=str(args.temporal.resolve()),parents=len(parent),windows=len(rows),camera_examples=len(candidates),sources=len(ds.sources),total_anchor_targets=totalanchors,maximum_future_seconds=max(r['seconds'] for r in rows),sparse_history_hz=1,maximum_history_frames=max(r['input_history_count'] for r in rows),maximum_future_anchor_rgb_lag_s=maxlag,maximum_xyz_interpolation_error_m=maximum_xyz_error,exact_gripper_transition_reconstruction=True,all_parent_native_intervals_tiled_once=True,source_phone_to_robot_alignment=False,required_before_training='Target-runtime finite-loss/image/history smoke; newvariant3 output only',evaluation_limitation='15mm intermediate requested-target reconstruction deviation needs physical rollout checking; this audit does not establish controller success.',hashes=hashes)
    result['pass']=result.pop('pass_');args.output.mkdir(exist_ok=True,parents=True);(args.output/'run3_export_alignment_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    with (args.output/'run3_window_checks.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--temporal',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
