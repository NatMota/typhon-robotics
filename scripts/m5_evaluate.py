"""Frozen M5 matched rollout panel and independent motion diagnostics."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys,json,argparse,hashlib,gzip,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from motion_policy_diagnostics import PrefixPolicy,audit
from supervisor.baseline.benchmark import run_episode
OUT=Path('results/personal-m5-20261009')

def supplemental(row,folder):
    records=[json.loads(s) for s in gzip.open(folder/'trajectory.jsonl.gz','rt')]
    states=[r['truth'] for r in records];initial=row['initial_truth'];z0=initial['object_position_m'][2]
    heights=np.array([s['object_position_m'][2]-z0 for s in states]);bilateral=np.array([s['bilateral_grasp_contact'] for s in states],bool)
    supported=bilateral & (heights>=.01)
    runs=[];start=None
    for i,yes in enumerate(supported.tolist()+[False]):
        if yes and start is None:start=i
        if not yes and start is not None:runs.append((start,i));start=None
    longest=max([b-a for a,b in runs],default=0)
    obj=np.array([s['object_position_m'] for s in states]);eef=np.array([s['eef_position_m'] for s in states]);begin=np.array(initial['object_position_m'])
    horizontal=np.linalg.norm(obj[:,:2]-begin[:2],axis=1)
    return {'minimum_eef_object_distance_mm':float(np.min(np.linalg.norm(eef-obj,axis=1))*1000),'maximum_lift_above_initial_mm':float(heights.max()*1000),'supported_lift_10mm_bilateral_steps':int(supported.sum()),'longest_supported_lift_s':longest/20,'supported_lift_10mm_3steps':longest>=3,'legacy_height_only_70mm':bool((heights>=.07).any()),'horizontal_transport_while_supported_mm':float(horizontal[supported].max()*1000) if supported.any() else 0.,'object_path_m':float(np.linalg.norm(np.diff(np.vstack([begin,obj]),axis=0),axis=1).sum()),'diagnostic_definition':'supported lift = >=10mm above initial resting centre with simultaneous bilateral contacts for >=3 consecutive20Hz steps; supplemental, strict score unchanged; visual inspection required','release_and_stability':row['evaluation']['criteria']}

def main(a):
    protocol=json.loads((OUT/'evaluation_protocol.json').read_text());scenarios=protocol['conditional_loading'] if a.loading else protocol['core']
    if a.model=='W1000':scenarios=scenarios[:1]
    if a.loading:
        sources=json.loads((OUT/'evaluation/A3000/summary.json').read_text())
        assert any(r['group']=='seen_source' and r['success'] for r in sources),'Conditional loading gate not met'
    checkpoint=None if a.model=='F' else OUT/'training'/('A3000' if a.model=='W1000' else a.model)/(a.model+'.pt')
    if checkpoint:
        v=json.loads(checkpoint.with_suffix('.reload.json').read_text());assert v['passed'] and v['fresh_process']
    folder=OUT/'evaluation'/(a.model+('-loading' if a.loading else ''));folder.mkdir(parents=True,exist_ok=False)
    manifest={'model':a.model,'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else 'original pinned 129e71460678b7236cee6fc9707f09d9fa0c3590','protocol_sha256':hashlib.sha256((OUT/'evaluation_protocol.json').read_bytes()).hexdigest(),'training_selection':'fixed budget, no best checkpoint selection','actor_truth':False,'H':30,'wrist':'native','extra_guards':False}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    policy=PrefixPolicy(30,'native','O2',checkpoint,folder/'worker.log');rows=[]
    policy.model_manifest['M5_checkpoint']=manifest
    try:
        for s in scenarios:
            if time.time()>=1791565730:raise RuntimeError('M5 rollout deadline reached; completed episodes retained')
            dest=folder/s['scenario_id'];r=run_episode(s,'pretrained',dest,policy,{'m5':manifest,'model':policy.model_manifest},episode_setup=policy.instrument)
            if hasattr(policy,'telemetry'):policy.telemetry.close()
            np.savez_compressed(dest/'raw_predictions.npz',chunks=np.asarray(policy.raw_chunks),histories=np.asarray([np.full(20,np.nan) if h is None else h for h in policy.histories]))
            if r['controller_exceptions'] or r['artifact_errors'] or r['evaluation_errors']:
                metrics={'id':s['scenario_id'],'group':s['group'],'success':r['success'],'integration_errors':{'controller':r['controller_exceptions'],'artifacts':r['artifact_errors'],'evaluation':r['evaluation_errors']}}
            else:metrics={'id':s['scenario_id'],'group':s['group'],**audit(r,dest),**supplemental(r,dest)}
            rows.append(metrics);(folder/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(a.model,metrics,flush=True)
    finally:policy.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['F','C3000','A3000','W1000'],required=True);p.add_argument('--loading',action='store_true');main(p.parse_args())
