"""Separately labelled reachability check; never a learned-policy success."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse,json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from supervisor.baseline.benchmark import run_episode,save


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(args.protocol.read_text());rows=[]
    wanted={'m7-personal_demo_0001','m7-personal_demo_0006','recovery-wrong_orientation-01'}
    manifest={'evidence_type':'privileged geometric expert reachability diagnostic',
        'uses_truth':True,'learned_policy_successes':0,
        'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        'no_change_to_policy_or_training_data':True}
    save(args.output/'manifest.json',manifest)
    for scenario in protocol['pilot']:
        if scenario['scenario_id'] not in wanted:continue
        result=run_episode(scenario,'expert',args.output/scenario['scenario_id'],provenance=manifest)
        rows.append({k:result[k] for k in ['scenario_id','group','status','success','position_error_mm','yaw_error_deg','controller_exceptions','artifact_errors','evaluation_errors','wall_duration_s']})
        save(args.output/'summary.json',{'results':rows,'evidence_type':manifest['evidence_type']})
        print(json.dumps(rows[-1]),flush=True)


if __name__=='__main__':main()
