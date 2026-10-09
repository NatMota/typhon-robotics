"""Matched M4 recovery rollouts, retaining the original M2 policy execution path."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import subprocess
from supervisor.adaptation.policy import AdaptedPolicy
from supervisor.adaptation.recovery import empty_grasp_setup,displacement_hook
from supervisor.baseline.benchmark import run_episode
from supervisor.baseline.protocol import load_suite,digest


def main(protocol_path,out,label,checkpoint):
    protocol,suite=load_suite(protocol_path)
    assert protocol['recovery_source_sha256']==hashlib.sha256(Path('supervisor/adaptation/recovery.py').read_bytes()).hexdigest()
    assert protocol['observation_source_sha256']==hashlib.sha256(Path('supervisor/adaptation/observations.py').read_bytes()).hexdigest()
    provenance={'label':label,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'protocol_sha256':digest(protocol),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None}
    if out.exists():assert json.loads((out/'run_manifest.json').read_text())==provenance
    else:out.mkdir(parents=True);(out/'run_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    policy=AdaptedPolicy('O2',checkpoint,out/'worker.log');rows=[]
    try:
        for scenario in suite['primary']:
            folder=out/scenario['scenario_id']
            if (folder/'episode.json').exists():row=json.loads((folder/'episode.json').read_text())
            else:
                setup=empty_grasp_setup if scenario.get('setup') else None
                hook=displacement_hook(scenario['intervention']) if scenario.get('intervention') else None
                row=run_episode(scenario,'pretrained',folder,policy,{**provenance,'model':policy.model_manifest},episode_setup=setup,chunk_intervention=hook)
            rows.append(row);(out/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            summary={'completed':len(rows),'scheduled':20,'by_category':{c:{'successes':sum(bool(r['success']) for r in rows if r['group']==c),'evaluated':sum(r['success'] is not None for r in rows if r['group']==c)} for c in protocol['category_counts']},
                'false_completion':'not applicable','already_completed':'goal preservation, not learned recognition'}
            (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(label,scenario['scenario_id'],row['success'],flush=True)
            if row['controller_exceptions'] or row['artifact_errors'] or row['evaluation_errors']:
                raise RuntimeError('Execution/artifact contract failure retained; stop for diagnosis')
    finally:policy.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--checkpoint',type=Path);a=p.parse_args();main(a.protocol,a.output,a.label,a.checkpoint)
