"""Matched final evaluation of immutable checkpoints on the versioned M2-R suite."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import subprocess
import time
from supervisor.adaptation.policy import AdaptedPolicy
from supervisor.baseline.benchmark import run_episode, summarize
from supervisor.baseline.protocol import load_suite, digest


def main(protocol_path,out,label,checkpoint,task=None):
    protocol,suite=load_suite(protocol_path)
    assert protocol['observation_source_sha256']==hashlib.sha256(Path('supervisor/adaptation/observations.py').read_bytes()).hexdigest()
    provenance={'label':label,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'protocol_sha256':digest(protocol),'scenarios_sha256':digest(suite),'observation_version':'O2',
        'checkpoint':str(checkpoint) if checkpoint else None,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
        'action_observer':'unchanged full M2 observation path; attempted optimization failed exact-equivalence gate and is not used','task_filter':task}
    if out.exists():
        assert json.loads((out/'run_manifest.json').read_text())==provenance,'Existing run differs; preserve it and choose a fresh output'
    else:
        out.mkdir(parents=True);(out/'run_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    scenarios=[s for s in suite['primary'] if task is None or s['task']==task]
    policy=AdaptedPolicy('O2',checkpoint,out/'worker.log');policy.model_manifest['method']=label
    rows=[];started=time.perf_counter()
    try:
        for scenario in scenarios:
            folder=out/scenario['scenario_id']
            if (folder/'episode.json').exists():row=json.loads((folder/'episode.json').read_text())
            else:row=run_episode(scenario,'pretrained',folder,policy,{**provenance,'model':policy.model_manifest})
            rows.append(row)
            (out/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            summary=summarize(rows);summary.update(label=label,completed=len(rows),scheduled=len(scenarios),run_wall_s=time.perf_counter()-started)
            (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(label,scenario['scenario_id'],row['success'],row['position_error_mm'],flush=True)
            if row['controller_exceptions'] or row['artifact_errors'] or row['evaluation_errors']:
                raise RuntimeError('Execution/artifact contract failure retained; stop for diagnosis')
    finally:policy.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--task',choices=['feeding','loading']);a=p.parse_args();main(a.protocol,a.output,a.label,a.checkpoint,a.task)
