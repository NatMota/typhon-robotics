"""Matched M6 initial/final evaluation plus a separately labelled released-Pt control."""
import os,sys,json,argparse,hashlib,gzip,time,subprocess
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from motion_policy_diagnostics import PrefixPolicy,audit
from m5_evaluate import supplemental
from supervisor.baseline.benchmark import run_episode
OUT=Path('results/personal-m6-joint-20261009')
class M6Policy(PrefixPolicy):
    def __init__(self,condition,checkpoint,log_path):
        root=Path(__file__).resolve().parents[1]
        self.horizon=30;self.wrist='native';self.proprio=None;self.inferences=[]
        self.model_manifest={'method':condition,'repository':'2toINF/X-VLA-Pt','revision':'c1c4a64a7e03ac5b95c468bf1578f3d03651b53b','observation_version':'O2','domain_id':3 if condition=='vanilla-domain3' else 29,'native_model_rotation_order':'concatenated columns' if condition=='vanilla-domain3' else 'interleaved columns','returned_OSC_adapter_order':'concatenated columns','weights_changed':condition!='vanilla-domain3','training_changed_weights':condition=='FT3000','custom_domain_reset':condition=='custom-init','H':30,'steps':10,'proprio_history':'initial measured pose then previous-command target','actor_truth':False}
        self.log=open(log_path,'a')
        cmd=[str(root/'.venv-xvla/bin/python'),'-m','supervisor.adaptation.m6_worker','--version','O2','--condition',condition]
        if checkpoint:cmd+=['--checkpoint',str(checkpoint.resolve())]
        self.process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,cwd=root,env=dict(os.environ,HF_HOME=str(root/'.cache-xvla/hf'),HF_HUB_OFFLINE='1'))
        try:
            self.load_info=self._receive(360)
            assert self.load_info['ready'],self.load_info
        except Exception:self.close();raise

def main(a):
    checkpoint=OUT/'training/final.pt' if a.condition=='FT3000' else None
    if checkpoint:assert json.loads(checkpoint.with_suffix('.reload.json').read_text())['passed']
    root=OUT/'evaluation'/a.condition;root.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((OUT/'evaluation_protocol.json').read_text());manifest={'condition':a.condition,'protocol_sha256':hashlib.sha256((OUT/'evaluation_protocol.json').read_bytes()).hexdigest(),'checkpoint':str(checkpoint),'selection':'fixed final, no tuned checkpoint selection','extra_completion_guard':False}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');policy=M6Policy(a.condition,checkpoint,root/'worker.log');rows=[]
    try:
        for s in protocol['core']:
            if time.time()>=1791565730:raise RuntimeError('Rollout cutoff; partial episodes retained')
            dest=root/s['scenario_id'];r=run_episode(s,'pretrained',dest,policy,{'m6':manifest,'model':policy.model_manifest},episode_setup=policy.instrument)
            if hasattr(policy,'telemetry'):policy.telemetry.close()
            np.savez_compressed(dest/'raw_predictions.npz',chunks=np.asarray(policy.raw_chunks),histories=np.asarray([np.full(20,np.nan) if h is None else h for h in policy.histories]))
            if r['controller_exceptions'] or r['artifact_errors'] or r['evaluation_errors']:metrics={'id':s['scenario_id'],'group':s['group'],'success':r['success'],'errors':{'controller':r['controller_exceptions'],'artifact':r['artifact_errors'],'evaluation':r['evaluation_errors']}}
            else:metrics={'id':s['scenario_id'],'group':s['group'],**audit(r,dest),**supplemental(r,dest)}
            rows.append(metrics);(root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(a.condition,metrics,flush=True)
    finally:policy.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=['custom-init','FT3000','vanilla-domain3'],required=True);main(p.parse_args())
