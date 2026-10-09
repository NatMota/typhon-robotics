"""M2 action contract with explicitly versioned observation and prompt weights."""
import os
import subprocess
from pathlib import Path
from supervisor.baseline.xvla import XvlaPolicy, MODEL_MANIFEST


class AdaptedPolicy(XvlaPolicy):
    def __init__(self, version, checkpoint=None, log_path=None):
        root=Path(__file__).resolve().parents[2]
        self.proprio=None;self.inferences=[]
        self.model_manifest={**MODEL_MANIFEST,'adapter_version':'m3-observation-v1-m2-actions',
            'observation_version':version,'weights_changed':checkpoint is not None,'prompt_checkpoint':str(checkpoint),
            'images':['full primary','full secondary','RGB-only crop if O2; otherwise masked']}
        self.log=open(log_path,'a')
        command=[str(root/'.venv-xvla/bin/python'),'-m','supervisor.adaptation.worker','--version',version]
        if checkpoint:command+=['--checkpoint',str(Path(checkpoint).resolve())]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,cwd=root,
            env=dict(os.environ,HF_HOME=str(root/'.cache-xvla/hf'),HF_HUB_OFFLINE='1'))
        try:
            self.load_info=self._receive(240)
            if not self.load_info.get('ready'):raise RuntimeError(self.load_info)
            self.model_manifest['inference_configuration']=self.load_info['inference_configuration']
        except Exception:
            self.close();raise
