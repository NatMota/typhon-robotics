"""JSON-lines inference for simulator virtual environments without Torch."""
import argparse,json,sys
import numpy as np
import torch
from .core import rule_action
from .evaluate import load_policy

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');a=p.parse_args()
    torch.set_num_threads(2)
    policy=load_policy(a.checkpoint)[0] if a.checkpoint else rule_action
    for line in sys.stdin:
        try:
            request=json.loads(line)
            action=policy(np.array(request['history'],np.float32),np.array(request['mask'],bool))
            print(json.dumps({'action':action}),flush=True)
        except Exception as e:
            print(json.dumps({'error':str(e)}),flush=True)
