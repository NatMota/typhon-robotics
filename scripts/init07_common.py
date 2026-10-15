"""INIT07 constants and receipts; private output, unchanged source banks."""
import os,sys,json,time,hashlib,copy
from pathlib import Path
R=Path(__file__).resolve().parents[1]
OUT=Path('/var/tmp/typhon-init07-20261015')
INIT=Path('/var/tmp/typhon-init01-20261015')
OLD=Path('/var/tmp/typhon-load01-20261013')
BASE=Path('/var/tmp/typhon-init02-20261015/train-T/step3000.pt')
BASE_SHA='4d7a4a2547f084b6026164f27a6b7d43b5cb4d3b83ee4875343d108908f35636'
SOURCES=['demo_0001','demo_0002','demo_0003','demo_0005']
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
