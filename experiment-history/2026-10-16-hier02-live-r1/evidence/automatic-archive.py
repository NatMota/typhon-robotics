from pathlib import Path
import json,subprocess,time,traceback
B=Path('/media/nm/Fast Drive/Typhon/research-20261015');P=B/'hier02-live-r1-preparation-20261016'
assert not(P/'automatic-archive-start.json').exists(),'Do not start duplicate archive'
(P/'automatic-archive-start.json').write_text(json.dumps({'epoch':time.time(),'command':'await postprocess-complete, then archive-pipeline.py'},indent=2)+'\n')
ssh=['ssh','-F','/var/tmp/typhon-ucl-archive-pull/ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=15','ucl-powerglide-transfer']
q="from pathlib import Path\nimport json,time\nO=Path('/var/tmp/typhon-hier02-live-r1-20261016')\nx={'clock':json.loads((O/'clock.json').read_text()),'epoch':time.time()}\nfor n in ['STOPPED.json','postprocess-complete.json','physical-complete.json']:\n p=O/n\n if p.exists():x[n]=json.loads(p.read_text())\nprint(json.dumps(x))"
try:
 while True:
  r=subprocess.run(ssh+['python3 -'],input=q,text=True,capture_output=True,timeout=30);assert r.returncode==0,r.stderr
  x=json.loads(r.stdout);(P/'automatic-archive-state.json').write_text(json.dumps(x,indent=2)+'\n')
  assert 'STOPPED.json'not in x,'Original runtime stopped'
  assert time.time()<x['clock']['archive_deadline']-60,'Archive deadline reached'
  if 'postprocess-complete.json'in x:
   assert x['postprocess-complete.json']['passed'],x['postprocess-complete.json'];break
  time.sleep(15)
 print('POSTPROCESS_PASSED_STARTING_ARCHIVE',flush=True)
 exec((P/'archive-pipeline.py').read_text())
except BaseException:
 (P/'automatic-archive-error.json').write_text(json.dumps({'epoch':time.time(),'error':traceback.format_exc()},indent=2)+'\n')
 raise
