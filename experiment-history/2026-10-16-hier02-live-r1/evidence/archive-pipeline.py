from pathlib import Path
import json,time,hashlib,subprocess,base64,shlex
B=Path('/media/nm/Fast Drive/Typhon/research-20261015');P=B/'hier02-live-r1-preparation-20261016';D=B/'hier02-live-r1-20261016';D.mkdir(exist_ok=False)
ssh=['ssh','-F','/var/tmp/typhon-ucl-archive-pull/ssh_config','-o','BatchMode=yes','ucl-powerglide-transfer']
code="""from pathlib import Path
import json,hashlib,time
O=Path('/var/tmp/typhon-hier02-live-r1-20261016')
assert(O/'physical-complete.json').exists()and(O/'analysis/validation.json').exists()and(O/'visual-review/provenance.json').exists()
assert json.loads((O/'analysis/validation.json').read_text())['passed']
files=[]
for p in sorted(O.rglob('*')):
 if p.is_file()and p.name not in ['archive-manifest-v1.json','archive-verification-v1.json']:
  assert not p.is_symlink()
  h=hashlib.sha256()
  with p.open('rb')as f:
   for b in iter(lambda:f.read(8<<20),b''):h.update(b)
  files.append({'path':str(p.relative_to(O)),'size':p.stat().st_size,'sha256':h.hexdigest()})
m={'schema':'sha256-file-tree-v1','source':str(O),'files':files,'file_count':len(files),'total_bytes':sum(f['size']for f in files),'epoch':time.time(),'experiment_status':'physical and independent analysis complete; raw outputs sealed'}
(O/'archive-manifest-v1.json').write_text(json.dumps(m,indent=2)+chr(10));print(json.dumps({'file_count':m['file_count'],'total_bytes':m['total_bytes'],'manifest_sha256':hashlib.sha256((O/'archive-manifest-v1.json').read_bytes()).hexdigest(),'archive_deadline':json.loads((O/'clock.json').read_text())['archive_deadline']}))
"""
r=subprocess.run(ssh+['python3 -'],input=code,text=True,capture_output=True,timeout=70);assert r.returncode==0,r.stderr
receipt=json.loads(r.stdout);(P/'archive-source-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
with(P/'archive-transfer.log').open('w')as log:
 r=subprocess.run(['rsync','-az','--partial','--info=progress2','-e','ssh -F /var/tmp/typhon-ucl-archive-pull/ssh_config','ucl-powerglide-transfer:/var/tmp/typhon-hier02-live-r1-20261016/',str(D)+'/'],stdout=log,stderr=subprocess.STDOUT,timeout=max(1,min(1500,receipt['archive_deadline']-time.time()-60)))
assert r.returncode==0
manifest=D/'archive-manifest-v1.json';assert hashlib.sha256(manifest.read_bytes()).hexdigest()==receipt['manifest_sha256']
m=json.loads(manifest.read_text())
for e in m['files']:
 p=D/e['path'];assert p.stat().st_size==e['size'];h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 assert h.hexdigest()==e['sha256'],e['path']
v={'verified':True,'files':m['file_count'],'bytes':m['total_bytes'],'manifest_sha256':receipt['manifest_sha256'],'hash_errors':0,'epoch':time.time(),'destination':str(D)}
(D/'archive-verification-v1.json').write_text(json.dumps(v,indent=2)+'\n');(P/'archive-verification-v1.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v),flush=True)
