"""Create/verify the new diagnostic artifact manifest; never touches frozen outputs."""
import argparse,hashlib,json,socket,datetime
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--create',action='store_true');p.add_argument('--root',default='results/motion-diagnostics-20261009');a=p.parse_args();root=Path(a.root);manifest=root/'artifact_manifest.json'
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
if a.create:
 files=[{'path':str(f.relative_to(root)),'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(root.rglob('*')) if f.is_file() and f.name not in ['artifact_manifest.json','backup_validation_mint.json']]
 manifest.write_text(json.dumps({'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_host':socket.gethostname(),'files':files,'scope':'All new diagnostic artifacts except manifest itself and destination validation'},indent=2)+'\n');print('Manifest files',len(files),'bytes',sum(f['bytes'] for f in files))
else:
 m=json.loads(manifest.read_text());errors=[]
 for row in m['files']:
  f=root/row['path']
  if not f.is_file() or f.stat().st_size!=row['bytes'] or digest(f)!=row['sha256']:errors.append(row['path'])
 r={'passed':not errors,'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'destination_host':socket.gethostname(),'root':str(root.resolve()),'files':len(m['files']),'bytes':sum(row['bytes'] for row in m['files']),'manifest_sha256':digest(manifest),'mismatches':errors};(root/'backup_validation_mint.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));assert not errors
