"""Verify old M3/M4 outputs against their pre-diagnostic artifact manifest."""
import hashlib,json,datetime
from pathlib import Path
root=Path('frozen_results').resolve().parent
manifest=root/'results/personal-m4-20261009/artifact_manifest.json'
expected='54eec35990c62dd4183de2e4527a9a20273524637ccff31eb5ac96a2dc0e9da0'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==expected
m=json.loads(manifest.read_text());errors=[];total=0
for item in m['files']:
 p=root/item['path'];h=hashlib.sha256()
 try:
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  total+=p.stat().st_size
  if p.stat().st_size!=item['bytes'] or h.hexdigest()!=item['sha256']:errors.append(item['path'])
 except OSError as e:errors.append({'path':item['path'],'error':str(e)})
result={'passed':not errors,'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_root':str(root),'manifest_sha256':expected,'files_checked':len(m['files']),'bytes':total,'mismatches':errors,'scope':m['scope']}
out=Path('results/motion-diagnostics-20261009/frozen_input_integrity.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));assert not errors
