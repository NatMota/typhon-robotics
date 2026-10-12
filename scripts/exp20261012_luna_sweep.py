"""Run the frozen visual-choice study with user-authorized Luna Decisions API."""
import importlib.util,json,os,shutil,hashlib,sys,datetime
from pathlib import Path
ROOT=Path('/var/tmp/typhon-ost-sweep-20261012/luna')
OLD=Path('/var/tmp/typhon-research-20261011')
def load(name):
 p=Path(__file__).with_name(name+'.py');s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
g=load('exp20261011_ghost_critic_v2');v=load('exp20261011_ghost_visual_goal_compare_v3')
def run():
 keyfile=Path.home()/'.config/typhon-secrets/openai-decisions.key'
 assert keyfile.stat().st_mode&0o077==0
 os.environ['OPENAI_API_KEY']=keyfile.read_text().strip()
 for oldname,candidates in [('ghost-critic-v2',OLD/'replanning-candidates/candidates'),('ghost-critic-supplement-v2',OLD/'replanning-candidates-supplement/candidates')]:
  old=OLD/oldname;dst=ROOT/oldname
  if not dst.exists():
   dst.mkdir(parents=True)
   for name in ['protocol.json','current_raw_audit.json','comparison_v2.json','validation_v2.json']:
    shutil.copy2(old/name,dst/name)
   for name in ['current','slates','secondary_visual_goal_v3']:shutil.copytree(old/name,dst/name)
   source_manifests=[]
   for mp in sorted((dst/'slates').glob('*/*/*/manifest.json')):
    m=g.read(mp);original=g.sha(mp)
    m['current_image']=str(dst/Path(m['current_image']).relative_to(old))
    assert g.sha(m['current_image'])==m['current_sha256']
    for c in m['candidates'].values():
     c['image']=str(dst/Path(c['image']).relative_to(old));assert g.sha(c['image'])==c['image_sha256']
    g.save(mp,m);source_manifests.append({'path':str(mp.relative_to(dst)),'original_sha256':original,'relocated_sha256':g.sha(mp)})
   g.save(dst/'relocation.json',{'source':str(old),'preserved_protocol_sha256':g.sha(dst/'protocol.json'),'pixels_verified_unchanged':True,'only_image_paths_relocated':True,'manifests':source_manifests,'time':g.stamp()})
  assert not any('luna' in str(p) for p in (old/'slates').glob('*/*/*/luna/choice.json'))
  if not list((dst/'slates').glob('*/*/*/luna/choice.json')):g.selfcheck(dst)
  for choice in sorted((dst/'slates').glob('*/*/*/luna/choice.json')):
   result=g.read(choice)
   if result['status']=='transport_error' and any(x in result.get('detail','') for x in ['504','502','503','timed out']):
    archived=choice.parent.with_name('luna-transport-attempt01')
    if not archived.exists():choice.parent.rename(archived)
  g.run(dst,'both',32)
  failures=[g.read(x) for x in (dst/'slates').glob('*/*/*/luna/choice.json') if g.read(x)['status']!='valid']
  if failures:
   g.save(ROOT/'blocked.json',{'set':oldname,'failures':failures});raise RuntimeError('Decision API failure; see redacted evidence')
  g.compare(dst,candidates)
  v.compare(dst)
 g.save(ROOT/'complete.json',{'at_utc':g.stamp(),'model':g.MODEL,'endpoint':g.ENDPOINT,'ordinary_rollouts_changed':False,'role':'offline privileged simulator-lookahead ranking experiment; no deployable world model claim'})
 os.environ.pop('OPENAI_API_KEY',None)
if __name__=='__main__':run()
