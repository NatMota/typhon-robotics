"""Audit actual optimizer-step sample consumption and saved report references."""
from load04_common import *
import collections,numpy as np,re
data=read(OUT/'data/qualified-dataset.json');ws={w['id']:w for w in data['windows']}
logs={m:[json.loads(x) for x in (OUT/('train-'+m)/'losses.jsonl').read_text().splitlines()] for m in ['C','B']}
counts={};flat={}
for m,rows in logs.items():
 assert [x['step'] for x in rows]==list(range(1,3001))
 ids=[s for r in rows for s in r['sample_ids']];assert len(ids)==12000
 flat[m]=[ws[x]['underlying_window'] for x in ids]
 c=collections.Counter((ws[x]['episode'],ws[x]['visibility']) for x in ids)
 if m=='B':assert len(c)==16 and set(c.values())=={750}
 else:
  assert len(c)==8 and set(c.values())=={1500}
  assert all(ws[x]['visibility']==('hidden' if ws[x]['task']=='feeding' else 'visible') for x in ids)
 assert len(set(flat[m]))==619
 counts[m]={'samples':len(ids),'underlying_windows':len(set(flat[m])),'cell_visibility_counts':{'/'.join(k):v for k,v in c.items()}}
assert flat['C']==flat['B']
rows=read(OUT/'analysis/episodes.json')
totals={}
for m in ['C','B']:
 a=[x for x in rows if x['model']==m]
 totals[m]={'n':len(a),'supported_pickups':sum(x['supported_lift_10mm_3steps'] for x in a),'requested_region':sum(all(x['scores'][x['requested_task']]['criteria'][k] for k in ['position','footprint_inside_target']) for x in a),'strict':sum(x['requested_success'] for x in a),'other_strict':sum(x['other_success'] for x in a)}
assert totals['C']==dict(n=16,supported_pickups=16,requested_region=8,strict=8,other_strict=8)
assert totals['B']==dict(n=16,supported_pickups=16,requested_region=16,strict=12,other_strict=0)
doc=R/'experiment-history/2026-10-14-load04'
refs=[]
for file in ['REPORT.md','REPRODUCTION.md','PROTOCOL.md']:
 for dest in re.findall(r'\]\(([^)]+)\)',(doc/file).read_text()):
  if '://' not in dest:
   assert (doc/dest).exists(),(file,dest)
   refs.append({'from':file,'to':dest})
dump(OUT/'analysis/final-audit.json',{'passed':True,'actual_update_logs_match':True,'underlying_sample_order_identical':True,'actual_counts':counts,'totals':totals,'local_document_links_checked':refs,'time':time.time()})
print(json.dumps({'passed':True,'totals':totals,'links':len(refs)}))
