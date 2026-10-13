"""Independent VIDEO11 arithmetic, implemented local geometry, and source preservation checks."""
import argparse,csv,json,math,hashlib,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from supervisor.feeding_loading.objects import _duplo_visual_v2
from exp20261011_hierarchy_runtime_v3 import ObservableBlock
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def run(base,audit,manifest):
 cfg=json.loads((base/'config.json').read_text());block=ET.Element('body');_duplo_visual_v2(block,cfg['object'],[])
 geoms=[g for g in block.findall('geom') if g.get('name','').startswith('duplo_v2_numeral_')]
 positions=np.array([list(map(float,g.get('pos').split())) for g in geoms]);sizes=np.array([list(map(float,g.get('size').split())) for g in geoms])
 areas=4*sizes[:,0]*sizes[:,1];centroid=np.average(positions[:,:2],weights=areas,axis=0);local=ObservableBlock(cfg).glyph_local
 assert np.max(abs(centroid-local))<1e-12
 top=positions[:,2]+sizes[:,2];assert np.max(abs(top-(cfg['object']['height_m']/2+.00008)))<1e-12
 rows=list(csv.DictReader((audit/'residuals.csv').open()));assert len(rows)==736
 output=[]
 for name in ['SEQ','END','SHUFFLE','STATIC']:
  group=[r for r in rows if r['condition']==name];assert len(group)==184
  numeric=lambda r,k:float(r[k])
  identity=[];residual=[];cosines=[];bmag=[];dmag=[]
  for r in group:
   c=[float(r['observer_ref_dx']),float(r['observer_ref_dy'])]
   a=[float(r['projected_ref_dx']),float(r['projected_ref_dy'])]
   b=[float(r['observer_projected_dx']),float(r['observer_projected_dy'])]
   delta=[float(r['depth_conversion_dx']),float(r['depth_conversion_dy'])]
   err=sum(v*v for v in c)-sum(v*v for v in a)-sum(v*v for v in b)-2*sum(x*y for x,y in zip(a,b))
   identity.append(abs(err));residual.append(sum((x-y)**2 for x,y in zip(b,delta)))
   nb=math.sqrt(sum(x*x for x in b));nd=math.sqrt(sum(x*x for x in delta));bmag.append(nb);dmag.append(nd)
   cosines.append(sum(x*y for x,y in zip(b,delta))/(nb*nd))
   assert float(r['assumed_plane_inverse_roundtrip_error'])<1e-5
  assert max(identity)<1e-12
  output.append({'condition':name,'n':len(group),'plane_shift_direction_cosine_mean':sum(cosines)/len(cosines),
    'observer_vs_fixed_landmark_RMSE':math.sqrt(sum(x*x for x in bmag)/len(bmag)),
    'fixed_pixel_depth_shift_RMSE':math.sqrt(sum(x*x for x in dmag)/len(dmag)),
    'remaining_vector_RMSE_after_analytic_plane_decomposition':math.sqrt(sum(residual)/len(residual)),
    'interpretation':'Offline geometric decomposition only; no fitted offset, revised actor, qualified visible-mask equivalence or new path score.'})
 m=json.loads(manifest.read_text());mismatch=[]
 for f in m['files']:
  p=base/f['path']
  if not p.exists() or p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:mismatch.append(f['path'])
 assert not mismatch
 result={'passed':True,'rows':len(rows),'implemented_glyph_scanline_boxes':len(geoms),
  'local_mask_area_centroid_matches_implemented_geometry':True,'numeral_top_above_body_m':float(top[0]-cfg['object']['height_m']/2),
  'old_scorer_height_discrepancy_m':.0002-(top[0]-cfg['object']['height_m']/2),'groups':output,
  'preserved_VIDEO10_files':len(m['files']),'preserved_VIDEO10_bytes':m['total_bytes'],'VIDEO10_mismatches':mismatch,
  'rendering_gate_not_promoted':True,'physical_rollouts':0}
 (audit/'independent_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();run(a.base,a.audit,a.manifest)
