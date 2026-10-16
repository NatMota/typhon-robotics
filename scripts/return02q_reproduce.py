"""Reproduce all SAT outcomes from the small exported collision arrays; no MuJoCo needed."""
import sys,ast,itertools,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
A=Path(sys.argv[1]);destination=Path(sys.argv[2])
source=(A/'source/ctrl03_geometry.py').read_text()
fn=next(ast.get_source_segment(source,n)for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='gap')
ns={'np':np,'itertools':itertools,'ConvexHull':ConvexHull};exec(fn,ns)
g=json.loads((A/'compiled-geometry.json').read_text());rows=json.loads((A/'pair-comparisons.json').read_text())
maxerror=0.;count=0
for row in rows:
 parent=row['parent'];ref=json.loads((A/'source-inputs'/parent/'geometry-screen.json').read_text())['sampled_references'][row['reference_index']]
 d=g[parent]['geoms']
 a=np.asarray(d[row['gripper']]['grip_local_vertices'])@np.asarray(ref['wrist_rotation']).T+ref['nominal_xyz'if row['condition']=='nominal'else'shifted_xyz']
 b=np.asarray(d[row['obstacle']]['world_vertices']);value=ns['gap'](a,b)
 error=abs(value-row['gap_m']);maxerror=max(maxerror,error);assert error<1e-12
 cls='certified_separation'if value>row['contact_margin_m']+1e-9 else'convex_overlap'if value<-1e-9 else'numerically_unresolved'
 assert cls==row['classification'];count+=1
assert count==630
out={'passed':True,'comparisons':count,'max_gap_difference_m':maxerror,'geometry_sha256':hashlib.sha256((A/'compiled-geometry.json').read_bytes()).hexdigest(),'simulation_steps':0,'training':False,'requirements':'Python3,NumPy,SciPy; compiled arrays already exported'}
destination.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
