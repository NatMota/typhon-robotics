"""Render saved EEF targets using installed gnuplot; no inference."""
import os,json,subprocess
from pathlib import Path
import numpy as np
O=Path(os.environ.get('TYPHON_FIT_ROOT','/var/tmp/typhon-eval02-fit-20261015'))
rows=json.loads((O/'summary.json').read_text());specs=[]
for packet in ['approach','postlift']:
 target=np.load(O/(packet+'.npz'))['targets']
 cols=[np.arange(30)/20,target[:,:3]];names=['Expert']
 for model in ['GOAL','FL','B']:
  for seed in [912171,912172]:
   row=next(r for r in rows if (r['model'],r['packet'],r['seed'])==(model,packet,seed))
   cols.append(np.load(Path(row['directory'])/'proposal.npz')['predicted'][:,:3]);names.append(model+' '+str(seed))
 data=O/(packet+'-plot.tsv');np.savetxt(data,np.column_stack(cols),delimiter='\t')
 specs.append((packet,data))
base=["set encoding utf8","set grid lc rgb '#dddddd'","set border lc rgb '#555555'","set tics textcolor rgb '#333333'","set xlabel 'Future time within chunk (s)'","set ylabel 'World coordinate (m)'","set multiplot layout 2,3 title 'Predicted EEF targets, not executed; solid/dashed lines show paired noise seeds' font ',16'"]
colors=['#111111','#3366cc','#3366cc','#dd8800','#dd8800','#cc3344','#cc3344']
for row,(packet,data) in enumerate(specs):
 for axis in range(3):
  base.append("set title '"+packet+" : "+'XYZ'[axis]+"'")
  base.append("set key top left font ',9'" if row==0 and axis==0 else "unset key")
  curves=[]
  for i,name in enumerate(['Expert','GOAL','GOAL seed2','FL','FL seed2','B','B seed2']):
   curves.append("'"+str(data)+"' using 1:"+str(2+3*i+axis)+" with lines lw "+('2.5' if i==0 else '1.8')+" dt "+('2' if i in [2,4,6] else '1')+" lc rgb '"+colors[i]+"' title '"+name+"'")
  base.append("plot "+", ".join(curves))
base.append('unset multiplot')
for kind in ['png','svg']:
 terminal="set terminal pngcairo size 1650,900 enhanced font 'DejaVu Sans,11'" if kind=='png' else "set terminal svg size 1650,900 enhanced font 'DejaVu Sans,11'"
 gp=O/('proposal-targets-'+kind+'.gp');gp.write_text(terminal+"\nset output '"+str(O/('proposal-targets.'+kind))+"'\n"+'\n'.join(base)+'\n')
 subprocess.run(['/usr/bin/gnuplot',str(gp)],check=True)
print('PNG and SVG plots rendered from saved arrays')
