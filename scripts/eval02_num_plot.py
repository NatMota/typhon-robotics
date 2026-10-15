"""Plot numeric intervention pairs from saved proposals; no new inference."""
import os,json,subprocess
from pathlib import Path
import numpy as np
O=Path(os.environ.get('TYPHON_NUM_ROOT','/var/tmp/typhon-eval02-num-20261015'))
rows=json.loads((O/'comparisons.json').read_text())
base=["set encoding utf8","set grid lc rgb '#dddddd'","set xlabel 'Future time within chunk (s)'","set multiplot layout 2,2 title 'Numeric instruction swap, fixed G1 image: predicted targets, not executed' font ',15'"]
for model in ['GOAL','B']:
 cols=[np.arange(30)/20]
 for seed in [912171,912172]:
  r=next(x for x in rows if x['model']==model and x['seed']==seed)
  z=np.load(Path(r['new_directory'])/'paired-proposals.npz');cols.extend([z['original'][:,1],z['swapped'][:,1],(z['swapped'][:,1]-z['original'][:,1])*1000])
 data=O/(model+'-plot.tsv');np.savetxt(data,np.column_stack(cols),delimiter='\t')
 base+=["set title '"+model+": original versus swapped numeric goal'","set ylabel 'Predicted EEF Y (m)'","set key bottom right font ',8'"]
 curves=[]
 for column,color,dash,name in [(2,'#3366cc',1,'Original 912171'),(3,'#cc3344',2,'Swap 912171'),(5,'#228855',1,'Original 912172'),(6,'#bb8800',2,'Swap 912172')]:
  curves.append("'"+str(data)+"' using 1:"+str(column)+" with lines lw 2 dt "+str(dash)+" lc rgb '"+color+"' title '"+name+"'")
 base.append('plot '+', '.join(curves))
 base+=["set title '"+model+": signed change (swap - original)'","set ylabel 'Delta predicted Y (mm)'","set key top left","plot '"+str(data)+"' using 1:4 with lines lw 2 lc rgb '#3366cc' title '912171', '"+str(data)+"' using 1:7 with lines lw 2 lc rgb '#228855' title '912172', 0 with lines lc rgb '#777777' dt 3 notitle"]
base.append('unset multiplot')
for kind in ['png','svg']:
 terminal="set terminal pngcairo size 1400,850 enhanced font 'DejaVu Sans,11'" if kind=='png' else "set terminal svg size 1400,850 enhanced font 'DejaVu Sans,11'"
 gp=O/('numeric-sensitivity-'+kind+'.gp');gp.write_text(terminal+"\nset output '"+str(O/('numeric-sensitivity.'+kind))+"'\n"+'\n'.join(base)+'\n');subprocess.run(['/usr/bin/gnuplot',str(gp)],check=True)
print('Saved PNG and SVG from four saved proposal pairs')
