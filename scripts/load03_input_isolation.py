
from load03_common import *
import numpy as np
p=read(OUT/'protocol.json');assert read(OUT/'render-qualification.json')['passed'];assert read(OUT/'input-qualification.json')['passed']
rows=[]
for scene in p['scenes']:
 for seed in p['seeds']:
  base=scene['source_demo_id']+'-seed'+str(seed)
  for task in ['feeding','loading']:
   a=np.load(OUT/'fixtures'/(base+'-visible')/('prepared-'+task+'.npz'));b=np.load(OUT/'fixtures'/(base+'-hidden')/('prepared-'+task+'.npz'))
   keys=['input_ids','image_mask','proprio','domain_id']
   assert a.files==b.files
   for k in keys:np.testing.assert_array_equal(a[k],b[k])
   assert not np.array_equal(a['image_input'],b['image_input'])
   # The last RGB-derived view is the crop; fixed initial crop content is exact.
   rows.append({'base':base,'task':task,'identical_keys':keys,'images_differ':True})
assert time.time()<read(OUT/'clock.json')['qualification_deadline']
dump(OUT/'visual-input-qualification.json',{'passed':True,'rows':rows,'time':time.time()})
