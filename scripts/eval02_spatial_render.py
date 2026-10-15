"""Decorative marker displacement during rendering only; restores geom_xpos."""
import numpy as np
NAMES=['staging_border_'+str(i) for i in range(4)]+['stage_orientation']
def install(env,dy):
 original=env.frame;sim=env.env.sim;model=sim.model;data=sim.data
 ids=np.asarray([model.geom_name2id(n) for n in NAMES],dtype=int)
 assert np.all(model.geom_contype[ids]==0) and np.all(model.geom_conaffinity[ids]==0)
 calls={'count':0,'restoration_checks':0}
 def frame(camera='primary'):
  calls['count']+=1
  if dy==0:return original(camera)
  before=np.asarray(data.geom_xpos).copy()
  try:
   data.geom_xpos[ids,1]+=dy
   return original(camera)
  finally:
   data.geom_xpos[:]=before
   assert np.array_equal(data.geom_xpos,before)
   calls['restoration_checks']+=1
 env.frame=frame
 return {'marker_names':NAMES,'ids':ids.tolist(),'render_delta_y_m':dy,'method':'temporary derived world geom_xpos displacement; no model or mocap change, forward, integration, image compositing or control change','calls':calls}
