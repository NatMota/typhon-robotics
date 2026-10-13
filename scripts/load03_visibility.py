"""Renderer-only fixture appearance removal; restores RGBA before returning."""
import hashlib,json,numpy as np
GEOMS=['fixture_floor']+['fixture_wall_'+str(i) for i in range(4)]
def state_fingerprint(env):
 sim=env.env.sim
 arrays={k:np.asarray(getattr(sim.data,k)).copy() for k in ['qpos','qvel','qacc_warmstart','act','ctrl','mocap_pos','mocap_quat']}
 h=hashlib.sha256()
 for k,v in arrays.items():h.update(k.encode());h.update(v.tobytes())
 h.update(str(float(sim.data.time)).encode())
 h.update(sim.model.get_xml().encode())
 return h.hexdigest()
def install(env,hidden):
 original=env.frame;model=env.env.sim.model
 ids=np.asarray([model.geom_name2id(n) for n in GEOMS],dtype=int)
 assert all(model.geom_matid[i]<0 for i in ids),'Fixture uses overriding material'
 initial=model.geom_rgba[ids].copy()
 def frame(camera='primary'):
  if not hidden:return original(camera)
  before=model.geom_rgba[ids].copy()
  try:
   model.geom_rgba[ids,3]=0.
   return original(camera)
  finally:model.geom_rgba[ids]=before
 env.frame=frame
 return {'ids':ids.tolist(),'names':GEOMS,'original_rgba':initial.tolist(),'hidden':hidden,'mechanism':'temporary renderer RGBA alpha zero; restored before return; no forward/step/physics change'}
