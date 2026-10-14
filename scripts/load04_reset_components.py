"""Additional reset-only controller/render equivalence; preserves stopped-attempt evidence."""
from load04_common import *
import numpy as np,re
from load04_data import CaptureEnv
D=OUT/'reset-component-qualification';D.mkdir(exist_ok=False)
source=next(e for e in read(OLD/'data/loading-manifest.json')['episodes'] if e['source_demo_id']=='demo_0001')
CaptureEnv.physical_config=copy.deepcopy(source['config']);records=[]
def serial(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,(str,bool,int,float,type(None))):return v
 if isinstance(v,(list,tuple)) and all(isinstance(x,(bool,int,float,str)) for x in v):return list(v)
 raise TypeError(type(v).__name__)
for task in ['loading','feeding']:
 env=CaptureEnv(endpoint(source['config'],task))
 try:
  env.reset(131902);sim=env.env.sim;ctrl=env.env.robots[0].controller
  c={};skipped={}
  for k,v in vars(ctrl).items():
   try:c[k]=serial(v)
   except TypeError:skipped[k]=type(v).__name__
  dump(D/(task+'-controller.json'),c);dump(D/(task+'-controller-skipped.json'),skipped)
  arrays={}
  for k in ['cam_pos','cam_quat','cam_fovy','cam_ipd','cam_bodyid','cam_mode','cam_targetbodyid','light_pos','light_dir','light_diffuse','light_ambient','light_specular','light_castshadow','light_active','light_bodyid','geom_matid','geom_rgba','mat_rgba','mat_texid','tex_data','qpos0']:
   try:arrays[k]=np.asarray(getattr(sim.model,k)).copy()
   except AttributeError:pass
  np.savez_compressed(D/(task+'-render.npz'),**arrays)
  records.append((c,skipped,arrays,env.config))
 finally:env.close()
a,b=records
diff=[k for k in a[0] if a[0][k]!=b[0][k]]
render={k:np.array_equal(a[2][k],b[2][k]) for k in a[2]}
config_diff=[k for k in a[3] if a[3][k]!=b[3][k]]
report={'controller_fields_exact':not diff,'controller_fields':list(a[0]),'controller_differences':diff,'skipped_controller_handles':a[1],'compiled_render_equality':render,'config_differing_top_level_fields':config_diff,'no_policy_or_expert_execution':True,'reset_instances':2,'settling_ticks_per_reset':40,'time':time.time()}
dump(D/'report.json',report);print(json.dumps(report,indent=2))
assert not diff and all(render.values()) and config_diff==['task']
