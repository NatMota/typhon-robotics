"""Reset-only diagnostic after stopped LOAD04. No trained policy or expert trajectory."""
from load04_common import *
import numpy as np,difflib
from load04_data import CaptureEnv
D=OUT/'reset-diagnostic';D.mkdir(exist_ok=False)
source=next(e for e in read(OLD/'data/loading-manifest.json')['episodes'] if e['source_demo_id']=='demo_0001')
cfg=source['config'];CaptureEnv.physical_config=copy.deepcopy(cfg)
records=[]
for task in ['loading','feeding']:
 env=CaptureEnv(endpoint(cfg,task))
 try:
  env.reset(131902);sim=env.env.sim
  arrays={k:np.asarray(getattr(sim.data,k)).copy() for k in ['qpos','qvel','qacc_warmstart','act','ctrl','mocap_pos','mocap_quat','qacc','qfrc_applied','xfrc_applied','qfrc_bias']}
  arrays.update(time=np.asarray(sim.data.time))
  np.savez_compressed(D/(task+'-state.npz'),**arrays)
  xml=sim.model.get_xml();(D/(task+'-model.xml')).write_text(xml)
  model={k:np.asarray(getattr(sim.model,k)).copy() for k in ['body_mass','body_inertia','body_pos','body_quat','geom_pos','geom_quat','geom_size','geom_friction','geom_solref','geom_solimp','geom_rgba','jnt_pos','jnt_axis','jnt_range','dof_damping','dof_frictionloss','actuator_gainprm','actuator_biasprm']}
  np.savez_compressed(D/(task+'-model-arrays.npz'),**model)
  records.append({'task':task,'state':arrays,'model':model,'xml':xml})
 finally:env.close()
a,b=records;diff=[]
for line in difflib.unified_diff(a['xml'].splitlines(),b['xml'].splitlines(),fromfile='loading',tofile='feeding',n=2):diff.append(line)
(D/'model.diff').write_text('\n'.join(diff)+'\n')
report={'no_policy_or_expert_executions':True,'reset_only_instances':2,'ordinary_reset_settling_ticks':40,'component_equality':{k:np.array_equal(a['state'][k],b['state'][k]) for k in a['state']},'model_array_equality':{k:np.array_equal(a['model'][k],b['model'][k]) for k in a['model']},'xml_exact':a['xml']==b['xml'],'xml_diff_lines':len(diff),'state_max_diffs':{k:float(np.max(np.abs(a['state'][k]-b['state'][k]))) if a['state'][k].size else 0. for k in a['state']},'time':time.time()}
dump(D/'report.json',report);print(json.dumps(report,indent=2));print('\n'.join(diff[:80]))
