from pathlib import Path
import os,sys,json,gzip
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np
w=Path.cwd();sys.path[:0]=[str(w),str(w/'scripts')]
from live01_shadow_analyze import OUT,read,records,lines
from supervisor.feeding_loading.environment import FeedingLoadingEnv
summary=[]
for scenario in read(OUT/'protocol.json')['cases']:
 sid=scenario['scenario_id'];folder=OUT/('shadow-'+sid+'-audit')
 rows={r['control_step']:r for r in records(folder/'states.pkl.gz')}
 published=lines(folder/'publisher.jsonl.gz');env=FeedingLoadingEnv(scenario['config']);samples=[]
 try:
  env.reset(scenario['seed']);sim=env.env.sim;robot=env.env.robots[0]
  indices={'joint_pos':list(map(int,robot._ref_joint_pos_indexes)),'joint_vel':list(map(int,robot._ref_joint_vel_indexes)),
    'gripper_pos':list(map(int,robot._ref_gripper_joint_pos_indexes)),'eef_site_id':int(robot.eef_site_id)}
  for p in published:
   r=rows.get(p['control_step'])
   if r is None:continue
   state=r['state']
   sim.data.qpos[:]=state['qpos'];sim.data.qvel[:]=state['qvel']
   sim.data.qacc_warmstart[:]=state['qacc_warmstart'];sim.data.ctrl[:]=state['ctrl']
   # This is a separate post-hoc environment. NEVER called in executed N/S runs.
   sim.forward()
   recomputed=sim.data.site_xpos[robot.eef_site_id].copy()
   reported=np.array(p['telemetry']['eef_position_m'])
   fingers=state['qpos'][robot._ref_gripper_joint_pos_indexes]
   samples.append({'step':p['control_step'],'position_error_m':float(np.linalg.norm(recomputed-reported)),
    'reported_eef':reported.tolist(),'fk_of_recorded_integrated_qpos':recomputed.tolist(),
    'gripper_exact':bool(np.array_equal(fingers,np.asarray(p['telemetry']['gripper_joint_position_m']))),
    'joint_position_exact':bool(np.array_equal(state['qpos'][robot._ref_joint_pos_indexes],np.asarray(p['telemetry']['joint_position_rad']))),
    'actuator_torque_exact':bool(np.array_equal(state['qfrc_actuator'][robot._ref_joint_vel_indexes],np.asarray(p['telemetry']['joint_torque_nm'])))})
  d={'scenario':sid,'scope':'Offline FK on separate environment, not a controller or observer change. No policy inference or action execution.',
    'model_timestep_s':float(sim.model.opt.timestep),'indices':indices,'samples':len(samples),
    'maximum_position_gap_m':max(r['position_error_m'] for r in samples),
    'median_position_gap_m':float(np.median([r['position_error_m'] for r in samples])),
    'all_gripper_exact':all(r['gripper_exact'] for r in samples),
    'all_joint_position_exact':all(r['joint_position_exact'] for r in samples),
    'all_actuator_torque_exact':all(r['actuator_torque_exact'] for r in samples),
    'worst':max(samples,key=lambda r:r['position_error_m']),
    'verdict':'Telemetry reads existing MuJoCo site field; it is not simultaneous FK of logged integrated joint positions. Acquisition/integration phase needs an explicit label.'}
  (OUT/(sid+'-fk-audit.json')).write_text(json.dumps({'summary':d,'samples':samples},indent=2)+'\n')
  summary.append(d)
 finally:env.close()
(OUT/'telemetry-phase-audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
