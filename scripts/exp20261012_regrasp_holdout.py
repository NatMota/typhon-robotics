"""Quarantined original six-state recovery stress test; no labels admitted to training."""
import copy,json,time,sys
from pathlib import Path
import numpy as np
import exp20261012_recovery_training as recovery
core=recovery.core
OUT=Path('/var/tmp/typhon-ost-sweep-20261012/regrasp-holdout-v3')
if not OUT.exists():
 OUT.mkdir();old=core.read(Path('/var/tmp/typhon-research-20261011/recovery-oracle/protocol.json'))
 spec=copy.deepcopy(old);spec.update(version='setdown-six-quarantined-states-20261012',variants=['setdown_regrasp'],training=False,created_before_execution=time.time(),qualification='Exact historical failure-state prefix; staged setdown/regrasp; original installed joint limits checked at every executed step. Contact paths still diagnostic, not globally collision-certified.',code_sha256=core.sha(__file__))
 core.save(OUT/'protocol.json',spec)
OriginalRecorder=core.Recorder
class JointRecorder(OriginalRecorder):
 def __init__(self,env,out):
  super().__init__(env,out);sim=env.env.sim;robot=env.env.robots[0];self.joint_indices=np.asarray(robot._ref_joint_pos_indexes);ids=[int(np.where(sim.model.jnt_qposadr==i)[0][0])for i in self.joint_indices];self.limits=sim.model.jnt_range[ids].copy();self.execution_limit_margins=[]
 def step(self,command):
  super().step(command);q=self.env.env.sim.data.qpos[self.joint_indices];margin=float(np.minimum(q-self.limits[:,0],self.limits[:,1]-q).min());self.execution_limit_margins.append(margin)
  if margin < -1e-6:raise RuntimeError('executed_joint_limit_violation')
 def close(self):
  super().close();core.save(self.out/'joint-audit.json',{'installed_limits':self.limits.tolist(),'minimum_margin_rad':min(self.execution_limit_margins)if self.execution_limit_margins else None,'samples':len(self.execution_limit_margins),'passed':bool(self.execution_limit_margins)and min(self.execution_limit_margins)>=-1e-6,'no_collision_certification':True})
core.Recorder=JointRecorder;core.graspaware=recovery.setdown
core.run(Path('/unused'),OUT,variant='setdown_regrasp')
rows=[]
for p in sorted(OUT.glob('*/setdown_regrasp/summary.json')):
 row=core.read(p);rows.append({'case':p.parent.parent.name,'summary':row,'joint':core.read(p.parent/'joint-audit.json')})
core.save(OUT/'summary-all.json',rows)
print('COMPLETE',len(rows),flush=True)
