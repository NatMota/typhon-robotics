"""Eight physical executions only: four old loading replays and four new feeding experts."""
from load04_common import *
import numpy as np
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.feeding_loading.evaluation import score_episode
from supervisor.adaptation.actions import absolute_target,proprio_vector
from supervisor.adaptation.observations import object_crop
from supervisor.baseline.xvla import task_prompt
from load03_visibility import GEOMS,state_fingerprint
import m15_execute

class CaptureEnv(FeedingLoadingEnv):
 physical_config=None
 active=None
 def __init__(self,requested):
  self.enabled=False;self.requested=copy.deepcopy(requested)
  super().__init__(copy.deepcopy(type(self).physical_config))
  self.packet={k:[] for k in ['primary_visible','secondary_visible','primary_hidden','secondary_hidden','frame_action_indices','targets','proprio']}
  self.crop_audit=[];self.post=[];self.last_target=None;self.counter=0;type(self).active=self
 def reset(self,seed=None):
  self.enabled=False;observation=super().reset(seed)
  self.config=copy.deepcopy(self.requested);self.episode_config=copy.deepcopy(self.requested)
  self.initial_truth=self.truth();sim=self.env.sim
  self.initial_arrays={k:getattr(sim.data,k).copy() for k in ['qpos','qvel','mocap_pos','mocap_quat']}
  self.initial_arrays.update({c:self.frame(c) for c in ['primary','secondary']},proprio=proprio_vector(self.observe()))
  self.initial_fingerprint=state_fingerprint(self);self.enabled=True
  return self.observe()
 def hidden_rgb(self):
  sim=self.env.sim;ids=[sim.model.geom_name2id(n) for n in GEOMS]
  rgba=sim.model.geom_rgba[ids].copy();qpos=sim.data.qpos.copy();qvel=sim.data.qvel.copy();t=sim.data.time
  try:
   sim.model.geom_rgba[ids,3]=0
   result={c:self.frame(c) for c in ['primary','secondary']}
  finally:sim.model.geom_rgba[ids]=rgba
  assert np.array_equal(sim.data.qpos,qpos) and np.array_equal(sim.data.qvel,qvel) and sim.data.time==t
  assert np.array_equal(sim.model.geom_rgba[ids],rgba)
  return result
 def step(self,command):
  if not self.enabled:return super().step(command)
  obs=self.observe()
  if self.counter%4==0:
   hidden=self.hidden_rgb()
   for c in ['primary','secondary']:
    self.packet[c+'_visible'].append(obs['rgb'][c].copy());self.packet[c+'_hidden'].append(hidden[c])
   self.packet['frame_action_indices'].append(self.counter)
   prop=proprio_vector(obs)
   if self.last_target is not None:prop[:9]=self.last_target[:9]
   self.packet['proprio'].append(prop)
   vc,vm=object_crop(obs['rgb']['primary']);hc,hm=object_crop(hidden['primary'])
   self.crop_audit.append({'action_index':self.counter,'visible':vm,'hidden':hm,'crop_metadata_equal':vm==hm,
    'crop_pixel_changes':None if vc is None or hc is None else int(np.any(vc!=hc,axis=2).sum())})
  target=absolute_target(command,obs,self.config);self.packet['targets'].append(target);self.last_target=target
  result=super().step(command);self.post.append(self.truth());self.counter+=1
  return result
 def save(self,identity,source,result,origin):
  p=OUT/'data'/(identity+'.npz')
  arrays={k:np.asarray(v,np.float32 if k in ['targets','proprio'] else None) for k,v in self.packet.items()}
  np.savez_compressed(p,**arrays)
  initial=OUT/'data'/('initial-'+identity+'.npz');np.savez_compressed(initial,**self.initial_arrays)
  dump(OUT/'data'/('crop-'+identity+'.json'),self.crop_audit)
  return {'id':identity,'source_demo_id':source,'task':self.config['task']['type'],'split':'load04_train','absolute_path':str(p),'sha256':sha(p),
    'config':self.config,'instruction':task_prompt(self.config['task']),'actions':self.counter,'frames':len(self.packet['proprio']),
    'initial_archive':str(initial),'initial_fingerprint':self.initial_fingerprint,'initial_truth':self.initial_truth,
    'success':result['success'],'source_path':origin,'action_labels':'privileged geometric expert robot commands, not measured human actions',
    'views':'both renderings from identical pre-action physics; no visibility-specific execution','crop_audit':str(OUT/'data'/('crop-'+identity+'.json'))}

def run():
 clock=read(OUT/'clock.json');entries=[];attempts=[];pairs=[]
 old=read(OLD/'data/loading-manifest.json')['episodes']
 try:
  for parent in SOURCES:
   e=next(x for x in old if x['source_demo_id']==parent);physical=copy.deepcopy(e['config'])
   source=Path(e['source_path']);orig=read(source/'episode.json')
   assert orig['success'];CaptureEnv.physical_config=physical
   pair=[]
   for task in ['loading','feeding']:
    assert time.time()<clock['data_deadline'],'Data deadline'
    identity='load04-'+parent+'-'+task
    requested=endpoint(physical,task)
    attempts.append({'source':parent,'task':task,'started':time.time(),'status':'started'})
    dump(OUT/'data/execution-ledger.json',attempts)
    if task=='loading':
     env=CaptureEnv(requested);errors=[]
     try:
      env.reset(orig['seed']);assert env.initial_truth==orig['initial_truth']
      for line in (source/'robot_actions.jsonl').read_text().splitlines():
       row=json.loads(line);env.step(row['normalized_command'])
       errors.append(float(np.max(np.abs(np.asarray(env.truth()['object_position_m'])-row['post_action_truth']['object_position_m']))))
      assert max(errors)<1e-8,('replay mismatch',parent,max(errors))
      scored=score_episode(env.initial_truth,env.truth(),env.post[-24:],requested,orig['attempts'])
      result={'success':scored['success'],'evaluation':scored,'replay_max_position_error':max(errors),'source_action_sha256':sha(source/'robot_actions.jsonl')}
      entry=env.save(identity,parent,result,str(source))
     finally:env.close()
    else:
     sc={'scenario_id':identity,'cohort':'load04_training_only','source_demo_id':parent,'category':'normal_feeding','seed':orig['seed'],
       'reconstruction_confidence':'existing personal-derived initial state; supplied feeding goal; matched loading physical scene',
       'initial_sim':{'position_xy_m':requested['initial_state']['position_xy_m'],'yaw_rad':requested['initial_state']['yaw_rad']},
       'target_sim':{'position_xy_m':requested['task']['target_position_xy_m'],'yaw_rad':requested['task']['target_yaw_rad']}}
     dump(OUT/'data/configs'/(identity+'.json'),requested)
     original=m15_execute.FeedingLoadingEnv;m15_execute.FeedingLoadingEnv=CaptureEnv
     try:result=m15_execute.execute(sc,OUT/'data')
     finally:m15_execute.FeedingLoadingEnv=original
     env=CaptureEnv.active
     entry=env.save(identity,parent,result,str(OUT/'data/episode_artifacts'/identity))
    dump(OUT/'data'/(identity+'-qualification.json'),result)
    attempts[-1].update(status='passed' if result['success'] else 'failed',finished=time.time(),success=result['success'])
    dump(OUT/'data/execution-ledger.json',attempts)
    assert result['success'],('strict expert qualification failed',parent,task)
    pair.append(entry);entries.append(entry);dump(OUT/'data/manifest.json',{'episodes':entries})
   a,b=[np.load(x['initial_archive']) for x in pair]
   assert a.files==b.files
   for k in a.files:np.testing.assert_array_equal(a[k],b[k],err_msg=parent+'/'+k)
   assert pair[0]['initial_fingerprint']==pair[1]['initial_fingerprint']
   pairs.append({'source':parent,'initial_native_rgb_proprio_physics_exact':True,'initial_fingerprint':pair[0]['initial_fingerprint']})
  assert len(entries)==8 and len(attempts)==8
  dump(OUT/'data/generation-gate.json',{'passed':True,'episodes':8,'physical_executions':8,'matched_pairs':pairs,'no_replacements_or_extra_replays':True,'time':time.time()})
 except BaseException:
  import traceback
  dump(OUT/'data/generation-gate.json',{'passed':False,'episodes_completed':len(entries),'executions_started':len(attempts),'error':traceback.format_exc(),'stopped_without_replacement':True,'time':time.time()})
  raise
if __name__=='__main__':run()
