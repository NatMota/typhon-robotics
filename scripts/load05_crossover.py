"""LOAD05 pre-contact crossover. No actor truth; frozen 90-tick acquisition diagnostic."""
import os
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import pathlib,sys,json,time,copy,gzip,hashlib,pickle,subprocess,traceback
import numpy as np
from scipy.spatial.transform import Rotation
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from supervisor.feeding_loading.environment import FeedingLoadingEnv
from supervisor.baseline.xvla import to_action,XvlaPolicy
from motion_pose_diagnostics import details
from load03_visibility import install
OUT=pathlib.Path('/var/tmp/typhon-load05-20261014');SRC=pathlib.Path('/var/tmp/typhon-load04-20261014')
def read(p):return json.loads(pathlib.Path(p).read_text())
def dump(p,v):p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def limit(name):assert time.time()<read(OUT/'clock.json')[name],name+' exceeded'
def tilt(q):
 x,y,z,w=np.asarray(q)[[1,2,3,0]]
 return float(np.degrees(np.arccos(np.clip(1-2*(x*x+y*y),-1,1))))
def eqobs(obs,io,i):
 for k in ['primary','secondary']:np.testing.assert_array_equal(obs['rgb'][k],io[k][i])
 for k in ['eef_position_m','eef_quat_xyzw']:np.testing.assert_array_equal(obs['proprio'][k],io[k][i])
def compare(d,old):
 for k in ['eef_xyz','joint_positions','controller_ee_xyz','controller_goal_xyz','controller_goal_rotation','controller_torques']:
  np.testing.assert_allclose(d[k],old[k],rtol=0,atol=1e-6)
 for k in ['object_position_m','eef_position_m','gripper_joint_position_m']:
  np.testing.assert_allclose(d['object_truth'][k],old['object_truth'][k],rtol=0,atol=1e-6)
 qa=np.asarray(d['object_truth']['object_quat_wxyz'])[[1,2,3,0]];qb=np.asarray(old['object_truth']['object_quat_wxyz'])[[1,2,3,0]]
 assert (Rotation.from_quat(qa)*Rotation.from_quat(qb).inv()).magnitude()<np.radians(1e-4)
 assert d['object_truth']['grasp_contact']==old['object_truth']['grasp_contact']
 assert d['object_truth']['bilateral_grasp_contact']==old['object_truth']['bilateral_grasp_contact']
def data_arrays(data):
 out={}
 for k in dir(data):
  if k.startswith('_'):continue
  try:v=getattr(data,k)
  except Exception:continue
  if isinstance(v,np.ndarray) and v.dtype.kind not in 'OV':out[k]=v.copy()
  elif isinstance(v,(float,int)):out[k]=np.asarray(v)
 return out
def objects(env):
 robot=env.env.robots[0]
 return {'wrapper':env,'environment':env.env,'robot':robot,'controller':robot.controller,'gripper':robot.gripper}
def memo_for(env):
 sim=env.env.sim;obj=objects(env);values=list(obj.values())+[sim,sim.data,sim.model,sim._render_context_offscreen]
 for o in [env.env,env.env.robots[0]]:
  for k in ['model','robot_model','mujoco_arena','mujoco_objects','cube','placement_initializer','viewer']:
   v=getattr(o,k,None)
   if v is not None:values.append(v)
 return {id(v):v for v in values}
def canon(x,refs,seen=None):
 seen=set() if seen is None else seen
 if id(x) in refs:return {'external':refs[id(x)]}
 if isinstance(x,np.ndarray):return {'dtype':str(x.dtype),'shape':x.shape,'sha':hashlib.sha256(x.tobytes()).hexdigest()}
 if isinstance(x,np.generic):return x.item()
 if x is None or isinstance(x,(str,int,float,bool)):return x
 if callable(x):return {'callable':getattr(x,'__module__','')+'.'+getattr(x,'__qualname__',type(x).__name__)}
 if id(x) in seen:return {'cycle':type(x).__name__}
 seen=seen|{id(x)}
 if isinstance(x,dict):return {str(k):canon(v,refs,seen) for k,v in sorted(x.items(),key=lambda kv:str(kv[0]))}
 if isinstance(x,(tuple,list)):return [canon(v,refs,seen) for v in x]
 if isinstance(x,set):return sorted([canon(v,refs,seen) for v in x],key=str)
 if hasattr(x,'__dict__'):return {'class':type(x).__module__+'.'+type(x).__name__,'state':canon(vars(x),refs,seen)}
 return {'static_type':type(x).__module__+'.'+type(x).__name__,'value':str(x)}
def signature(env):
 refs={i:type(v).__name__ for i,v in memo_for(env).items()}
 p={k:canon(vars(v),refs) for k,v in objects(env).items()}
 a={k:canon(v,{}) for k,v in data_arrays(env.env.sim.data._data).items()}
 # Physical data covers warm-start, actuator, derived and solver arrays; Python covers controller and caches.
 return {'physical':a,'python':p,'sha':hashlib.sha256(json.dumps([a,p],sort_keys=True).encode()).hexdigest()}
def snapshot(env):
 memo=memo_for(env)
 return {'data':copy.copy(env.env.sim.data._data),'python':{k:copy.deepcopy(vars(v),memo) for k,v in objects(env).items()},'rng':np.random.get_state(),'signature':signature(env)}
def restore(env,s):
 env.env.sim.data._data=copy.copy(s['data'])
 memo=memo_for(env)
 for k,o in objects(env).items():
  state=copy.deepcopy(s['python'][k],memo);o.__dict__.clear();o.__dict__.update(state)
 np.random.set_state(s['rng'])
 assert signature(env)==s['signature'],'full copied state restoration differs'
def rich(env,target,command):
 d=details(env,target);sim=env.env.sim;rob=env.env.robots[0];o=sim.data.body_xpos[env.env.cube_body_id].copy();ro=sim.data.body_xmat[env.env.cube_body_id].reshape(3,3).copy()
 rg=sim.data.site_xmat[rob.eef_site_id].reshape(3,3).copy();g=sim.data.site_xpos[rob.eef_site_id].copy()
 d.update(command=np.asarray(command).tolist(),object_tilt_deg=tilt(d['object_truth']['object_quat_wxyz']),R_GO=(rg.T@ro).tolist(),p_GO=(rg.T@(o-g)).tolist())
 d['finger_geoms']=[]
 for name in rob.gripper.contact_geoms:
  i=sim.model.geom_name2id(name);pos=sim.data.geom_xpos[i].copy();b=sim.model.geom_bodyid[i]
  d['finger_geoms'].append({'name':name,'world_position':pos.tolist(),'object_relative_offset':(ro.T@(pos-o)).tolist(),'body_velocity':sim.data.get_body_xvelp(sim.model.body_id2name(b)).tolist()})
 d['contacts_full']=[{'geom1':sim.model.geom_id2name(c.geom1),'geom2':sim.model.geom_id2name(c.geom2),'world_position':c.pos.tolist(),'object_local_position':(ro.T@(c.pos-o)).tolist(),'normal_world':np.asarray(c.frame).reshape(3,3)[0].tolist(),'distance':float(c.dist)} for c in sim.data.contact[:sim.data.ncon]]
 return d
class Worker(XvlaPolicy):
 def __init__(self,name):
  e=read(SRC/'trained-models.json')[name];assert sha(e['checkpoint'])==e['sha256']
  self.log=open(OUT/('worker-'+name+'.log'),'w')
  self.process=subprocess.Popen([str(R/'.venv-xvla/bin/python'),'scripts/load05_worker.py','--checkpoint',e['checkpoint'],'--probe',e['probe'],'--verification',str(OUT/('reload-'+name+'.json'))],cwd=R,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,env={**os.environ,'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
  assert self._receive().get('ready')
def request(worker,obs,history,source,index,dest,prepare=False,expected=False):
 p={'op':'prepare' if prepare else 'predict','images':[obs['rgb']['primary'],obs['rgb']['secondary']],'proprio':history,'instruction':source['scenario']['instruction_override'],'audit_directory':str(dest),'audit_index':index,'saved_noise':str(source['path']/'input-audit'/('noise-%03d.npz'%index))}
 if expected:p['expected_all']=read(source['path']/'input-audit'/('input-%03d.json'%index))['input_hashes']
 return worker._request(p)
def main():
 limit('qualification_deadline');rows=read(SRC/'analysis/episodes.json');sources={}
 for m in ['C','B']:
  row=next(x for x in rows if x['model']==m and x['visibility']=='visible' and x['scene']=='demo_0012' and x['requested_task']=='loading' and x['seed']==912171)
  path=SRC/'evaluation'/m/'visible'/'demo_0012'/row['id'];tr=[json.loads(x) for x in gzip.open(path/'controller_trace.jsonl.gz','rt')];io=np.load(path/'policy_io.npz');history=np.load(path/'raw_predictions.npz')['histories']
  sources[m]={'path':path,'trace':tr,'io':io,'history':history,'scenario':read(path/'scenario.json'),'episode':read(path/'episode.json')}
 first={m:next(x['step'] for x in s['trace'] if x['object_truth']['grasp_contact']) for m,s in sources.items()}
 start=sources['C']['episode']['initial_truth']['control_steps'];bound=max(k for k in range(start,start+1440,30) if k<min(first.values()));index=(bound-start)//30
 assert index>=1
 dump(OUT/'selection.json',{'first_contacts':first,'branch_control_tick':bound,'query_index':index,'rule':'last shared native query boundary strictly before first gripper-brick contact in either source','source_files':{m:{'episode':sha(s['path']/'episode.json'),'policy_io':sha(s['path']/'policy_io.npz')} for m,s in sources.items()}})
 envs={};states={};workers={};checks=[]
 try:
  for m,s in sources.items():
   limit('qualification_deadline');env=FeedingLoadingEnv(s['scenario']['config']);envs[m]=env;env.reset(s['scenario']['seed']);install(env,False)
   assert env.truth()==s['episode']['initial_truth']
   env.capture_callback=lambda e:(e.frame('primary'),e.frame('secondary'))
   env.frame('primary');env.frame('secondary');env.frame('primary');env.frame('secondary')
   trace_by={x['step']:x for x in s['trace']}
   for i in range((bound-start)+30):
    if i%30==0:
     obs=env.observe();eqobs(obs,s['io'],i//30)
    if env.steps==bound:
     t=env.truth();initial=s['episode']['initial_truth'];assert not t['grasp_contact'] and tilt(t['object_quat_wxyz'])<10
     assert np.linalg.norm(np.asarray(t['object_position_m'])-initial['object_position_m'])<=.002
     assert np.linalg.norm(t['object_linear_velocity_m_s'])<=.01 and np.linalg.norm(t['object_angular_velocity_rad_s'])<=.1
     state=snapshot(env);states[m]=state;dump(OUT/m/'restored-context.json',state['signature']);np.savez_compressed(OUT/m/'mjdata-arrays.npz',**data_arrays(env.env.sim.data._data))
    target=s['io']['model_actions'].reshape(-1,20)[i];obs=env.observe();command=to_action(target,obs,env.episode_config);env.step(command);d=details(env,target);compare(d,trace_by[env.steps])
   assert m in states;restore(env,states[m]);obs=env.observe();eqobs(obs,s['io'],index)
   workers[m]=Worker(m);request(workers[m],obs,s['history'][index].copy(),s,index,OUT/m/'prepared-qualification',prepare=True,expected=True)
   checks.append({'origin':m,'prefix_and_first_chunk_replayed':True,'full_copy_restore_exact':True,'prepared_input_exact':True,'original_hidden_state_direct_comparison_available':False,'trace_samples':bound-start+30})
   dump(OUT/'qualification-progress.json',checks)
  limit('qualification_deadline');dump(OUT/'qualification.json',{'passed':True,'records':checks,'qualified_at':time.time(),'limitations':'Original full internal snapshots were not recorded. Exact full-context checks are against recovered replay state; original RGB/prepared tensors and saved dynamic/controller trace are compared directly.'})
  same=states['C']['signature']==states['B']['signature'] and np.array_equal(sources['C']['history'][index],sources['B']['history'][index])
  cases=[('C','C'),('B','B')] if same else [('C','C'),('B','B'),('C','B'),('B','C')]
  summaries=[]
  for origin,model in cases:
   limit('execution_deadline');s=sources[origin];env=envs[origin];restore(env,states[origin]);history=s['history'][index].copy();dest=OUT/'continuations'/(origin+'-to-'+model);dest.mkdir(parents=True,exist_ok=False)
   obs=env.observe();eqobs(obs,s['io'],index);request(workers[model],obs,history,s,index,dest/'start-prepared',prepare=True,expected=True)
   import imageio.v2 as imageio
   writer=imageio.get_writer(str(dest/'rollout.mp4'),fps=20,codec='libx264');records=[];chunks=[]
   try:
    for j in range(3):
     obs=env.observe();res=request(workers[model],obs,history,s,index+j,dest/'input-audit');chunk=res['actions'];chunks.append(chunk.copy());history[:9]=chunk[-1,:9]
     if origin==model:np.testing.assert_array_equal(chunk,s['io']['model_actions'][index+j])
     for target in chunk:
      obs=env.observe();command=to_action(target,obs,env.episode_config);env.step(command);d=rich(env,target,command);records.append(d)
      if origin==model:compare(d,{x['step']:x for x in s['trace']}[env.steps])
      writer.append_data(np.concatenate([env.frame('primary'),env.frame('secondary')],axis=1))
   finally:writer.close()
   with gzip.open(dest/'trace.jsonl.gz','wt') as f:
    for row in records:f.write(json.dumps(row)+'\n')
   np.savez_compressed(dest/'predictions.npz',chunks=np.asarray(chunks))
   initial=s['episode']['initial_truth'];supported=[x for x in records if x['object_truth']['bilateral_grasp_contact'] and x['object_truth']['object_position_m'][2]-initial['object_position_m'][2]>=.01]
   summary={'origin_state':origin,'checkpoint':model,'steps':len(records),'max_object_tilt_deg':max(x['object_tilt_deg'] for x in records),'final_object_tilt_deg':records[-1]['object_tilt_deg'],'first_tilt_over15':next((x['step'] for x in records if x['object_tilt_deg']>15),None),'first_bilateral':next((x['step'] for x in records if x['object_truth']['bilateral_grasp_contact']),None),'first_supported_lift':supported[0]['step'] if supported else None,'supported_lift_tilt_deg':supported[0]['object_tilt_deg'] if supported else None,'diagonal_reproduced':True if origin==model else None,'final_placement_evaluation':False}
   dump(dest/'summary.json',summary);summaries.append(summary);dump(OUT/'summary.json',summaries);print(json.dumps(summary),flush=True)
  dump(OUT/'complete.json',{'passed':True,'continuations':len(summaries),'branch_control_tick':bound,'distinct_states':not same,'finished':time.time(),'training':False})
 finally:
  for w in workers.values():w.close()
  for env in envs.values():env.close()
if __name__=='__main__':
 try:main()
 except BaseException:
  dump(OUT/'STOPPED.json',{'stopped':True,'time':time.time(),'error':traceback.format_exc(),'outcome_retries':0});raise
