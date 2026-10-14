"""Pinned semantic comparison for deterministic prefix reconstruction; no restore."""
import pathlib,json,hashlib,xml.etree.ElementTree as ET
import numpy as np,mujoco
import load05_crossover as b
import load05q_state as q
R=pathlib.Path(__file__).resolve().parents[1]
MANIFEST=R/'experiment-history/2026-10-14-load05r/state-manifest.json'
def normalize(x,excluded,path=''):
 if isinstance(x,dict):
  out={}
  for k,v in x.items():
   if k=='identity' and any(t in x for t in ['external_reference','callable','cycle_class']):
    excluded.append({'path':path+'/'+k,'reason':'process identity; structural role/code preserved','value':v});continue
   out[k]=normalize(v,excluded,path+'/'+k)
  return out
 if isinstance(x,list):return [normalize(v,excluded,path+'/'+str(i)) for i,v in enumerate(x)]
 return x
def record(env,dest,history,source):
 dest=pathlib.Path(dest);dest.mkdir(parents=True,exist_ok=False)
 sim=env.env.sim;m=sim.model._model;d=sim.data._data
 assert mujoco.__version__=='2.3.7'
 opts={k:int(getattr(m.opt,k)) for k in ['solver','integrator','jacobian','enableflags','disableflags','cone','noslip_iterations']}
 opts.update(timestep=float(m.opt.timestep),isDual=int(mujoco.mj_isDual(m)),isSparse=int(mujoco.mj_isSparse(m)))
 assert opts['solver']==2 and opts['integrator']==0 and not opts['isDual'] and not opts['isSparse']
 assert not m.nplugin and not m.ntendon and opts['enableflags']==0 and opts['disableflags']==0
 manifest=b.read(MANIFEST);arrays=b.data_arrays(d);omitted={k:arrays.pop(k) for k in manifest['excluded_native_fields'] if k in arrays}
 assert set(arrays)==set(manifest['retained_native_fields']),(set(arrays)^set(manifest['retained_native_fields']))
 assert arrays['efc_J'].size==int(d.nefc)*m.nv
 np.savez_compressed(dest/'arrays.npz',**arrays);np.savez_compressed(dest/'excluded-arrays.npz',**omitted)
 refs={i:type(v).__module__+'.'+type(v).__name__ for i,v in b.memo_for(env).items()}
 excluded=[]
 py=normalize({k:q.canonical(vars(v),refs,'/python/'+k) for k,v in b.objects(env).items()},excluded)
 contacts=[]
 for c in d.contact[:d.ncon]:
  item={k:np.asarray(getattr(c,k)).copy().tolist() for k in ['dist','pos','frame','includemargin','friction','solref','solreffriction','solimp','mu','dim','geom1','geom2','exclude','efc_address']}
  # H is the cone Hessian populated for active elliptic contact constraints.
  if int(c.efc_address)>=0 and opts['cone']==1:item['H_active']=np.asarray(c.H).reshape(6,6)[:int(c.dim),:int(c.dim)].copy().tolist()
  contacts.append(item)
 xml=sim.model.get_xml();(dest/'model.xml').write_text(xml)
 # Compare XML structure; physical fields such as statistic.meansize are retained.
 state={'schema':'load05r-semantic-v1','options':opts,'arrays':{k:q.canonical(v) for k,v in arrays.items()},'contacts':contacts,'python':py,'xml':q.canonical(ET.fromstring(xml)),
 'actor_context':{'history':q.canonical(history.copy()),'source':source,'query_index':2,'pending_cursor':0},'numpy_rng':q.canonical(np.random.get_state())}
 b.dump(dest/'state.json',state);b.dump(dest/'excluded-process-identities.json',excluded)
 b.dump(dest/'field-manifest.json',manifest)
 return state
def assert_equivalent(old,new,path):
 diffs=q.differences(old,new);b.dump(path,{'passed':not diffs,'differences':diffs,'atol':0,'rtol':0})
 assert not diffs,'Independent-prefix state differs; saved field-level differences'
