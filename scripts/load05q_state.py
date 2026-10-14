"""Content signatures and evidence only. Restore assignments preserve LOAD05's procedure."""
import copy,json,hashlib,pathlib,types,xml.etree.ElementTree as ET
import numpy as np
import load05_crossover as base
ACTIVE=None
SCHEMA='load05q-content-state-v1'
class Unsupported(TypeError):pass
def canonical(x,refs=None,path='',seen=None):
 refs={} if refs is None else refs;seen=set() if seen is None else seen
 if id(x) in refs:return {'external_reference':refs[id(x)],'identity':id(x)}
 if isinstance(x,ET.Element):
  return {'xml_tag':x.tag,'attributes':dict(sorted(x.attrib.items())),'text':x.text,'tail':x.tail,'children':[canonical(c,refs,path+'/children/'+str(i),seen) for i,c in enumerate(x)]}
 if isinstance(x,ET.ElementTree):return {'xml_tree':canonical(x.getroot(),refs,path+'/root',seen)}
 if isinstance(x,np.ndarray):return {'array_dtype':str(x.dtype),'array_shape':list(x.shape),'array_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'values':x.tolist()}
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.dtype):return {'numpy_dtype':str(x)}
 if isinstance(x,range):return {'range':[x.start,x.stop,x.step]}
 if isinstance(x,slice):return {'slice':[x.start,x.stop,x.step]}
 if x is None or isinstance(x,(str,int,float,bool)):return x
 if isinstance(x,(bytes,bytearray)):return {'bytes_hex':bytes(x).hex()}
 if callable(x):
  return {'callable':getattr(x,'__module__','')+'.'+getattr(x,'__qualname__',type(x).__name__),'identity':id(x),'code_sha256':hashlib.sha256(x.__code__.co_code).hexdigest() if hasattr(x,'__code__') else None}
 if id(x) in seen:return {'cycle_class':type(x).__name__,'identity':id(x)}
 seen=seen|{id(x)}
 if isinstance(x,dict):return {str(k):canonical(v,refs,path+'/'+str(k),seen) for k,v in sorted(x.items(),key=lambda kv:str(kv[0]))}
 if isinstance(x,(tuple,list)):return [canonical(v,refs,path+'/'+str(i),seen) for i,v in enumerate(x)]
 if isinstance(x,set):return sorted([canonical(v,refs,path,seen) for v in x],key=lambda v:json.dumps(v,sort_keys=True))
 if hasattr(x,'__dict__'):return {'class':type(x).__module__+'.'+type(x).__name__,'state':canonical(vars(x),refs,path+'/state',seen)}
 raise Unsupported(path+': unsupported '+type(x).__module__+'.'+type(x).__name__)
def signature(env):
 refs={i:type(v).__module__+'.'+type(v).__name__ for i,v in base.memo_for(env).items()}
 py={k:canonical(vars(v),refs,'/python/'+k) for k,v in base.objects(env).items()}
 a={k:{'dtype':str(v.dtype),'shape':list(v.shape),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for k,v in base.data_arrays(env.env.sim.data._data).items()}
 out={'schema':SCHEMA,'physical':a,'python':py,'model_xml_sha256':hashlib.sha256(env.env.sim.model.get_xml().encode()).hexdigest()}
 return out
def differences(a,b,path=''):
 out=[]
 if type(a)!=type(b):return [{'path':path,'before':a,'after':b,'kind':'type'}]
 if isinstance(a,dict):
  for k in sorted(set(a)|set(b)):
   if k not in a or k not in b:out.append({'path':path+'/'+k,'kind':'missing','before':a.get(k),'after':b.get(k)})
   else:out.extend(differences(a[k],b[k],path+'/'+k))
 elif isinstance(a,list):
  if len(a)!=len(b):out.append({'path':path,'kind':'length','before':len(a),'after':len(b)})
  else:
   for i,(x,y) in enumerate(zip(a,b)):out.extend(differences(x,y,path+'/'+str(i)))
 elif a!=b:
  if isinstance(a,float) and np.isnan(a) and np.isnan(b):return []
  out.append({'path':path,'kind':'value','before':a,'after':b})
 return out
def record(env,d):
 d=pathlib.Path(d);d.mkdir(parents=True,exist_ok=False)
 arrays=base.data_arrays(env.env.sim.data._data);np.savez_compressed(d/'arrays.npz',**arrays)
 xml=env.env.sim.model.get_xml();(d/'model.xml').write_text(xml)
 s=signature(env);base.dump(d/'signature.json',s)
 # Persist XML instances separately, including child order and text.
 found=[]
 def walk(x,p=''):
  if isinstance(x,dict):
   if 'xml_tag' in x:found.append({'path':p,'content':x})
   for k,v in x.items():walk(v,p+'/'+k)
  elif isinstance(x,list):
   for i,v in enumerate(x):walk(v,p+'/'+str(i))
 walk(s);base.dump(d/'xml-records.json',found)
 return s
def restore(env,s):
 # Same assignments and copies as LOAD05; only the old final assertion becomes recorded comparison.
 env.env.sim.data._data=copy.copy(s['data'])
 memo=base.memo_for(env)
 for k,o in base.objects(env).items():
  state=copy.deepcopy(s['python'][k],memo);o.__dict__.clear();o.__dict__.update(state)
 np.random.set_state(s['rng'])
 before=base.read(ACTIVE/'before/signature.json');after=record(env,ACTIVE/'after')
 diffs=differences(before,after);saved_diffs=differences(before,s['signature'])
 old=np.load(ACTIVE/'before/arrays.npz');new=np.load(ACTIVE/'after/arrays.npz');numeric=[]
 for k in sorted(set(old.files)|set(new.files)):
  if k not in old.files or k not in new.files:numeric.append({'field':k,'missing':True});continue
  a,b=old[k],new[k]
  if a.shape!=b.shape or a.dtype!=b.dtype:numeric.append({'field':k,'shape_before':a.shape,'shape_after':b.shape,'dtype_before':str(a.dtype),'dtype_after':str(b.dtype)});continue
  if a.tobytes()!=b.tobytes():
   mask=~(a==b)
   if a.dtype.kind in 'fc':mask &= ~(np.isnan(a)&np.isnan(b))
   inds=np.argwhere(mask)
   vals=[{'index':i.tolist(),'before':a[tuple(i)].item(),'after':b[tuple(i)].item()} for i in inds[:32]]
   numeric.append({'field':k,'changed_elements':len(inds),'examples':vals})
 passed=not diffs and not saved_diffs and not numeric
 base.dump(ACTIVE/'comparison.json',{'passed':passed,'schema':SCHEMA,'differences':diffs,'snapshot_vs_before_differences':saved_diffs,'numerical_differences':numeric,'before_record_independent_disk_files':True})
 assert passed,'Restored content/state differs; inspect saved comparison'
def copy_tests(root):
 root=pathlib.Path(root);root.mkdir(parents=True,exist_ok=False);results={}
 e=ET.fromstring('<m xmlns:x="urn:check"><x:a k="1"> keep </x:a><b z="2"/></m>');c=copy.deepcopy(e)
 a=canonical(e);b=canonical(c);assert id(e)!=id(c) and a==b;results['deepcopy_equal']=True
 c[0].set('k','9');ds=differences(a,canonical(c));assert any(x['path'].endswith('/attributes/k') for x in ds);base.dump(root/'attribute-difference.json',ds);results['attribute_detected']=True
 c=copy.deepcopy(e);c[:]=list(reversed(list(c)));assert canonical(c)!=a;results['child_order_retained']=True
 c=copy.deepcopy(e);c[0].text='changed';assert canonical(c)!=a;results['text_retained']=True
 x=np.array([1.,2.]);y=x.copy();y[1]=3.;before=canonical({'state':x});after=canonical({'state':y});ds=differences(before,after);assert any(v['path'].endswith('/values/1') for v in ds)
 base.dump(root/'numerical-before.json',before);base.dump(root/'numerical-after.json',after);base.dump(root/'numerical-difference.json',ds);results['numerical_change_detected']=True
 try:canonical(object(),path='/unsupported')
 except Unsupported as ex:base.dump(root/'unsupported.json',{'explicit_error':str(ex)});results['unsupported_reported']=True
 else:raise AssertionError('opaque value accepted')
 base.dump(root/'complete.json',{'passed':True,'tests':results,'simulators_created':0})
