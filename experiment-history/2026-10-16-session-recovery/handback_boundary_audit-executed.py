import pathlib,json,zipfile,struct,ast,gzip,hashlib
r=pathlib.Path('/media/nm/Fast Drive/Typhon/research-20261015');a=r/'init10-lower-path'
def arr(path,name):
 b=zipfile.ZipFile(path).read(name+'.npy');assert b[:6]==b'\x93NUMPY';ver=b[6]
 if ver==1:n=struct.unpack('<H',b[8:10])[0];start=10
 else:n=struct.unpack('<I',b[8:12])[0];start=12
 h=ast.literal_eval(b[start:start+n].decode('latin1'));assert not h['fortran_order'];fmt={'<f4':'f','<f8':'d','<i8':'q','<i4':'i'}[h['descr']];body=b[start+n:];v=list(struct.unpack('<'+str(len(body)//struct.calcsize(fmt))+fmt,body));return h,v
out=[]
for p in (a/'reference-inputs').iterdir():
 e=json.loads((p/'episode.json').read_text());ch,c=arr(p/'predictions.npz','chunks');hh,h=arr(p/'predictions.npz','histories')
 with gzip.open(p/'controller_trace.jsonl.gz','rt')as f:rows=[json.loads(s) for s in f]
 assert ch['shape'][1:]==(30,20)
 for i in range(150):assert rows[i]['target']==c[i*20:(i+1)*20],(p.name,i)
 row={'case':p.name,'chunks_header':ch,'history_header':hh,'branch_before':rows[59]['step'],'first_segment_action':rows[60]['step'],'last_segment_action':rows[119]['step'],'next_action':rows[120]['step'],'next_query_index':4,'next_query_before_step':rows[119]['step'],'native_last_target':c[119*20:120*20],'native_next_history':h[4*20:5*20],'source_episode_sha256':hashlib.sha256((p/'episode.json').read_bytes()).hexdigest(),'source_predictions_sha256':hashlib.sha256((p/'predictions.npz').read_bytes()).hexdigest(),'episode_policy_actions':e['policy_actions_executed'],'policy_actions_used_before_handback':120,'remaining_policy_actions':e['policy_actions_executed']-120,'original_final_control_step':e['control_step_end'],'original_passive_settle':e['control_step_end']-e['control_step_start']-e['policy_actions_executed']}
 row['native_history_equals_last_raw_target']=row['native_next_history']==row['native_last_target']
 out.append(row)
print(json.dumps(out,indent=2))
(r/'session-closeout-20261016/handback-boundaries.json').write_text(json.dumps({'prepared_only':True,'executions':0,'method':'Read saved NPY arrays with Python standard library; compare first 150 native trace targets exactly to saved chunks. No simulation/model calls.','cases':out},indent=2)+'\n')
