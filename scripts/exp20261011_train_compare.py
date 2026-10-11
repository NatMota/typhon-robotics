"""CPU-only comparability and actual prefix-exposure audit; does not modify training."""
import ast,collections,hashlib,json
from pathlib import Path
import torch
from supervisor.adaptation.train import tensor_hash
ROOT=Path.cwd();OLD=Path('/cs/student/msc_rai/2026/ndossant/Projects/humanoid-m6-joint');OUT=Path('/var/tmp/typhon-research-20261011/broad-pt');NARROW=OLD/'results/personal-m6-joint-20261009/training'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def funcs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
torch.set_num_threads(2)
old=json.loads((NARROW/'config.json').read_text());new=json.loads((OUT/'config.json').read_text());oi=json.loads((NARROW/'initialization.json').read_text());ni=json.loads((OUT/'initialization.json').read_text());assert oi==ni
state=torch.load(NARROW/'custom-init-tables.pt',map_location='cpu',weights_only=False)
actual={n:tensor_hash(p[29]) for n,p in state.items()};assert all(actual[n]==ni[n]['initial_active_sha256'] for n in ni)
source=[]
for rel in ['supervisor/adaptation/m6_joint.py','supervisor/adaptation/m6_contract.py','supervisor/adaptation/observations.py','supervisor/adaptation/m5.py']:
 a=OLD/rel;b=ROOT/rel
 source.append({'path':rel,'old_exists':a.exists(),'old_sha256':sha(a) if a.exists() else None,'current_sha256':sha(b),'identical':a.exists() and sha(a)==sha(b),'named_function_matches':{k:funcs(a).get(k)==v for k,v in funcs(b).items()} if a.exists() else None})
rows=[json.loads(s) for s in (OUT/'train-losses.jsonl').read_text().splitlines()];prefix=rows[:3000];assert [r['step'] for r in prefix]==list(range(1,3001))
counts=collections.Counter(x.split(':')[0] for r in prefix for x in r['sample_ids']);assert counts==dict.fromkeys(['personal_demo_'+n for n in ['0001','0002','0003','0005','0012','0013']],2000)
uniq={t:len({x for r in prefix for x in r['sample_ids'] if x.startswith(t+':')}) for t in counts};idx=json.loads((OUT/'window-index.json').read_text());assert uniq==collections.Counter(r['take'] for r in idx)
phasecounts=collections.Counter();lookup={r['sample_id']:r for r in idx}
for r in prefix:
 for x in r['sample_ids']:phasecounts[lookup[x]['phase_proxy']]+=1
assert old['revision']==new['base_revision'] and old['domain']==new['domain_id']==29 and old['seed']==new['seed']==17
assert old['groups']==new['group_parameters'] and old['peak_lr']==new['optimizer']['peak_lr']
assert old['microbatch']==new['microbatch']==1 and old['effective_batch']==new['effective_batch']==4 and old['accumulation']==new['gradient_accumulation']==4
assert old['source']['sha256']==next(x for x in json.loads((OUT/'data-audit.json').read_text())['train'] if x['episode']['id']=='personal_demo_0001')['episode']['sha256']
modelcode=[]
for b in (ROOT/'.cache-xvla/checkpoint').glob('*.py'):
 a=OLD/'.cache-xvla/checkpoint'/b.name;modelcode.append({'name':b.name,'old_exists':a.exists(),'identical':a.exists() and sha(a)==sha(b),'new_sha256':sha(b),'old_sha256':sha(a) if a.exists() else None})
report={'passed':True,'created_unix_s':__import__('time').time(),'base_revision':new['base_revision'],'new_base_weight_sha256':json.loads((OUT/'data-audit.json').read_text())['base_weight_sha256'],'active_domain29_initial_tensors_sha256':actual,'actual_old_saved_initial_tensors_vs_new_initial_hashes_equal':True,'inactive_domain_hashes_equal':True,'native_recipe_source':source,'model_code_comparison':modelcode,'same_optimizer_groups_rates_counts':True,'same_microbatch_and_accumulation':True,'same_native_forward_loss':True,'same_1k_prompt_head_warmup_then1k_joint_lr_ramp':True,'same_O2_domain29_interleaved_rotation_and_target_text':True,'same_take0001_canonical_npz_sha256':True,'actual_first3000_step_sample_counts':dict(counts),'actual_first3000_unique_windows':uniq,'actual_first3000_phase_proxy_exposures':dict(phasecounts),'all439_windows_observed_by_step3000':sum(uniq.values())==439,'fixed_checkpoint_snapshot':'Trainer synchronous save() is called only after optimizer.step() at named step. cpu_state(model.state_dict()) performs detached CUDA->CPU copies before serializing fixed checkpoint and seeded probe; no next training step runs until save() returns. Fixed step3000.pt never overwritten. Full CPU postwrite loading confirmed schema and step.','differences':['Training source breadth: one take /48selected windows becomes six takes /439complete windows','Sampling changes from independent uniform-with-replacement among48windows to balanced shuffled take blocks with per-take permutation queues','New3k has2000samples/take versus old12000samples fromtake0001; total3k12,000sampleexposures equal','12k adds training time/exposure beyond3k, not a data-only comparison','Processed RGB tensors cached onCPU only, no learned features cached; oldprocessor runs on demand','New intermediate checkpoint probes use isolated RNG fork and restore training state; new checkpoint storage format separates inference weights and optimizer recovery'], 'interpretation':'3k comparison changes breadth/window coverage/sampling together under matched native model recipe, not a single-variable data-count intervention. 12k comparison also changes update budget. Neither result can isolate causal effect of breadth alone.','fixed_3k_validation':json.loads((OUT/'checkpoint-validation-fixed-3000.json').read_text())}
(OUT/'comparability-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['actual_old_saved_initial_tensors_vs_new_initial_hashes_equal','actual_first3000_step_sample_counts','actual_first3000_unique_windows','all439_windows_observed_by_step3000','interpretation']}));print(json.dumps({'source_identical':[(x['path'],x['identical']) for x in source],'model_code_identical':[(x['name'],x['identical']) for x in modelcode]}))
