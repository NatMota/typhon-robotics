"""Generate compact INIT02 tables from frozen recorded results."""
import sys,csv
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'scripts'))
from init02_common import OUT,read,dump,sha
def yes(x):return x is True or x=='True'
def main():
 s=read(OUT/'validation_summary.json');assert s['complete']and s['core_completed']==69
 assert read(OUT/'reproduction-validation.json')['passed']
 a=list(csv.DictReader((OUT/'comparisons.csv').open()));by={(r['model'],r['case']):r for r in a};cases=read(OUT/'evaluation-cases.json')['core']
 assert read(OUT/'train-R/initial-parameters.json')==read(OUT/'train-T/initial-parameters.json')
 changes={}
 for m,ref in [('R','P'),('T','P'),('T','R')]:
  for panel in s['panels'][m]:
   cc=[c for c in cases if c['panel']==panel]
   changes[m+'-'+ref+'-'+panel]={metric:{'gains':[c['id']for c in cc if yes(by[m,c['id']][metric])and not yes(by[ref,c['id']][metric])],'losses':[c['id']for c in cc if not yes(by[m,c['id']][metric])and yes(by[ref,c['id']][metric])]}for metric in ['qualified','supported_lift']}
 dump(OUT/'paired-changes.json',changes)
 p=s['panels'];out=['# INIT02 — matched loading-initiation adaptation','',
 'P is the frozen FL parent. R receives further rehearsal and completed-feeding preservation training. T receives the same common samples, with 1,500 old-loading presentations replaced by new loading-transition demonstrations.','',
 '| Test | P: parent | R: control | T: new loading |','|---|---:|---:|---:|']
 for panel,title in [('initiation','Load from completed feeding'),('preservation','Preserve completed feeding'),('prior_feeding','Earlier feeding'),('prior_loading','Earlier loading')]:
  out.append('| '+title+' | '+' | '.join(str(p[m][panel]['qualified'])+'/'+str(p[m][panel]['planned'])for m in ['P','R','T'])+' |')
 out+=['','These panels are reported separately. The eight initiation trials are two seeds at four nearby starts, not eight independent scene families.','',
 '## Supported pickup','',
 '| Panel | P | R | T |','|---|---:|---:|---:|']
 for panel in ['initiation','prior_feeding','prior_loading']:
  out.append('| '+panel.replace('_',' ')+' | '+' | '.join(str(p[m][panel]['supported_lift'])+'/'+str(p[m][panel]['planned'])for m in ['P','R','T'])+' |')
 out+=['','Supported pickup requires the existing contact-supported lift of at least 10 mm for three consecutive steps. It is distinct from strict endpoint success.','',
 '## Completed-feeding motion','',
 '| Model | Maximum brick motion (mm) | Contact cases | Mean maximum wrist motion (mm) | Mean wrist path (m) |','|---|---:|---:|---:|---:|']
 for m in ['P','R','T']:
  rr=[r for r in a if r['model']==m and r['panel']=='preservation']
  out.append('| %s | %.4f | %d/8 | %.1f | %.3f |'%(m,max(float(r['maximum_object_displacement_mm'])for r in rr),sum(yes(r['any_contact'])for r in rr),sum(float(r['wrist_max_displacement_mm'])for r in rr)/8,sum(float(r['wrist_path_m'])for r in rr)/8))
 out+=['','Preservation requires strict feeding success, no gripper-object contact and at most 1 mm object movement. Wrist travel is descriptive; object preservation alone does not prove a completion-recognition mechanism.','',
 '## Exact prior-skill cases','',
 '| Case | P | R | T |','|---|---|---|---|']
 for c in cases:
  if not c['panel'].startswith('prior_'):continue
  def cell(m):
   r=by[m,c['id']]
   return '%s; pickup %s; %.1f mm; %.1f deg%s'%('PASS'if yes(r['qualified'])else'FAIL','yes'if yes(r['supported_lift'])else'no',float(r['position_error_mm']),float(r['yaw_error_deg']),'; '+r['criteria_failed']if r['criteria_failed']else'')
  out.append('| '+c['id']+' | '+' | '.join(cell(m)for m in ['P','R','T'])+' |')
 out+=['','All paired gains and losses in both strict result and pickup are enumerated in paired-changes.json. The full 69-case table, including loading failures, is comparisons.csv.','',
 '## Predetermined advancement gate','',
 '| Criterion | Passed |','|---|---|']
 g=s['optional_gate']
 out+=['| '+k.replace('_',' ')+' | '+str(v)+' |'for k,v in g['criteria'].items()]
 optional=[read(q)for m in ['P','R','T']for q in (OUT/'episodes'/m).glob('dev-*/summary.json')]
 if g['passed']:
  assert len(optional)==12,'Development branch pending'
  out+=['','The gate passed. The following development-only demo0012 handover was previously inspected and remains quarantined; it is not an untouched family.','',
  '| Model | Loading strict | Feeding preservation |','|---|---:|---:|']
  for m in ['P','R','T']:
   rr=[r for r in optional if r['model']==m]
   out.append('| %s | %d/2 | %d/2 |'%(m,sum(r['requested_success']for r in rr if r['panel']=='initiation'),sum(r['preservation_qualified']for r in rr if r['panel']=='preservation')))
 else:
  assert not optional
  out+=['','The gate failed; the optional development panel was not run. No extra updates, new examples or outcome-selected checkpoint were added.']
 if g['retention_lost_cases']:out+=['','Earlier strict-success cases lost by T: '+', '.join(g['retention_lost_cases'])+'.']
 out+=['','## Training','',
 '| Model | Joint updates | Wall seconds | Weight SHA-256 |','|---|---:|---:|---|']
 for m in ['R','T']:
  t=read(OUT/('train-'+m)/'complete.json')
  out.append('| %s | %d | %.2f | %s |'%(m,t['step'],t['wall_s'],t['checkpoint_sha256']))
 out+=['','Both runs start from the identical original FL weights and initial RNG, with fresh optimizer state, 12,000 presentations, effective batch four and seed 151901. All shared VLM/core weights and selected prompt/head parameters train jointly; inactive domain rows remain unchanged. Final update 3,000 is the only candidate. Optimizer moments were not archived.','',
 '## Interpretation limits and reproduction','',
 'The new examples use expert robot action trajectories at personally derived scenarios. They are not recovered human motor commands or DAgger data. The four initial geometries span only 10.542 mm. See METHODS.md, protocol.json and PLANNER_PROTOCOL.md for the frozen design and success definitions.','',
 'Independent re-scoring was run twice; comparisons.csv, validation_summary.json and optional-gate.json are byte-identical, and both passes verify unchanged raw inputs. Technical interruptions and pre-action initialization failures are preserved separately under integration-issue directories. The qualification amendment requires exact actual actor inputs; raw rendering is not claimed bitwise reproducible.','',
 'Read-only reproduction from this worktree:','',
 '    .venv-sim/bin/python scripts/init02_analyse.py',
 '    TYPHON_INIT02_ANALYSIS_OUTPUT=/var/tmp/init02-independent-check .venv-sim/bin/python scripts/init02_analyse.py',
 '    .venv-sim/bin/python scripts/init02_media.py',
 '    .venv-sim/bin/python scripts/init02_report.py','',
 'Runtime: /var/tmp/typhon-init02-20261015. Archive root on Mint: /media/nm/Fast Drive/Typhon/research-20261015/init02-matched-adaptation. Large weights, tensor packets, traces and videos are archived outside Git. Full execution needs the frozen inputs at their documented paths; moving to another machine requires explicit path remapping and a fresh output namespace.','',
 'Four matched videos and four contact sheets are derived only from saved frames. media-validation.json records source hashes and frame counts. No new model calls or physics rollouts are used to create them.','',
 'The fixed INIT02 experiment ends at this branch; the wider OST programme continues with the planner.','']
 text='\n'.join(out);(OUT/'INIT02_report.md').write_text(text)
 (W/'experiment-history/2026-10-15-init02-matched-adaptation/INIT02_report.md').write_text(text)
 dump(OUT/'report-validation.json',{'passed':True,'core_episodes':69,'optional_episodes':len(optional),'paired_change_groups':len(changes),'report_sha256':sha(OUT/'INIT02_report.md')})
 print('REPORT_COMPLETE',69,len(optional))
if __name__=='__main__':main()
