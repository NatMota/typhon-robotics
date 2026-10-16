"""Saved-trace limitation audit; no simulator imports or model calls."""
from pathlib import Path
import json,gzip,time
O=Path('/var/tmp/typhon-init14-20261016')
def lines(p):
 with gzip.open(p,'rt')as f:return[json.loads(x)for x in f]
def main():
 out={}
 for cond in ['N','C']:
  p=O/'episodes'/cond/'retain-load01-training-demo_0001';fr=lines(p/'controller-force-trace.jsonl.gz');tr=lines(p/'controller_trace.jsonl.gz')
  acts=[r['policy_action']for r in fr if r['policy_action']];closed=[r['index_zero_based']for r in acts if r['issued_target'][9]>.5]
  viol=[{'record_index':i,'step':r['step'],'margin_rad':r['nearest_joint_limit_rad']}for i,r in enumerate(tr)if r['nearest_joint_limit_rad']<0]
  ep=json.loads((p/'episode.json').read_text());rule=json.loads((p/'rule-summary.json').read_text())
  out[cond]={'closed_commands':len(closed),'first_close':closed[0]if closed else None,
   'gripper_probability_range':[min(r['issued_target'][9]for r in acts),max(r['issued_target'][9]for r in acts)],
   'any_object_contact':any(r['object_truth']['grasp_contact']for r in tr),'D5_trigger':rule['trigger'],
   'joint_violation_sample_count':len(viol),'first_joint_violation':viol[0]if viol else None,
   'minimum_joint_margin_rad':min(r['nearest_joint_limit_rad']for r in tr),'strict_evaluation':ep['evaluation']}
 assert out['N']==out['C']
 data={'case':'retain-load01-training-demo_0001','conditions':out,'matching_summaries':True,
 'interpretation':'No closure proposed or object contact; both primitives inactive. Same existing joint excursion in native and assisted runs. No causal benefit claimed.',
 'nonexact_pairs':[p for p in json.loads((O/'analysis/pairs.json').read_text())if not p['gate']['exact_effective_pretrigger_equivalence']],
 'epoch':time.time()}
 (O/'analysis/limitations.json').write_text(json.dumps(data,indent=2)+'\n')
 print(json.dumps(data))
if __name__=='__main__':main()
