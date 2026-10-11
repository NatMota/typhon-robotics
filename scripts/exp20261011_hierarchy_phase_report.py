"""Recompute advisory phase-transition timing; no fitting or threshold changes."""
import argparse,json
from pathlib import Path
from collections import Counter

def report(path):
 rows=json.load(open(path/'predictions.json'));phases=['approach','grasp','lift','orient','place','release','retreat'];rr=[r for r in rows if r['split']=='P_validation'];events=[]
 for take in sorted(set(r['take'] for r in rr)):
  rs=[r for r in rr if r['take']==take]
  for phase in phases:
   actual=next((r['action_index'] for r in rs if r['phase']==phase),None);raw=next((r['action_index'] for r in rs if r['predicted']==phase),None)
   stable=next((r['action_index'] for i,r in enumerate(rs[:-1]) if r['predicted']==phase and rs[i+1]['predicted']==phase),None)
   events.append(dict(take=take,phase=phase,teacher_first_sample=actual,raw_first_sample=raw,first_two_consecutive_start=stable,raw_offset_s=None if raw is None or actual is None else (raw-actual)/20.,two_consecutive_offset_s=None if stable is None or actual is None else (stable-actual)/20.))
 result={'scope':'Held-out P_validation diagnostic; raw predictions advisory only. Teacher labels are command phases, not verified physical phase truth. No model or threshold tuning.','frames':len(rr),'accepted_frames':sum(r['accepted'] for r in rr),'accepted_coverage':sum(r['accepted'] for r in rr)/len(rr),'ungated_phase_later_than_teacher_frames':sum(phases.index(r['predicted'])>phases.index(r['phase']) for r in rr),'ungated_release_before_teacher_release_frames':sum(r['predicted']=='release' and phases.index(r['phase'])<5 for r in rr),'accepted_release_before_teacher_release_frames':sum(r['accepted'] and r['predicted']=='release' and phases.index(r['phase'])<5 for r in rr),'first_occurrence_timing':events,'ungated_release_confusion_by_teacher_phase':dict(Counter(r['phase'] for r in rr if r['predicted']=='release')),'online_conclusion':'Abstention selects zero samples; zero accepted false transitions does not establish safe phase advancement. Learned head remains disabled.'}
 text=json.dumps(result,indent=2)+'\n';out=path/'transition-analysis.json'
 if out.exists():assert out.read_text()==text,'Existing result differs; preserve and inspect'
 else:out.write_text(text)
 print('Verified transition analysis',len(rr),'held-out frames')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('path',type=Path);a=ap.parse_args();report(a.path)
