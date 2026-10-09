"""Separate assisted diagnostic: Qwen/RGB completion and hard geometric vetoes.

Never counted as unassisted learned-policy success or a replacement pilot result.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from m7_evaluate import M7Policy
from m7_service_smoke import run_case
from supervisor.planning.m7_service import QwenGoalChecker


class BoundedM7Policy(M7Policy):
    deadline=None

    def _receive(self,timeout=180):
        if self.deadline is not None:timeout=min(timeout,max(0.,self.deadline-time.monotonic()))
        return super()._receive(timeout)

    def propose(self,observation,deadline):
        self.deadline=deadline
        try:return self.predict(observation)
        finally:self.deadline=None

    def commit(self,target):
        # M7Policy.to_action already commits only executed target/history steps.
        pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(a.protocol.read_text());wanted=['m7-personal_demo_0001','recovery-already_completed-01','recovery-wrong_orientation-01']
    selected=[next(s for s in protocol['pilot'] if s['scenario_id']==sid) for sid in wanted]
    with a.checkpoint.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    manifest={'evidence_type':'separate assisted diagnostic, not learned-policy benchmark',
      'checkpoint':str(a.checkpoint),'checkpoint_sha256':digest,'qwen_model':'Qwen3-VL-2B-Instruct',
      'reference_source':'supplied task goal rendered by existing geometry, never observed object truth',
      'completion_head':'Qwen claim AND independent RGB reference/numeral veto AND measured open gripper AND3consecutiveframes',
      'guarded_actions':'workspace/no-go/current joint/speed constraints;3candidate samples,3step prefix,10sdeadline',
      'controller_is_not_a_safety_certified_system':True,'new_training':False,
      'same_common_intent_request_as_pilot':protocol['requested_action'],
      'source_code_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
       [Path(__file__),ROOT/'scripts/m7_evaluate.py',ROOT/'scripts/m7_service_smoke.py',ROOT/'supervisor/planning/m7_service.py',ROOT/'supervisor/planning/m7_goal_worker.py',ROOT/'supervisor/planning/m7_telemetry.py']}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    checker=policy=None;rows=[]
    try:
        checker=QwenGoalChecker(ROOT,ROOT.parent/'models/Qwen3-VL-2B-Instruct',a.output/'qwen.log')
        policy=BoundedM7Policy(a.checkpoint,a.output/'xvla.log',request=protocol['requested_action'])
        if policy.config['condition']!='raw':raise ValueError('This diagnostic is intentionally V1/raw; V2 needs explicit causal annotation binding.')
        manifest.update(checkpoint_load=policy.load_info,qwen_load=checker.load_info)
        (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for scenario in selected:
            row=run_case(scenario,a.output/scenario['scenario_id'],checker,policy,4,10.)
            row['evidence_type']='assisted diagnostic; excluded from unassisted success counts';rows.append(row)
            (a.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    finally:
        if policy:policy.close()
        if checker:checker.close()


if __name__=='__main__':main()
