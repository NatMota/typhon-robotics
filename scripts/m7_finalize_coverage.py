"""Attach the explicitly authorized completed protocol to a read-only audit."""
import argparse,csv,hashlib,json
from pathlib import Path

def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def main(root,out,protocol_file):
 root=root.resolve();protocol=json.loads(protocol_file.read_text());write(out/'new_completed_protocol.json',protocol)
 rows=json.loads((out/'personal_examples.json').read_text())
 for r in rows:
  permitted='personal_'+r['example_id'] in protocol['new_personal_completed_train_ids']
  r['new_completed_protocol_split']='C_train' if permitted else None
  r['new_completed_native_export_status']='authorized_pending_separate_generation_validation' if permitted else None
 write(out/'personal_examples.json',rows)
 summary=json.loads((out/'validation_summary.json').read_text());summary.update(completed_split_status='C_train authorized for0018/0019 in NEW protocol; frozen exclusion unchanged; native export generated/verified by temporal agent separately',new_synthetic_completed_authorized=2,new_protocol_file='new_completed_protocol.json',raw_curated_clip_bytes=sum(v['original_clip']['bytes'] for r in rows for v in r['real_camera_views']))
 write(out/'validation_summary.json',summary)
 synthetic=json.loads((out/'synthetic_examples.json').read_text());excluded=json.loads((out/'curated_exclusions.json').read_text())['excluded_demonstrations']
 flat=rows+synthetic+[dict(example_id=e['demo_id'],cohort='personal',category=e['attempt_type'],existing_split='excluded_curated',train_eligible_existing=False,native_robot_rgb_action_pair_available=False,human_native20D_targets_available=False,m7_semantic_v1_windows=0,canonical_exclusion_reason='user excludes short pause from curated set; not a failed human trial',training_next_step='retain excluded source clips and original annotations') for e in excluded]
 keys=['example_id','cohort','category','existing_split','train_eligible_existing','native_robot_rgb_action_pair_available','human_native20D_targets_available','m7_semantic_v1_windows','canonical_exclusion_reason','training_next_step','new_completed_protocol_split','new_completed_native_export_status']
 with (out/'coverage.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(flat)
 source=root.parent/'humanoid-task-supervisor-m15/results/personal-m15-20261009';sc=json.loads((source/'reconstructed_scenarios.json').read_text())['synthetic'];legacy=[]
 for r in sc:
  folder=source/'episode_artifacts'/r['scenario_id'];files=[dict(path=str(f.resolve()),bytes=f.stat().st_size) for f in folder.iterdir() if f.is_file()]
  legacy.append(dict(example_id=r['scenario_id'],cohort='legacy_M15_synthetic',category=r['category'],split='unassigned_legacy_not_eligible_without_new_protocol',native_rgb_action_npz_available=False,source_episode_exists=(folder/'episode.json').exists(),source_files=files,reason='Old M15 scenario reconstruction comparison, not canonical M3 S_train. Historical visual version and action representation require explicit re-render/export before native training; no automatic import.'))
 write(out/'legacy_m15_synthetic.json',legacy)
 report=(out/'REPORT.txt').read_text();report=report.split('\nProtocol addendum:')[0]
 report+='\nProtocol addendum: parent authorized NEW C_train for personally derived0018/0019 plus2new independently generated completed cases. This audit keeps old excluded_motor_training fields and adds new protocol fields. No generated rows/NPZs are asserted until the separate temporal export is validated.\nLegacy M15 also contains15synthetic scenario comparison traces. They are inventoried separately in legacy_m15_synthetic.json, with no assigned native training split; do not conflate them with canonical S_train.\n';(out/'REPORT.txt').write_text(report)
 (out/'REPRODUCE.txt').write_text('Run on Powerglide with the archived semantic-v1 index and new_completed_protocol.json supplied as read-only inputs:\npython3 scripts/m7_audit_training_coverage.py --root . --output NEW_COVERAGE_DIR --semantic-index coverage-input-semantic-v1.csv\npython3 scripts/m7_finalize_coverage.py --root . --output NEW_COVERAGE_DIR --protocol results/personal-m7-training-20261009/coverage-v1/new_completed_protocol.json\nThe first output directory must be new/nonexistent. Inputs remain unchanged. No GPU is needed.\n')
 files=[]
 for p in sorted(out.rglob('*')):
  if p.is_file() and p.name!='bundle_manifest.json':files.append(dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 write(out/'bundle_manifest.json',dict(schema='m7-coverage-bundle-v1',files=files,total_bytes=sum(x['bytes'] for x in files),source_scripts=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [root/'scripts/m7_audit_training_coverage.py',root/'scripts/m7_finalize_coverage.py']]))
 print(json.dumps({'rows':len(flat),'personal_curated':len(rows),'personal_excluded':len(excluded),'canonical_synthetic_attempts':len(synthetic),'legacy_synthetic_traces':len(legacy),'bundle_bytes':sum(x['bytes'] for x in files)}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);a=p.parse_args();main(a.root,a.output,a.protocol)
