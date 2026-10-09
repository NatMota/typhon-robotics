"""Read-only verification of frozen M0/M1/M15/M16 artifacts and M2 Git results."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone


def main(out):
    projects=Path.home()/'Projects'
    m16=projects/'humanoid-task-supervisor-m16/results/personal-m16-20261009'
    frozen=json.loads((m16/'frozen_inputs.json').read_text())
    records=[(Path(frozen['roots'][r['root']])/r['path'],r['sha256']) for r in frozen['files']]
    manifest=json.loads((m16/'artifact_manifest.json').read_text())
    records.extend((m16/r['path'],r['sha256']) for r in manifest['files'])
    mismatches=[]
    for path,expected in records:
        if not path.exists():mismatches.append({'path':str(path),'reason':'missing'});continue
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:mismatches.append({'path':str(path),'reason':'changed','expected':expected,'actual':actual})
    m2=projects/'humanoid-task-supervisor-m2'
    git_diff=subprocess.check_output(['git','-C',str(m2),'diff','HEAD','--','results','supervisor'],text=True)
    result={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'files_checked':len(records),
            'mismatches':mismatches,'M2_tracked_results_and_source_unchanged':not git_diff,
            'passed':not mismatches and not git_diff}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
