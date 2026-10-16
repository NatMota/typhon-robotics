"""Explicit HIER02-LIVE-R1 activation; scientific parent runner unchanged."""
from pathlib import Path
import time,traceback
import hier02_live as run
run.O=Path('/var/tmp/typhon-hier02-live-r1-20261016')
if __name__=='__main__':
    try:
        activation=run.read(run.O/'activation.json')
        assert activation['name']=='HIER02-LIVE-R1'
        assert activation['scientific_parent_commit']=='55ff6aeba5ff371d3df170aece2a431e9289b440'
        assert activation['restoration_verified']and activation['original_attempt_policy_outcomes']==0
        run.main()
    except BaseException:
        run.dump(run.O/'STOPPED.json',{'activation':'HIER02-LIVE-R1','phase':run.PHASE,
            'error':traceback.format_exc(),'epoch':time.time(),'no_retry':True})
        raise
