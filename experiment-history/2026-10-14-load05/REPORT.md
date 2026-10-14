# LOAD05: restart qualification failed before the crossover

The pre-contact C/B comparison did **not** run. One saved-action qualification replay was attempted; zero policy workers were started, zero new model queries were made, and zero crossed continuations were executed.

The amended protocol used visible demo0012, loading instruction and seed912171. Archived first gripper–brick contact occurred at control tick131 for C and tick107 for B. The last shared native query boundary strictly before either contact is **tick100, query2**. The initial post-grasp proposal was superseded before execution.

## What passed before the stop

The C reset matched the archived initial truth. The first 90 saved actions (tick41 through130) reproduced the saved dynamic/controller trace within the existing position/rotation tolerances. RGB and measured end-effector inputs matched exactly at the native query boundaries encountered. The captured tick100 brick met the upright, displacement and stability requirements.

## What failed

After that prefix and first resumed chunk, the verifier restored the copied tick100 MuJoCo and Python state. Its complete signature equality assertion failed. The runner stopped immediately before preparing model tensors or starting inference. Neither B's qualification prefix nor any crossover continuation was attempted.

The pre-copy recovered arrays and signature are saved. The differing post-restore signature was not retained, so the exact set of mismatching fields cannot be reconstructed from this attempt. This is an instrumentation limitation and prevents certifying the restart.

A subsequent read-only source audit found **36 XML pointer-address strings** inside the saved gripper signature. A copy-only XML test shows the generic canonicalizer rejects a deep copy even when the serialized XML content is identical. This establishes a verifier defect; it does **not** establish that this was the only discrepancy in the failed restoration.

The original code and failed evidence remain unchanged. No outcome retry, new training, guard, extra simulator-forward call or tolerance relaxation followed the stop.

## Decision

LOAD04's counterbalanced destination-selection result is unaffected. LOAD05 supplies no evidence about which checkpoint or incoming state causes tipping. Repair and qualify the restart verifier under a separately frozen amendment before attempting the four crossed continuations.

The next repair should compare XML structure/content, retain both signatures and field-level differences before asserting, and test the copy-only failure first. Any physical replay must be newly scoped and must stop on an unexplained physical/controller/input mismatch.

## Evidence

- [Selection](selection.json)
- [Stop record](STOPPED.json)
- [Copy-only verifier audit](verifier-source-audit.json)
- [Planner amendment](PLANNER_AMENDMENT.md)

Raw Powerglide output: /var/tmp/typhon-load05-20261014.
Private branch: research/load05-precontact-crossover.
No final-placement success rate is reported for this unexecuted acquisition diagnostic.
