EVAL36 existing-trace audit, 16October2026

Reproduce with Python3 standard library only:
python3 audit-eval36-traces.py --self-check
python3 audit-eval36-traces.py simulation-eval36-full-output-20261016.zip NEW_AUDIT_DIRECTORY

Input archive SHA256:4fba9e72f78600a900dc77999997d3da6b19531b316a45cc0d612084616a9813. The archive is retained on Mint at /media/nm/Fast Drive/Typhon/research-20261015/simulation-eval36-full-output-20261016.zip. All read members are checked against the archived output manifest; SOURCE_HASH_MANIFEST.json records every member used. No simulator/model imports or calls. The source archive and frozen experiment are never modified.

Results:144/144stages and72/72workflow engineering/primary flag comparisons agree exactly with saved summaries and ledger; DISCREPANCIES.json is empty. The audit reconstructs185forbidden contact records across7stages. These are contact records at20Hz post-control logging resolution, not185independent collisions or trials. The smallest saved nearest-joint-limit margin is-0.031060351448643206rad. The script recomputes the minimum across1460saved control samples per stage, including24settle samples; it does not reconstruct margins from model joint-limit bounds. Primitive completion is cross-checked against executed offset/clearance flags and final clearance state. No physics-substep contact absence or hardware-safety claim follows.

Draw024:all48noise hashes match within each stage, and feeding initial physical boundaries match. Raw secondary-image and prepared-image hashes first differ at zero-based query35(the36thquery). Raw primary-image and prepared-proprioception hashes first differ at query36. Logged target/command/state records first differ at feeding action index1050, global control step1091. Feeding final object positions are equal, but complete physical boundary hashes differ. Loading starts from those unequal boundaries; its prepared image/proprioception inputs differ from query0. The saved evidence establishes this sequence, not its cause. Draw024 remains in all36pairs; it was neither excluded nor rerun. Its downstream loading comparison includes unequal upstream state and must not be described as starting from an identical loading boundary.

Report integration:the parent preprint is limited to eight pages including references. Keep this script,144-row stage table,72-row workflow table,96-row draw024 query table,full forbidden-contact records and hash manifests in a separate supplement.

Both draw024modes retain primary success=True in the original ledger and independently rechecked engineering table. This is a saved outcome, not a reason to exclude or reinterpret the pair.
