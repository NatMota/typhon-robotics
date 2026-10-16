# Four archived feeding handover references

Only **one of the four archived feeding endpoints** is represented by the existing Q1 handover tests. Those tests used two loading-policy seeds and two control modes, but all four started from the same demo0012/912171 feeding boundary. This ledger adds no robot trials, inference, rendering or training.

| Archived feeding execution | Feeding success | Relationship to existing Q1 handover | Brick delta (mm / degrees) | Measured wrist delta (mm / degrees) |
|---|---|---|---:|---:|
|demo_0012 / 912171|True|Exact source-state linkage|0.0000 / 0.0000|0.0000 / 0.0000|
|demo_0012 / 912172|True|Related, not matched|0.5942 / 0.1685|3.5797 / 0.3717|
|demo_0013 / 912171|True|Untested boundary in available evidence|8.7253 / 0.3315|2.6201 / 0.2979|
|demo_0013 / 912172|True|Untested boundary in available evidence|10.1336 / 0.3978|0.8247 / 0.0949|

All four archived feeding episodes meet the existing requested-feeding strict score, end stable, and actually execute OPEN as their last gripper command. The original generic episode score uses the physical loading-scene target; the archived dispatch-specific feeding score is used here. These are simulated metres/angles under the inherited geometry, not new physical calibration.

**The second demo0012 endpoint is related, not matched.** Its loading seed appears in INIT14, but that evaluation replayed the first feeding seed. Both demo0013 feeding endpoints remain untested as Q1 handovers in the inspected evidence. Near-identical object pose does not establish matching wrist, controller, velocity or solver state.

The exact linkage for demo0012/912171 is supported by all1460recorded rich trace rows matching the earlier continuous feeding stage, its saved exact input/prediction/noise/terminal-truth receipt, and the existing Q1 replay qualification verifying its boundary state. No new replay was performed. The other rows are not given full-state-equivalence credit.

[ledger.json](ledger.json) records full object quaternion/height/velocity, measured wrist quaternion and position, both measured finger-joint coordinates, actual last executed command, controller target pose and torques, and frame/input SHA provenance. Finger-joint separation is a travel-coordinate proxy; it is not a calibrated inner-pad gap. The final PNG follows settling; query47 is the last recorded actor input and is not claimed to occur at the same instant.

**Implication for HIER02-LIVE-R1:** keep the original eight-workflow panel. It must create its own live endpoints without forcing them back to these references. Three of its four feeding contexts have no exact prior Q1 boundary linkage. This is interpretation evidence, not a target state or an execution success prediction.

Source audit:46local files; all hashes unchanged after analysis. Four fixed rows; one exact linkage, one related endpoint, two untested boundaries. Scope is the archived HIER01-DEV and continuous-feeding records plus INIT14 Q1 handover records; this does not assert that every historical archive was searched.

## Read-only reproduction

On Mint, point data-root at the existing research archive and choose a new output directory. Create clock.json in that output with activated_epoch=time.time() and deadline_epoch=activated_epoch+1200, then run:

    python3 analyse.py --data-root "/media/nm/Fast Drive/Typhon/research-20261015" --output /var/tmp/typhon-hier02-reference-reproduction

This uses Python standard-library JSON/gzip/hash arithmetic only. It never imports a simulator or model. The supplied clock.json retains the original capped review clock; do not overwrite the original ledger.
