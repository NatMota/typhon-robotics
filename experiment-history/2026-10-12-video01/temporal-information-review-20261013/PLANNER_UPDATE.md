Codex implementation update, continuing Nat's authorised OST research. I completed your capped VIDEO01-C #11018 acceptance review and moved only that review ticket to Done. Broad VIDEO01 #11015 remains active. VIDEO10/11 stay frozen/closed; no observer tuning, physics, new fits or controller runs.

Evidence:
- All16 curated takes reviewed at seven predetermined paired raw instants each:112 pairs/224 view samples. Every contact sheet was inspected. This is sampled review, not a claim to have watched every intervening frame. All8 originals still match SHA256; demo0017 excluded (demo0006 included).
- Source-selected three examples, locked BEFORE per-example feature audit:0001 first normal pickup,0012 first pause-resume,0018 first already-completed. Distinct decision classes; not selected by favorable embeddings. 0007 remains the prior VIDEO10 negative control.
- Full16-row ledger contains initial/final observations, additional temporal evidence, original filenames/frame IDs/PTS in both views, sync uncertainty, task-authority source, possible robot consequence, representation status and limitations.
- Forty inspected view-instants linked to existing2644-row feature file.13 exact-native-frame matches;27 explicitly nearest existing frames, max0.066667s delta. No new feature extraction/tuning.
- Mint copy244files/10,174,190bytes verified zero mismatches; private branch pushed ce954163e61a1d0a55e738e03b2cde332aee2987.

Decision: REPRESENTATION BOTTLENECK, not 'only style' and not acceptance of complete-sequence control benefit.
1) Independent user authority exists for pickup/place/orientation subgoals. Same endpoints via pushing can fail requested pickup.0001 has visual held transfer and later release/withdrawal absent from endpoints alone, but not forces/contact timing/metric lift. Existing primary/secondary dense samples128/128; centres128/92, hands43/26. P_train, not held-out.
2)0012 has held intermediate observations; earlier VIDEO07 reports reduced progress then resumed movement. User's pause label preserved, but M1.5 explicitly says a human pause is not a robot command. No mandatory pause, external obstacle or second-actor intervention invented. P_train; primary/secondary centres101/60, hands45/35.
3) Independent M1.5 instruction says recognize an already-completed state without unnecessary manipulation.0018 stays goal-like and hand-free at all inspected instants.111dense frames/view, all centres,0hands. A needless pickup/replacement could preserve endpoints while violating instruction. HOWEVER initial-state recognition+supplied rule can already choose no action, so full-video decision necessity is unproven. Remains observation-only.
- All correction takes show intermediate motion but no independently required route;0007's END/STATIC physical success prevents declaring a full-path gain.
-0004 source/early lift uncertain in this sampled review; retained, not silently removed. All16 specific limitations documented.

Representation audit:
Ordered observations,2D centres/yaw, native flow and supported hand landmarks are retained. Existing VIDEO03–05 held probes provide limited selected-interior discrimination, but known excluded-take pre-contact false positives remain; no autonomous grasp/release authority. All2644 dense is_grasped and metricXYZ fields null. Cross-view timing/order retrieval/goal similarity ≠ correct causal phase semantics. No new model/controller/control-benefit test was run.

Private report:
https://github.com/NatMota/typhon-robotics/blob/research/video01-c-temporal-review/experiment-history/2026-10-12-video01/temporal-information-review-20261013/REPORT.md

Recommendation recorded: retain phase/state source evidence; do not start another motor-training/full-human-path run on an assumed benefit. If pursued later, narrowly qualify causal phase transitions versus endpoint/current-observation controls with whole-take exclusion and hard approach/withdrawal negatives. This review did NOT start that work.

Please review this decision and choose ONE next bounded step from actual remaining OST work, avoiding another open-ended observer/annotation loop. Current remaining Review:
11016 VIDEO01-A object-path review;11017 VIDEO01-B hand/pinch review;
11004 LIVE01 live harness/controller-telemetry qualification;
10996 EVAL02 confirmatory comparison/family-level uncertainty;
10992 DATA07 failure/correction coverage;
11007 DEL01 dashboard/media provenance;
11008 DEL02 accurate private write-up.
Blocked remain10984 observation authority,10989 video-action alignment,10991 calibration,10998 recovery,11000 selector,11002 residualRL,11003 offlinevalue/PPO.
Powerglide4090 available through16Oct02:00BST. No GPU training is currently running. Nat's latest instruction is continue previous work; keep repo private and bulk data on Mint.
Give an acceptance/stop criterion and time cap. If the two old A/B representation reviews can close against existing evidence, say what to reconcile rather than inventing more tests.
