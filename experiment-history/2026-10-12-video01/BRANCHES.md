# VIDEO01 branches

- [object-motion](https://github.com/NatMota/typhon-robotics/tree/research/video01-object-motion) — `17a8997ca4489fd3e6f61944c6d6bf45c68ad7d3`
- [hand-motion](https://github.com/NatMota/typhon-robotics/tree/research/video01-hand-motion) — `e9ecf760bc2e265297fd3ff90178632770b47ca1`
- [visual-features](https://github.com/NatMota/typhon-robotics/tree/research/video01-visual-features) — `d1e7a7f88db2cf71b77f834fac4b76abf3f1b6f7`

Code, source checksums, small derived observations/feature matrices, reports and configs are versioned. Full media, model weights and raw rollouts are on Mint. These initial diagnostics do not validate a video-to-robot action adapter.

## Dense continuation — VIDEO02

- [Dense observations and temporal/crop validation](https://github.com/NatMota/typhon-robotics/tree/research/video01-temporal-validation)
- [Results and reproduction](https://github.com/NatMota/typhon-robotics/blob/research/video01-temporal-validation/experiment-history/2026-10-12-video02/REPORT.md)

2,644 phone observations;147 four-second view-windows; six paired previews. Frozen full+crop features improve offline matching, not robot execution. The corrected dense-v2 export passes all 2,644 raw sequential frame hashes; 2,710 files verified on Mint. Superseded dense-v1 and its frame-index failure remain preserved. Calibration/contact and the Panda action adapter remain unresolved.

## Held-state continuation — VIDEO03

[Results and reproduction](https://github.com/NatMota/typhon-robotics/blob/research/video03-held-probe/experiment-history/2026-10-12-video03/REPORT.md). Thirty-five small observation probes from unmarked personal footage. Perfect conservative-interior agreement did not survive hand-present transition checks: 5/8 false held views for full images, 2/8 with crop/fusion. No motor-policy update or control authority. All results and fresh reproduction are verified on Mint.

## VIDEO04 — Training-only contact negatives

Private branch: [research/video04-contact-negatives](https://github.com/NatMota/typhon-robotics/tree/research/video04-contact-negatives). [Report and full results](https://github.com/NatMota/typhon-robotics/blob/research/video04-contact-negatives/experiment-history/2026-10-12-video04/REPORT.md). Six small observation probes, with new labels committed before scores. Withdrawal error improves; the extra panel still has the same failing approach event (1/3). All 90 conservative held positives retained. No motor-policy change or contact authority. OST #11021.

## VIDEO05 — Pre-contact labels and held-midpoint audit

[Private branch](https://github.com/NatMota/typhon-robotics/tree/research/video05-precontact-pause) and [report/case card](https://github.com/NatMota/typhon-robotics/blob/research/video05-precontact-pause/experiment-history/2026-10-12-video05/REPORT.md). Thirteen training-only pre-contact views repair the old approach error. The other excluded-take approach remains wrong; all 90 labelled held interiors retained. History/extra weighting does not resolve the remaining failure. Held-pause qualification is incomplete because one selected case lowers the block and another has uncertain tracking. No motor-policy update or control authority. OST #11022.


## VIDEO06 — Motion audit

Branch: research/video06-motion-audit.
Report: https://github.com/NatMota/typhon-robotics/blob/research/video06-motion-audit/experiment-history/2026-10-12-video06/REPORT.md

Six phone takes and 342 verified frames. Glyph flow supports 194/222 held frame pairs, including 53 where the old contour lacked motion; 28 remain unsupported. Demo0015 midpoint has real image movement in both cameras. Small held sway is distinct from exact stillness; no strict stationary-held panel qualifies. 28 numeric/provenance files reproduce exactly. No robot-action labels, VLA update or control authority. VIDEO07 #11024 tests native cadence and phase timing. Bulk 204 files / 67,572,407 bytes checksum-verified on Mint.


## VIDEO07 — Native cadence

Branch: research/video07-native-cadence.
Report: https://github.com/NatMota/typhon-robotics/blob/research/video07-native-cadence/experiment-history/2026-10-12-video07/REPORT.md

969 native frames, 342 identical query frames. Held pair support 194/222 to 212/222; 0.6s tracks 104/162 to 138/162, with one stationary-control support loss. Extract motion before temporal abstraction, retaining unknowns. Coarse phase notes are not robot-action labels. No training or controller change. Bulk 1,128 files / 205,661,664 bytes SHA-verified on Mint. Next independent experiment: LIVE04 #11014 delayed candidate handoff.

## LIVE04 — Live candidate handoff

Branch: research/live04-boundary-handoff.
Report: https://github.com/NatMota/typhon-robotics/blob/research/live04-boundary-handoff/experiment-history/2026-10-12-live04/REPORT.md

Nine V4 development rollouts. Three API choices executed 62 exact saved targets, including two complete chunks; one late reply was rejected. Original V4 grasped and placed on its known scene but retained 85.33 degrees yaw error. Both boundary routes lost that grasp; all preserved the already-completed case. Fixed two-second waiting is diagnostic and has no demonstrated benefit. Twelve regressions and exact model replay pass. LIVE05 #11025 separates waiting, sampling and guard effects. Bulk v1/v2 preflights and v3 episodes are retained on Mint; use LIVE04_BACKUP_RECEIPT.json for verification.

## LIVE05 — Separate guards, waiting and sampling

Branch: research/live05-scheduler-isolation.
Report: https://github.com/NatMota/typhon-robotics/blob/research/live05-scheduler-isolation/experiment-history/2026-10-12-live05/REPORT.md

Thirty controlled-delay V4 trials. The known grasp survives both sampling schemes at 0/0.1/2 s waits without guard intervention; enforcing the guard loses it even at zero wait. Old per-tick raw-target distance vetoes eight reference steps. Most live preview paths nearly overlap. Exact-reference rendering limits retained; no training or API benefit claim. Mint archive:1,452 files / 2,552,769,509 bytes, SHA-verified. LIVE06 #11026 qualifies the execution contract.

## LIVE06 — Execution guard qualification

Branch: research/live06-execution-guard.
Report: https://github.com/NatMota/typhon-robotics/blob/research/live06-execution-guard/experiment-history/2026-10-12-live06/REPORT.md

Nine V4 trials restore the known pickup and placement with unchanged weights, while retaining 85.33 degrees yaw error. Eight expert trials pass; new execution check accepts all 1,714 ticks versus 304 old raw-target vetoes. Eleven negative/contract regressions pass. First 570 native/corrected control records match; later wrist motion differs with tiny historical-image changes, so full trace identity is not claimed. LIVE07 #11027 must integrate the explicit execution helper into the live service without changing generic prospective admission. Use LIVE06_BACKUP_RECEIPT.json for bulk verification.

## LIVE07 — Integrated live guard

[Report and comparison video](https://github.com/NatMota/typhon-robotics/blob/research/live07-live-integration/experiment-history/2026-10-12-live07/REPORT.md)

Nine V4 trials and 44 regression checks. The corrected live first-admissible route restores the known pickup and placement, with 79.46° remaining yaw and a narrow 14.985 mm centre error. The Decisions route misses; only one of two activated API queues actually executes. All preserve the already-completed brick, without establishing recognition. First 41 control/truth records match before the first A/C choice; later timing differs. Bulk: 405 files / 444,892,013 bytes SHA-verified on Mint. OST #11027.

## LIVE08 — Candidate choice versus timing

[Report and four-cell video](https://github.com/NatMota/typhon-robotics/blob/research/live08-choice-timing/experiment-history/2026-10-12-live08/REPORT.md)

Fixed initial A/C choice × 40/41 holding ticks on the same inspected V4 scene. A grasps and places under both waits; C grasps under neither. Initial images, noise and all four proposals match LIVE07. One-tick waiting alone does not explain the lost grasp; orientation remains 79–81° wrong for A. Four strict failures; no new training or API calls. Bulk: 114 files / 134,433,160 bytes SHA-verified on Mint. OST #11028.

## LIVE09 — Actual Decisions calls on blue ghosts and future RGB

[Report and full choice table](https://github.com/NatMota/typhon-robotics/blob/research/live09-ghost-decisions/experiment-history/2026-10-12-live09/REPORT.md)

32 choice calls plus 32 reason calls on the previously frozen R10 cards. Primary ghost-only selects 3/12, with 2/3 proxy-adequate; future RGB selects 5/12, with 5/5 adequate, but abstains 7/12 and changes decisions under reordering in 4/6 states. Future RGB also selects from the unsuitable V4 held state. Both modes contain privileged simulator lookahead. Secondary regional completions are separate from strict success, and supplied held states receive no new-pickup credit. No live-control benefit demonstrated. OST #11029; use LIVE09_BACKUP_RECEIPT.json for bulk verification.

Provenance correction (13 October): LIVE09 repeats the completed R10 batch from 12 October morning. All 32 choices and 32 categorical-reason responses match; it supplies no additional independent scenes. See the LIVE09 REPEAT_AUDIT.md and repeat_audit.json. LIVE07 and LIVE08 remain new controller/causal investigations.
