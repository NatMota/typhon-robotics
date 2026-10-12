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
