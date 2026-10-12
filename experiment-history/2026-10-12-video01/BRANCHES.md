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
