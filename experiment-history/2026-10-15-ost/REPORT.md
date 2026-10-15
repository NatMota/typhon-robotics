# Opportunity solution tree — 15 October 2026

The live Design viewer and this offline export contain **102 work items across ten views**. Each full-tree experiment/work item has a distinct result box to its right. A Done ticket means its bounded investigation finished; it does not imply a positive experimental result.

Latest additions:
- CTRL03: nominal sampled clearance did not explain the actual finger-shell collision; the proposed readiness gate was not qualified.
- LIVE01-P: capture/transform lineage and offline tensor preservation passed; exact live paired behavioral transparency failed and remains in Review.
- CTRL04: a fixed four-second pregrasp pause improved entry alignment but did not prevent tipping. Strict success 0/2.
- AlphaNIST's fuller implementation remains Parked by request.

All ten served SVG responses match the exported bytes. Offline validation covers ticket completeness, result alignment, bounds and non-overlap. The live viewer loaded successfully; the recovery branch and its latest result cards were visually checked. The viewer's separate “Backup offline” indicator does not describe these verified repository/archive exports.

## Open and reproduce

[Live Design viewer](http://127.0.0.1:5176/llm-design-tool/ui/index.html?project=typhon-research&file=opportunity-tree&page=recovery&artboard=recovery-tree) uses the existing Mac-to-Mint local relay.

```bash
python3 render-offline.py
```

This regenerates SVG, CSV, layout JSON and Mermaid from the preserved source inputs and snapshots without contacting the Design app. The offline validator reports live installation as pending because it does not perform that check; `served-validation-current.json` separately records the completed live-byte verification. The refresh scripts mutate the live Mint Design project and should only be run deliberately.

CSV fields remain `id,type,label,parent_id,x,y,color`. Full-tree includes every work item; overview selects 19. Historical snapshots and earlier date folders are retained unchanged.

## Evidence locations

- [CTRL03 report](https://github.com/NatMota/typhon-robotics/blob/685ee7e269e198de2c65f16a4850faa191b9365c/experiment-history/2026-10-15-ctrl03/REPORT.md)
- [LIVE01-P report](https://github.com/NatMota/typhon-robotics/blob/4a801d13da1dc5f564e6505e4a6646d87878af9b/experiment-history/2026-10-15-live01-p/REPORT.md)
- [CTRL04 report and video](https://github.com/NatMota/typhon-robotics/blob/54c029da69f4fc6235bbecc4d06d2e9e2c620819/experiment-history/2026-10-15-ctrl04/REPORT.md)

New raw archives reside on Mint Fast Drive under `Typhon/research-20261015/`, with per-file SHA-256 verification receipts. This documentation revision does not modify experiment outputs or trained models. The repository remains private.
