# OST continuation, 15 October 2026

The live tree and this reproducible snapshot cover **112 work items across ten views**. Every experiment/work item has a separate result or blocker card to its right. All 112 IDs appear exactly once in the full tree; cards do not overlap. All ten served SVG files match the offline regeneration byte-for-byte.

The latest RETENTION, FIT, RET-C and NUM cards were inspected at readable scale in the signed-in Chrome bridge. Current rendered evidence is `generalisation-latest112.png`. Earlier dated snapshots and frozen experimental outputs remain unchanged.

## What changed

| Work | Actual observation | Decision |
|---|---|---|
| XY #11059 | B followed the existing square; +/-20 mm numeric offsets caused less than 1 mm endpoint response | No useful local numeric steering established |
| SPATIAL #11060 | Renderer QA changed 29 protected foreground pixels; zero model calls | Blocked; no cue-dominance conclusion |
| COVERAGE #11061 | Latest B continuation omitted G1/G2; prior stages included them | Diagnose retention before attributing causality |
| RETENTION #11062 | GOAL/FL 2/2 at both trained goals; B 2/2 at G0 and 0/2 at G1, near old G0 | Local retained-behavior regression |
| FIT #11063 | B post-lift prediction differs on identical teacher inputs | Closed-loop state divergence alone is insufficient |
| RET-C #11064 | Matched C retains G0 2/2, G1 1/2 with a centre overshoot | Mixed comparison, distinct from B's old-goal bias |
| NUM #11065 | -150 mm written Y change yields 0–0.061 mm final predicted Y change | Retention loss is not demonstrated loss of numerical steering |

Done means the bounded work was executed and recorded, not that a proposed method succeeded. Nine items remain blocked, four remain in review and three are parked. AlphaNIST stays parked.

## Reproduce the drawing

```bash
python3 render-offline.py
```

The renderer uses checked-in JSON, CSV, historical snapshots and standard-library SVG generation. `offline-validation.json` checks coverage, links, bounds and card alignment. `served-validation-current.json` records the separately verified live assets. Offline rendering makes no live-service changes. The included snapshot JSON files supply historical result labels; omitting them prevents exact regeneration.

## Programme continuity

See `CONTINUITY.md`. This report closes a drawing revision only. The authorized research programme is **active**. Completing a capped diagnostic is a review point, not permission to close the overall goal. Planner review of RET-C/NUM was sent to Hm challeng p2 at 13:45 UTC; its next protocol is pending at this snapshot.

All seven new experiment branches were pushed privately; their raw evidence has verified Mint fast-drive backups. Small temporary Mac QA screenshots are removed after copying. Models, videos and raw arrays are kept off the Mac.
