# Typhon opportunity solution tree — 14 October

99 work items across ten views. Each experiment has a separate result or blocker to its right. Done means the bounded investigation finished, not that the method succeeded.

Open [recovery](recovery.svg), [generalisation](generalisation.svg), [overview](overview.svg) or [the complete tree](full-tree.svg). CSV uses id,type,label,parent_id,x,y,color. JSON adds bounds and edges; Mermaid preserves the hierarchy.

## Latest results

- LOAD04 counterbalancing improved requested destination selection from 8/16 to 16/16. Strict loading remained 4/8.
- LOAD05-R2 and LOAD06 isolated a local acquisition rescue on development family 0012. A same-state C pose substitution reduced tilt at supported lift from 62.76° to 1.12°. This was acquisition only.
- DATA07-R tested the four eligible loading training families. Native B passed 3/4; its sole failure remained a failure after the C pose substitution followed by B: 0/1 conversions. No successful corrective records qualified. [Report and matched video](https://github.com/NatMota/typhon-robotics/tree/8bab3bb2e08c3263d56e7680f9a0c1bf98de8d8c/experiment-history/2026-10-14-data07r).
- DATA07-GE handed that exact pre-contact state to the unchanged geometric teacher. It placed and released in the fixture (1.22 mm position error; 0.016° yaw error), but left the brick sideways (90° tilt). Strict result: 0/1. Its height-only lift flag accepted the sideways grasp; the independent full-task score rejected it. [Report and video](https://github.com/NatMota/typhon-robotics/tree/696d023889f7b89f8ea037a931e125d117b665d6/experiment-history/2026-10-14-data07ge).
- The donor/teacher search stopped after that single expert attempt. No new training ran. Before corrective fitting, upright acquisition needs qualification on learner-reached states.
- ALPHA03 #11051 remains **Parked by user request**. No broader AlphaNIST implementation started.

B/C name the LOAD04 checkpoints, not task labels or the older V1–V4 models. These small diagnostics do not estimate generalisation.

## Live view and preservation

[Open the recovery tree on the Mac](http://127.0.0.1:5176/llm-design-tool/ui/index.html?project=typhon-research&file=opportunity-tree&page=recovery&artboard=recovery-tree).

Mint access recovered. The interrupted docs worktree, index/worktree patches and canonical directory were preserved before creating a fresh continuation worktree. SHA256 verification passed for:

| Mirror | Files | Raw bytes |
|---|---:|---:|
| Earlier LOAD05-R/R2/LOAD06 evidence | 45 | See receipt |
| DATA07-R raw | 809 | 347,341,705 |
| DATA07-GE raw | 223 | 161,891,895 |

All 99 tickets occur once in the full tree. Cards have checked bounds and no overlap; results align to the right. All ten installed SVGs were fetched over HTTP and matched their exports byte-for-byte. The native browser recovery view was visually inspected.

Current evidence: board-20261014-99-final.json, refresh-input-complete.json, offline-validation.json, refresh-validation.json, served-validation-current.json and recovery-viewer-final99.png. The earlier 98-item screenshot and snapshots remain historical. Verification receipts are in verification-receipts.

Canonical Mint directory: /media/nm/Fast Drive/Typhon/research-20261014/ost-update.
Fresh Git worktree: /media/nm/Fast Drive/Typhon/research-20261014/ost-restored-git-worktree.
Private branch: docs/ost-20261014-continuation.

## Reproduction

Run python3 render-offline.py to regenerate SVG, CSV, layout JSON and Mermaid without installing them. refresh-svg.py explicitly updates the live Design app; validate the served bytes afterward.

Bulk camera/state arrays remain on Mint Fast Drive. Code, reports and representative videos are on the private research branches. Frozen M0/M1/M1.5 and prior experiment worktrees remain unchanged. Fresh-machine restoration remains untested.
