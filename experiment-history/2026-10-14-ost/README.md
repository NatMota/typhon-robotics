# Typhon opportunity solution tree — 14 October

98 work items across ten views. Each experiment has a separate result or blocker to its right. Done means the bounded investigation finished, not that a method succeeded.

Open [generalisation](generalisation.svg), [recovery](recovery.svg), [overview](overview.svg) or [the complete tree](full-tree.svg). CSV uses id,type,label,parent_id,x,y,color; JSON adds bounds and edges; Mermaid retains the hierarchy.

## Latest findings

- LOAD04 counterbalanced supervision improves requested destination choice from 8/16 to 16/16. Strict loading remains 4/8; its failures tip before acquisition.
- LOAD05-R2 isolates a state–policy interaction: only B acting from its own pre-contact state tips the block.
- LOAD06 swaps pose and grip components from proposals made in that same state. Pose-only substitution yields 1.12° tilt at supported lift versus 62.76° native; grip-only yields 61.04°. B controls the following 60 actions in every branch. This is one selected case and acquisition only, not full placement or model training.
- The repaired pose branch briefly commands OPEN after lifting without measured contact loss. That concern is retained in the report.
- DATA07-R #11050 completed: B passed3/4 native training-family loading trials. The sole eligible failure also failed after one C-pose/B-grip chunk followed by B (0/1 conversions). No successful corrective records qualified and no training started. [Report and matched video](https://github.com/NatMota/typhon-robotics/tree/8bab3bb2e08c3263d56e7680f9a0c1bf98de8d8c/experiment-history/2026-10-14-data07r).
- AlphaNIST-style PPO remains a separate negative pilot. ALPHA03 #11051 records the fuller GPU-parallel implementation, reset-bank, training/evaluation and visual-orientation requirements. It is **Parked by user request**, not prioritised. No AlphaNIST work or training has started; The bounded DATA07-R diagnostic is now complete. See [the deferred ticket](alpha03-deferred-ticket.json).

[LOAD06 report and video](https://github.com/NatMota/typhon-robotics/tree/f6638f99fd0958f79c9a208ccd466983ea6d3b68/experiment-history/2026-10-14-load06) · [Planner review and queued protocol](planner-load06-review-and-data07-next.md)

## Offline export versus live viewer

Mint access has returned. LOAD05-R/R2/LOAD06 mirrors and the DATA07-R raw archive are SHA256-verified. The interrupted Mint worktree and canonical directory were preserved before recovery. The live Design viewer was last verified at92 items; installation and served-page checks for these98-item exports are in progress under OPS07 #11049.

Historical *-current.json, *-snapshot.json, refresh-validation.json and served-validation.json remain unchanged from their earlier live snapshots. The authoritative evidence for this export is board-20261014-98-data07r.json, refresh-input-complete.json and offline-validation.json.

Canonical live source when Mint returns: `/media/nm/Fast Drive/Typhon/research-20261014/ost-update`

[Live viewer](http://127.0.0.1:5176/llm-design-tool/ui/index.html?project=typhon-research&file=opportunity-tree&page=generalisation&artboard=generalisation-tree)

Before recovery, preserve any uncommitted changes in the interrupted Mint docs worktree. Reconcile this branch, install the SVGs deliberately, and verify served bytes and the viewer. Do not overwrite the interrupted tree blindly.

## Reproduce and preserve

Run `python3 render-offline.py` from this directory to regenerate SVG, CSV, layout JSON and Mermaid without contacting the Design app. The existing refresh scripts operate on the live Mint application and must be run deliberately.

All 98 ticket IDs are covered exactly once in the full tree. Results are aligned to the right, with zero overlapping cards and all cards inside bounds. Generalisation and the updated recovery branch are rasterized and visually reviewed separately.

The new LOAD05-R, R2 and LOAD06 raw archives were independently downloaded from private GitHub and SHA256-verified. Receipts are in verification-receipts; their new Mint copies have also been SHA256-verified (45 files). Earlier LOAD04 backup and the full Powerglide/labs integrity inventory remain verified as last checked. Fresh-machine restore remains untested.

No model weights or raw videos are added to this OST branch. The repository stays private; frozen experiment outputs remain unchanged.
