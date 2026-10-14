# Typhon opportunity solution tree — 14 October

98 work items across ten views. Each experiment has a separate result or blocker to its right. Done means the bounded investigation finished, not that a method succeeded.

Open [generalisation](generalisation.svg), [recovery](recovery.svg), [overview](overview.svg) or [the complete tree](full-tree.svg). CSV uses id,type,label,parent_id,x,y,color; JSON adds bounds and edges; Mermaid retains the hierarchy.

## Latest findings

- LOAD04 counterbalanced supervision improves requested destination choice from 8/16 to 16/16. Strict loading remains 4/8; its failures tip before acquisition.
- LOAD05-R2 isolates a state–policy interaction: only B acting from its own pre-contact state tips the block.
- LOAD06 swaps pose and grip components from proposals made in that same state. Pose-only substitution yields 1.12° tilt at supported lift versus 62.76° native; grip-only yields 61.04°. B controls the following 60 actions in every branch. This is one selected case and acquisition only, not full placement or model training.
- The repaired pose branch briefly commands OPEN after lifting without measured contact loss. That concern is retained in the report.
- DATA07-R #11050 is queued, not run: test acquisition repair on the four training families, continue through the full task and qualify any corrective data before training. Development families and diagnostic hybrids remain quarantined.
- AlphaNIST-style PPO remains a separate negative pilot. ALPHA03 #11051 records the fuller GPU-parallel implementation, reset-bank, training/evaluation and visual-orientation requirements. It is **Parked by user request**, not prioritised. No AlphaNIST work or training has started; DATA07-R remains the next experiment. See [the deferred ticket](alpha03-deferred-ticket.json).

[LOAD06 report and video](https://github.com/NatMota/typhon-robotics/tree/f6638f99fd0958f79c9a208ccd466983ea6d3b68/experiment-history/2026-10-14-load06) · [Planner review and queued protocol](planner-load06-review-and-data07-next.md)

## Offline export versus live viewer

These 98-item SVGs were generated on Powerglide because Mint stopped accepting SSH connections. The live Design viewer was last verified at 92 items. Its refresh and the new Mint mirrors are pending OPS07 #11049. Do not interpret an offline export as a served-page verification.

Historical *-current.json, *-snapshot.json, refresh-validation.json and served-validation.json remain unchanged from their earlier live snapshots. The authoritative evidence for this export is board-20261014-98.json, refresh-input-complete.json and offline-validation.json.

Canonical live source when Mint returns: `/media/nm/Fast Drive/Typhon/research-20261014/ost-update`

[Live viewer](http://127.0.0.1:5176/llm-design-tool/ui/index.html?project=typhon-research&file=opportunity-tree&page=generalisation&artboard=generalisation-tree)

Before recovery, preserve any uncommitted changes in the interrupted Mint docs worktree. Reconcile this branch, install the SVGs deliberately, and verify served bytes and the viewer. Do not overwrite the interrupted tree blindly.

## Reproduce and preserve

Run `python3 render-offline.py` from this directory to regenerate SVG, CSV, layout JSON and Mermaid without contacting the Design app. The existing refresh scripts operate on the live Mint application and must be run deliberately.

All 98 ticket IDs are covered exactly once in the full tree. Results are aligned to the right, with zero overlapping cards and all cards inside bounds. Generalisation and the updated recovery branch are rasterized and visually reviewed separately.

The new LOAD05-R, R2 and LOAD06 raw archives were independently downloaded from private GitHub and SHA256-verified. Receipts are in verification-receipts; this is not a claim of a completed Mint mirror. Earlier LOAD04 backup and the full Powerglide/labs integrity inventory remain verified as last checked. Fresh-machine restore remains untested.

No model weights or raw videos are added to this OST branch. The repository stays private; frozen experiment outputs remain unchanged.
