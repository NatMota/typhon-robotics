# Active Typhon programme
Updated: 16 October 2026, 00:11 UTC (01:11 BST).

The overall research goal is ACTIVE. Finishing an experiment does not finish the programme. Continue through the planner and the unfinished OST work; do not wait for Nat merely because a bounded experiment ends.

## Current constraints and next action
- Work solo. No paid pods. AlphaNIST #11051 stays parked.
- Run robot simulation/training on the authorized UCL workstation, not Mint. Mint Fast Drive holds archives, worktrees and reports.
- Preserve original personal holdouts, quarantines, previous scores and frozen outputs.
- Conservative Powerglide cutoff: 16 October 01:00 UTC / 02:00 BST. No extension is verified. Archive before the cutoff.
- Current planner instruction: after INIT10-A, preservation and nonphysical consolidation only. No automatic second depth, stronger grip, full-horizon continuation or training.
- Latest planner update was sent at approximately 00:10 UTC. Read its completed reply next; do not infer its recommendation.
- Planner: Hm challeng p2, https://chatgpt.com/c/6ac8f7df-fef0-83eb-98c3-48bd796980a1, Chrome tab 1373571506.
- Use the mac-chrome-bridge skill and verified nat.c.mota / Default profile. Keep normal Chrome running.
- Next useful work: source-linked decision/evidence index and a separately frozen future handback-to-Q1 protocol, subject to planner review. Existing work is preserved.

## Latest results, in plain language
1. INIT07 compared two equally sized supervised continuations of the same parent T model. Q0 rehearsed original starts; Q1 used paired corrective examples at states the learner had actually reached. Q1 preserved all eight already-completed feeding cases, but completed only four of eight loading cases. Q0 completed all eight loading cases but disturbed all eight completed feeding cases. Neither passes the full repair gate.
2. INIT08/09 traced two of Q1's lost loading cases to shallow acquisition. The brick lost support while the gripper was still commanded closed, about 1.6 seconds before the opening command. Q0 later reacquired; its recovery states were different, so that is not a matched recovery comparison.
3. INIT10 replayed two failing Q1 grasp segments exactly and compared them with the same saved targets lowered by 5 mm. Both original segments lost the brick; both lowered paths retained an upright lift around 140 mm. Gripper commands were identical. Placement and release were not tested.
4. INIT10-A qualified the saved force records and compared fixed intervals. Lowered paths retained lower, loaded pad contacts. Original paths lost enclosure. Initial net gripper force pushed downward in both conditions; one failing native interval had more upward force than its successful comparison. Therefore force magnitude alone is not the explanation. The evidence supports better enclosure/contact support, not a unique friction mechanism.

No result establishes that the policy recognizes the blue numeral, a universal 5 mm correction, a complete task repair, or new training-label qualification.

## Verified artifacts and private commits
Base archive directory: /media/nm/Fast Drive/Typhon/research-20261015

| Work | Archive directory | Files / bytes | Private commit |
|---|---|---|---|
| INIT07 | init07-paired-state-repair | 6,068 / 8,894,923,975 | c3e41920238b394c08d2a604b9df703c450a6484 |
| INIT08 | init08-acquisition-crossover | 56 / 40,288,650 | dcd8793d0a16c231b537678d3afaaeba817341e4 |
| INIT09 | init09-contact-retry | 51 / 17,636,798 | 9124ce11186fe9ad889e712f0a5c267aa8bab07b |
| INIT10 | init10-lower-path | 122 / 29,103,906 | 276deeae2bfdc3773700e36b78e99047c7824237 |
| INIT10-A | init10a-force-enclosure | 34 / 2,118,953 | dcf9657b1726798b183f47550fbe1773c90f45bc |

Each archive has raw-manifest.json and raw-verification.json with zero SHA errors. These are verified source-file counts; destination transfer receipts add files.
INIT10-A source manifest SHA: 48a023dd8c10ff3c4c6ad948136cf8873acf9823f6817a42ac4671bec6718657.
INIT10-A scientific commit: 8dd42960abb339bf71ab38052c97a65864333a38.
INIT10-A private branch: research/init10a-force-enclosure.
Reports/code are in corresponding Mint *-worktree directories and private GitHub branches; bulk data/weights remain on Mint.

Q1 checkpoint SHA: 26d573eb90f791d9af86bbff30e138f51c5aa658ac7abefe76787ba352363a7a.
Q0 checkpoint SHA: 54c86e744e0c66cd5a55f81b56b949cd71000ed01726125550e878dcfaed081c.
Both weights are verified on Mint. Full optimizer moments were not archived; do not claim full resumability.
Parent T SHA: 4d7a4a2547f084b6026164f27a6b7d43b5cb4d3b83ee4875343d108908f35636.
INIT02-R restore is paused. Do not restart it without a concrete need.

## Source locations and reproduction
Powerglide raw INIT10: /var/tmp/typhon-init10-20261015 (frozen).
Powerglide raw INIT10-A: /var/tmp/typhon-init10a-20261016 (frozen).
INIT10-A worktree: /var/tmp/typhon-worktrees-20261015/init10a.
Its analysis script is scripts/init10a_force_audit.py. Later read-only reproduction uses --audit-context pointing at the archived addendum; it never resets the original 20-minute clock.
Original input SHA preservation and independent read-only reproduction receipts are included.

Powerglide home quota is full. New experiment Git objects used /var/tmp/typhon-research-git-20261016.git. Do not delete user files to make space.
For Powerglide access through Mint, use the existing /var/tmp/typhon-ucl-archive-pull/ssh_config and ucl-powerglide-transfer alias. Forward an ephemeral agent with the existing UCL key; do not copy private keys.
The VirtualBox XML warning during SSH is unrelated; do not repair unrelated VM configuration.

## Media qualification
One INIT10 native case0005 render is defective and retained as invalid presentation evidence. Suspected context contamination is not established.
Qualified comparisons use explicitly labelled archived INIT07 native imagery matched to the identical physical trace and capture ticks, paired with new D5 imagery. See INIT10 analysis-qualified/media-provenance.json.
Do not present archival native frames as newly captured INIT10 frames.

## OST and remaining work
All 129 tickets appear once in the full tree, with result/blocker boxes to the right. Ten live SVGs match saved exports; lower recovery rows were visually inspected.
Board: 112 Done, 9 Blocked, 5 Review, 3 Parked. A Done diagnostic is not a passed policy.
Private OST branch docs/ost-20261015-overnight, commit 77c6537ad588bfda01738789f1405cfa38ecefd0.
Snapshot: experiment-history/2026-10-16-ost-init10a.
Live: http://127.0.0.1:5176/llm-design-tool/ui/index.html?project=typhon-research&file=opportunity-tree&page=recovery&artboard=recovery-tree
Current files: ost-update/refresh-input-complete.json, render-offline.py, served-validation-current.json, visual-review-129.json.
Full-tree layer uses a short source pointer to avoid oversized update requests; exported full-tree.mmd and CSV are authoritative current hierarchy.

Unexecuted work remains explicit:
- INIT03 Part B native conditional loss remains unmeasured.
- INIT07 optional eight handover episodes were not run because the frozen repair gate failed.
- EVAL02 still has 68 intervention episodes unrun; 136 baseline episodes are not a completed 204-episode suite.
- Observer/held-state qualification remains blocked; no new veto or phase authority has been granted.
- LIVE01 metadata/provenance review and delivery synthesis still require their own evidence review.
- Any future lower-path handback/full-task qualification needs a frozen protocol and native reference. The current diagnostic trajectories stay quarantined.
