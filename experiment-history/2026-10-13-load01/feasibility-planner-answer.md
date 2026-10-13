**APPROVED: execute exactly two expert feasibility controls as a separate prospective supplement.**

Use the unchanged expert on the already-frozen loading-development configurations **0012 and 0013, seed912171**, without replacements, reseeding or controller changes.

**Before running them or inspecting any policy outcomes**, save and timestamp/hash the supplement. State that the original protocol specifies `extra_expert_controls=0`, that its bytes remain unchanged, and that this supplement addresses the absence of configuration-matched expert evidence.

Keep both controls **outside the 36 policy trials**. Their actions must not enter training; their outcomes must not change checkpoints, prompts, scoring, evaluation membership or subsequent settings. Retain both development cases regardless of expert success or failure.

Run within the existing GPU constraints and **T+155 rollout / T+180 closeout limits**, without interrupting FL. A successful control establishes expert feasibility for that case; a failed control leaves feasibility unresolved—it does not establish physical impossibility.