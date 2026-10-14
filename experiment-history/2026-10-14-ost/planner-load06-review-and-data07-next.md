**LOAD06 supports a specific local causal conclusion: replacing B’s pose sequence with C’s—while retaining B’s gripper schedule—prevents tipping and permits upright acquisition through the following 60 steps of B control.**

| First-chunk pose / grip | Maximum tilt, first 30 steps | Supported-lift tilt | Interpretation |
|---|---:|---:|---|
| B / B | 62.35° | 62.76° | Original failure |
| B / C | 62.35° | 61.04° | Donor gripper schedule does not repair it |
| **C / B** | **1.49°** | **1.12°** | Pose substitution alone is sufficient here |
| C / C | 1.36° | 1.06° | Consistent positive control |

**The donor gripper schedules differ only at tick124, after tipping begins at110. That difference cannot explain the onset.** This does not rule out every hypothetical earlier-closing intervention, but it removes the justification for pursuing one now.

We have isolated **the pose sequence as a sufficient intervention**, not XYZ versus wrist orientation, descent speed versus lateral offset, or a universal acquisition fix. The result also stops at acquisition—not final placement.

## Next OST priority: training-family acquisition-repair qualification

**Queue a linked DATA07/recovery follow-up; keep the original reviews and LOAD06 closed. Do not launch it tonight automatically.**

The question should be:

> **Does this acquisition failure occur on training-eligible learner states, and can a one-chunk pose correction repair it while B completes the requested task?**

That is more valuable than further dissecting the same development example before establishing whether it supplies a legitimate training opportunity. Corrective imitation is useful when supervision addresses states the learner actually visits, but the corrective actions themselves must be appropriate. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v15/ross11a.html?utm_source=chatgpt.com)

### Proposed bounded scope

| Stage | Scope |
|---|---|
| **Native source check** | B on the four exact LOAD04 loading-training configurations: **0001/0002/0003/0005**, fixture visible, loading instruction, one frozen policy seed |
| **Conditional repair** | For at most the first **two pre-acquisition tipping failures**, in frozen demo-ID order, qualify the same **C-pose/B-grip first-chunk substitution**, then return to B |
| **Evaluation** | Continue to the original terminal horizon; require upright acquisition, correct destination, seating, release and stability |
| **Cap** | **45 minutes; four native episodes plus at most two repaired continuations**, using the existing qualified prefix machinery |

Retain every native result. **If these source cases do not exhibit the failure, stop rather than manufacture a training problem or import demo0012.** Keep loading-development `0012/0013` and all LOAD05/06 hybrid traces quarantined.

## What would justify targeted training?

Not this single repaired development state alone. I would require:

| Requirement | Why |
|---|---|
| Verified repairs on **more than one training family** | Evidence beyond a one-scene patch |
| Successful full-task continuation, not merely an upright lift | Acquisition improvement must not conceal a downstream failure |
| Exact observation/action alignment and documented intervention provenance | C is a donor policy, not automatically an expert oracle |
| Preservation of LOAD04’s task/visibility counterbalancing | Do not restore C’s visual shortcut while teaching its useful approach |
| A matched extra-update control and frozen evaluation panel | Separate corrective-data benefit from additional optimization |

Any admitted data would be labelled **qualified, policy-generated corrective supervision**, not human actions or a general geometric-expert demonstration. No training is authorized by this recommendation.

**For Nat: we have narrowed the failure from “loading placement goes wrong” to a particular approach-motion sequence. A brief pose substitution fixes acquisition while preserving B as the continuing policy. The next useful question is whether that repair can be demonstrated on legitimate training states—not whether we can keep improving the same inspected example.**

Seal LOAD06 and confirm its independent raw-evidence backup first. Mint remains unreachable; the private download/SHA verification should be reported complete only once it actually passes.

Queued as DATA07-R #11050. No new experiment or training started.
