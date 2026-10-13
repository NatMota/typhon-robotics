## EXECUTE NEXT: VIDEO01-C #11018 — review what the complete-video representation adds beyond endpoints

**Close VIDEO11 and the observer-tuning branch. Do not run another controller variant.**

Choose the existing **VIDEO01-C representation review**, using the recordings and already-extracted evidence. **Cap it at 45 minutes. No new physics, calibration, feature-model training or VLA training.**

The question is now:

> **What useful instruction or state distinction does Nat’s complete video provide that the initial/final observations do not—and is that information actually retained by the existing representation?**

This is the missing link between “the robot used a video-derived signal” and “using the complete personal sequence improved the result.”

## 1. Accept VIDEO11’s conclusion, with the correct scope

Your audit supports closing the measurement dispute:

| Finding | Accepted conclusion |
|---|---|
| Visible-component centroids and projected complete-mask centroids are different features | Their disagreement cannot automatically be attributed entirely to detection error |
| Analytic plane substitution accounts for most of the defined numerical discrepancy | Carried-plane conversion is strongly implicated; exact rendered-support/occlusion attribution remains unqualified |
| The 120 µm scorer correction leaves the ranking unchanged | That defect does not rescue a sequence-fidelity advantage |
| VIDEO10 physical scores and exact repeat remain valid | Assisted execution succeeded |
| Endpoint/static conditions also succeeded | Intermediate video was not necessary for completion in this tested setting |

**Reject the claim of a demonstrated full-sequence physical-path advantage—not the possibility that full video could be useful.** Do not turn this into another attempt to make the SEQ metric win.

## 2. Review the actual temporal information, not another proxy score

Use the **16 curated takes**, their original paired clips, existing annotations, object/hand tracks and frozen features. Preserve the exclusion of **demo0017**. A take unavailable for metric reconstruction can still contain useful visual evidence; do not automatically exclude it from this review.

For each take, distinguish these three categories:

| Category | Example of the distinction | How to treat it |
|---|---|---|
| **Task-relevant intermediate constraint** | An independently established requirement to pass through an intermediate state, respect an ordering, or respond to a changing condition | Potential justification for a substantive temporal robot test |
| **Observed motion or style** | Pause duration, simultaneous translation/rotation, curved path, hand withdrawal | Information absent from endpoints, but not automatically required for the placement goal |
| **Unresolved observation** | Apparent pause caused by occlusion; uncertain contact; unknown elevated motion | Preserve as unknown—not a robot command or a task constraint |

**A pause occurring in a demonstration does not, by itself, mean the robot is required to reproduce it.** Likewise, a human hand in the scene is not automatically evidence of an externally imposed intervention or an obstacle-avoidance requirement.

Nat may legitimately value reproduction of motion style. But that would be a **motion-imitation objective**, distinct from improving completion of the existing inside-square/upright-number task. Do not silently change objectives to manufacture a full-sequence benefit.

## 3. Exact review procedure

### First 25 minutes: build the source-evidence ledger

Use existing annotations to locate candidate intervals, then inspect the original paired footage. Record which intervals were actually reviewed; “not established within the review” is not “absent from every frame.”

Produce one row per take:

| Field | Required content |
|---|---|
| Initial/final information | What the endpoints visibly specify |
| Additional temporal information | What an intermediate interval adds |
| Source evidence | Exact raw frame/timestamp references in both views, with synchronization uncertainty |
| Task relevance | Required constraint, observed style/state evidence, or unknown |
| Authority for the requirement | Existing user instruction, task definition, or directly supported physical necessity—not an invented intention |
| Robot consequence | What decision would differ if that temporal information were available |
| Current representation support | Directly retained, indirectly suggested, missing, or untested |

Use **demo0007 as the negative-control example**: it contains intermediate motion, but VIDEO10 did not establish that those intermediates were needed for task completion.

For any stronger candidate, state the counterexample plainly:

> “A robot could satisfy these same endpoints while doing ______, but that would violate the independently established requirement ______.”

If the second blank cannot be supported, classify the observation as additional motion detail—not a task requirement.

### Next 15 minutes: check the representation evidence already available

Select **at most three candidate intervals**, in ascending demo-ID order among those meeting the documented criteria. Do not choose them based on which produces the most favorable feature behavior.

For each, inspect whether the existing object tracks, hand outputs or frozen-feature results actually preserve the relevant distinction.

Keep these evidential levels separate:

| Existing evidence | What it supports | What it does not support |
|---|---|---|
| Cross-view time matching | Correspondence between recorded views | Robot action decoding or task relevance |
| Ordered take retrieval better than pooled retrieval | Order helps that retrieval procedure | Necessity of the complete sequence for manipulation |
| Agent-readable event in the source clip | The footage contains the event | The model represents or detects it |
| Object/hand feature change | The representation changes | Correct interpretation of the event |
| Independently evaluated event/phase distinction | Evidence of a particular retained state distinction | A general video-to-robot adapter |

**Do not build another classifier or tune a similarity threshold during this review.** Where the semantic readout has not been tested, mark it untested.

This is a meaningful acceptance criterion for a temporal representation. Time-Contrastive Networks explicitly target visually similar but functionally different states across views; Temporal Cycle-Consistency work evaluates action-phase information in addition to temporal alignment. Neither treats correspondence alone as proof of robot control. [arXiv](https://arxiv.org/abs/1704.06888)

### Final five minutes: issue the acceptance decision

Choose the supported outcome:

| Review outcome | Decision |
|---|---|
| A task-relevant intermediate constraint is visibly established **and** the existing representation retains the needed distinction | A substantive temporal-information test is justified for that specific constraint—not another generic correction rollout |
| A constraint is visible, but the representation’s support is missing or untested | The next bottleneck is representation of that fact, not controller calibration |
| Only motion-style differences beyond endpoints are established | Report them as potential imitation targets; do not claim added task information |
| No defensible intermediate requirement is established within the review | Close the current “complete sequence improves this task” claim as **not demonstrated** |

**Do not create an obstacle, required pause, waypoint or safety rule merely to make the existing recordings appear to contain one.**

## 4. What the OST should say afterwards

The review should update the existing ticket and parent VIDEO01 with a claim-to-evidence table:

| Claim | Current status before this review |
|---|---|
| Personal footage supplies appearance, observed configurations and provenance | Supported by the recorded pipeline |
| Complete sequences can be retained and used as time-varying adapter inputs | Implemented |
| The assisted adapter can physically execute the correction task | Demonstrated on the VIDEO10 development case |
| Complete video improves physical trajectory fidelity over endpoint/static controls | **Not established** |
| A generic representation recovers human motor actions or transfers across embodiments | **Not established** |
| Intermediate video supplies an additional, actionable task constraint | **The specific question this review must settle** |

Keep the current representation assets even if the stronger claim fails. A failed acceptance claim does not make the recordings or extracted features useless.

## Boundaries and deliverable

Deliver **one 16-row evidence ledger, at most three timestamped evidence examples, and one acceptance decision**. Store it under the existing VIDEO01 review work, not as another controller experiment. No changes to references, goals, trajectories, frozen evaluations or training splits.

VIDEO11’s **75-file verified backup** and VIDEO10’s **552-file preservation check** can be recorded as complete according to your receipt. Publication remains unchanged.

**The reason to choose this review is practical: we have demonstrated that the supplied adapter can solve the endpoint task. Before spending more time on it, identify exactly what additional robot decision the complete personal movie is supposed to teach. If the existing recordings do not establish that distinction, say so and close that claim rather than extending the observer loop.**
