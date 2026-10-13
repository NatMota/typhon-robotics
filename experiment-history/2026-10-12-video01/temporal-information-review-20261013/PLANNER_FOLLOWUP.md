## EXECUTE NEXT: DATA07 #10992 — reconcile training-only failure and correction coverage

**Cap: 45 minutes, existing evidence only. No training, new annotations, physical rollouts or observer changes.**

The required output is one actionable answer:

> **Which existing examples could legitimately teach the missing manipulation decisions, which remain quarantined, and which lack qualified labels or provenance?**

This moves the programme beyond repeatedly finding that the observer is unreliable. It also prevents personal-video evidence, generated expert actions and development-state corrections from being treated as interchangeable training data.

## 1. Accept VIDEO01-C as a completed review—with a narrower conclusion

**The review supports “the current representation-to-action link is unqualified,” not “we have identified representation as the sole bottleneck.”**

The footage contains information absent from endpoints: pickup rather than pushing, held transport, and release. But an independently supplied “pick up the block” instruction can already specify that requirement. The review has not shown that the complete sequence is necessary to teach it, or that the extracted features encode it sufficiently for control.

Similarly, demo0018 supports an already-completed-state example, but does not establish that temporal history is needed instead of the current observation and supplied goal.

Two qualifications should stay explicit:

- **Null `is_grasped` or metric-XYZ fields establish missing qualified outputs—not that the images or embeddings contain no relevant information.**
- The 27 nearest-feature-frame matches are suitable for the disclosed exploratory review, not certification of precise grasp/release transition timing.

**Keep #11018 Done as an evidence review.** VIDEO01’s stronger control-benefit claim remains unaccepted; VIDEO09–11 stay closed.

## 2. DATA07: build one provenance-and-coverage ledger

Start by locating the existing DATA07 report and manifests. **If the review already exists, reconcile and finalize it rather than repeating it.**

Use one row per underlying generated episode or correction suffix, linked to its source take and training recipe. Do not count multiple windowing schemes as new human demonstrations.

| Required field | What to establish from existing records |
|---|---|
| **Identity and lineage** | Episode ID, parent take, source/configuration hashes, generator and checkpoint where applicable |
| **Original split and permitted use** | P_train, reserved correction training, validation/development, observation-only, excluded, or unresolved |
| **What the human video supplies** | Appearance, initial state, goal, intermediate observations or reviewed events |
| **What supplies the robot actions** | Existing geometric expert, learner rollout, qualified correction controller, or **no robot-action labels** |
| **Physical qualification** | Verified outcome and criteria; failed, partial and unexecuted examples remain distinct |
| **Temporal/input qualification** | Existing alignment checks, visibility limitations, command-history convention and missing evidence |
| **Actual exposure** | Unique trajectories, derived windows and recorded sampling exposure—separately |
| **Disposition** | Verified existing training material; development quarantine; observation-only; failed execution; missing evidence |

### Reconcile these distinctions explicitly

| Potential conflation | Required resolution |
|---|---|
| **Six P_train takes versus V1–V3’s 18 episodes** | Preserve both lineages. Establish overlap through manifests; do not assume they are disjoint or add their counts together |
| **V4’s three accepted recovery episodes versus later recovery qualifications** | Keep original training additions separate from later development-derived suffixes |
| **Human held-transfer evidence versus simulated contact labels** | State which evidence supplies each label; neither is automatically the other |
| **demo0007’s reserved-correction role versus its later VIDEO09/10 use** | Preserve its historical role and development exposure; do not relabel it as an untouched test |
| **demo0018 already-completed footage versus motor supervision** | Observation-only unless an independently documented robot-label source exists |
| **Missing recovery arrays versus verified source-code hashes** | Code identity does not certify unavailable array contents or their numerical distribution |

Do not change membership, reclassify development cases as training, or manufacture new successful sub-trajectories from failed executions during this review.

## 3. Answer three coverage questions—not another general dataset summary

Use existing labels and telemetry only. Unknowns remain unknown.

| Question | Why it changes the next decision |
|---|---|
| **Do verified training-eligible corrections cover off-axis acquired grasps, or mainly near-expert grasp relationships?** | Determines whether grasp-relative compensation is actually represented in usable supervision |
| **Do they cover learner-visited approach and empty-close failures, or only successful expert approaches?** | Distinguishes teaching recovery from repeating ideal trajectories |
| **Which required decisions have observable inputs and qualified targets—and which only have agent-reviewed video evidence?** | Identifies whether the missing component is a robot-action label, a state estimate, provenance, or coverage |

Report demonstrated gaps, not speculative ones. For example:

> “No verified training-eligible off-axis correction was located in the inspected manifest”

is defensible where

> “The model has never seen an off-axis grasp”

may not be.

The development-derived recovery suffixes can demonstrate that a state is recoverable. **They remain quarantined and cannot silently become the answer to a training-coverage gap.**

## 4. Acceptance and stop criteria

| Outcome | Ticket treatment |
|---|---|
| Known dataset lineages and splits are reconciled, with usable/quarantined/unknown material identified | **DATA07 review complete** |
| Some arrays or manifests are unavailable | Complete the review with a precise missing-evidence entry; do not certify those examples as training-ready |
| No sufficiently qualified training-only corrections exist | Valid negative review result—not permission to collect or train automatically |
| Prior review already meets these requirements | Verify its references and close without duplicating calculations |

**Time allocation:** ten minutes to locate existing evidence, twenty-five to reconcile, ten to report and verify the small output archive. Stop at 45 minutes. Do not restart stopped pods, generate missing labels or reconstruct unavailable trajectories.

Deliver one ledger, a short coverage decision, and exact links to the existing supporting artifacts. **No new training manifest is activated.** Keep the repo private and bulk data on Mint.

## 5. VIDEO01-A/B can close by evidence reconciliation—not new tests

These are administrative reconciliations against completed work, not two additional experimental assignments.

| Review | Reconcile before closure | Supported completion statement |
|---|---|---|
| **VIDEO01-A #11016 — object-path review** | Sequential-decoding provenance; visible-centroid versus rigid-landmark definitions; missing observations; VIDEO08 calibration limits; VIDEO09–11 observer/scorer distinctions | **Object-image representation implemented and audited, with explicit limits. Calibrated human 3D trajectories and superior full-path transfer remain unestablished.** |
| **VIDEO01-B #11017 — hand/pinch review** | Hand-landmark coverage; existing hard-negative results; approach/withdrawal false positives; lack of qualified contact/force/grasp-state outputs | **Hand representation reviewed. It is not qualified to authorize autonomous grasp or release.** |

If their original acceptance wording requires motor-ready trajectories or a validated contact estimator, **do not mark that capability achieved**. Close the *review* and link the unmet capability to the existing blocked tickets—10989 for action alignment or 10984 for observation authority—rather than leaving another open-ended review running.

**The next active work is DATA07. Its purpose is to leave Nat with a concrete choice between using already-qualified evidence and addressing one specific missing dependency—not another vague recommendation to improve perception.**

*I attempted to retrieve the pinned private report and its branch URL through the GitHub connector; both returned 404. This assessment therefore uses your supplied execution receipt, not an independently fetched report.*

