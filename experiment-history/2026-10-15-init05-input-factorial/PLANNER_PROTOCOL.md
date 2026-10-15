## EXECUTE NEXT: INIT05 — separate image-driven from proprioception-driven initiation

**Choose the input-factor test first, using q0 and q1—not q3/q4.** The first substantial approach proposal appears at q1, while the supplied-versus-measured wrist gap is only **0.64 mm**. That makes a large tracking discrepancy an unlikely complete explanation for the onset. The later crop failure also occurs too late to initiate that first approach.

The useful question is:

> **Does the change from q0 to q1 produce unwanted approach primarily through the images, the effective proprioception, or their combination—and does the instruction alter that dependence?**

This requires the existing inference path, not the still-unqualified native-loss wrapper. **T only; 16 generated chunks maximum; zero physics or training.** Keep R’s restore paused.

## 1. What INIT04 establishes

The instruction remains intact, yet stronger hold wording does not prevent approach and closure in the later packets. That supports **state-dependent behavior with inadequate tested hold compliance**, not a timer-triggered transition or a substituted loading request.

However, the initial approximately 11–12 mm departure is not a successful stationary policy merely because it is much closer to the hold labels than to the loading labels. **The policy’s own small initial motion changes what it observes next.** Sequential imitation can consequently leave the states represented in demonstrations even when initial predictions look relatively close. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v15/ross11a.html)

INIT03’s native-loss question remains open. We should not label this teacher-forcing failure, insufficient capacity or an image-recognition failure without further evidence.

## 2. Freeze two packets at the earliest divergence

Use the already-selected **T feeding-preservation execution, demo0001/seed912171**:

| Packet | Meaning |
|---|---|
| **q0** | Exact task-start observation |
| **q1** | First subsequent native observation, before the approximately 127 mm departure proposal |

Keep the exact T checkpoint:

```text
4d7a4a2547f084b6026164f27a6b7d43b5cb4d3b83ee4875343d108908f35636
```

Define two input blocks:

- **I:** the complete actual processed visual input, including both views, crop slot and validity masks.
- **P:** the complete **effective native proprioception tensor**, after the preprocessing actually used by the policy.

Do not select a later packet because it produces a more dramatic effect. Do not recover q1 by nearest-frame substitution or rerendering.

### Preparation gate: twelve minutes

Confirm that all effective actor inputs other than **I, P and the deliberately changed instruction** are equal or explicitly held fixed: domain, duration fields, inference settings and any auxiliary inputs.

Record q0/q1’s XYZ difference, decoded rotation difference, raw 6D norms/conditioning and measured-pose relationship. **Do not normalize or repair P.** If the effective proprioception includes representation differences as well as pose differences, they remain part of this block-level intervention.

Check the q0/q1 crop masks. If they differ, image-block effects include that mask change; do not call the result a pure pixel-content effect.

## 3. The fixed 16-chunk matrix

| Condition | Images | Effective proprioception | Role |
|---|---|---|---|
| **00** | q0 | q0 | Original starting state |
| **10** | q1 | q0 | Change visual input only |
| **01** | q0 | q1 | Change proprioceptive input only |
| **11** | q1 | q1 | Original later state |

For each condition, use:

- The exact original **feeding-preservation instruction**.
- The exact corresponding **loading instruction**.
- The same **two saved q1 action-noise tensors**, from seeds912171/912172, reused across every condition.

\[
4\text{ input conditions}\times2\text{ instructions}\times2\text{ noises}
=\boxed{16\text{ outputs}}.
\]

Do not repeat H-explicit: INIT04 already tested its failure locally. Keeping this matrix to the two trained instruction forms makes the result easier to interpret.

**Conditions10 and01 are deliberately mixed-time input interventions.** They need not describe a physically realizable complete state. They can demonstrate input dependence; they cannot establish that executing their outputs would be a safe or effective repair. Targeted interventions help distinguish correlated cue use from the interpretation suggested by ordinary successful or failed trajectories. [arXiv](https://arxiv.org/abs/1905.11979)

## 4. Qualification and execution rules

Run **11 / original feeding / original q1 noise first**. It must reproduce the archived q1 proposal exactly under the established inference settings. This is one of the 16 cells, not an extra rollout.

Then require:

| Check | Requirement |
|---|---|
| Input isolation | Actual tensors differ only in the designated I/P blocks and instruction tokens |
| Paired noise | Exact same two initial noise tensors across all cells |
| Native processing | No duplicate gripper sigmoid, new normalization, channel reinterpretation or changed action decoder |
| No hidden mutation | Model parameters and persistent buffers remain unchanged |
| Original evidence | Archive arrays, masks, instructions, outputs and earlier qualification failures remain untouched |

Changing only an upstream field that native preprocessing subsequently removes is not a valid intervention. Save and compare the **effective tensors received by the model**.

If the no-op reproduction or input isolation fails, stop inference and report that failure. Do not substitute approximate equality or another packet.

## 5. Report absolute proposals as well as relative errors

For each cell, retain:

| Measurement | Purpose |
|---|---|
| Absolute first/final XYZ targets and whole-chunk XYZ sequence | Establishes whether predictions actually change |
| Maximum and final departure from the supplied P position | Measures the model’s proposed motion relative to its input |
| The same quantities relative to the fixed measured q1 wrist | Gives a common physical reference for interpreting all four outputs |
| Proposed distance toward the actual block | Distinguishes generic motion from approach; object truth used offline only |
| Full rotation and gripper probabilities | Detects unwanted orientation or closure |
| Paired image, proprioception and interaction effects | Identifies which input changes reproduce the initiation tendency |

**Changing P also changes the reference used by a relative-error metric.** Do not report a change in “departure from supplied pose” as a prediction effect unless the absolute output changed too.

For any prespecified scalar metric \(m\), report:

\[
\text{image effect at }P_0=m_{10}-m_{00},
\]

\[
\text{proprio effect at }I_0=m_{01}-m_{00},
\]

\[
\text{interaction}=m_{11}-m_{10}-m_{01}+m_{00}.
\]

Show both noise results separately. These are local finite contrasts, not percentages of responsibility for the final manipulation.

Do not compare q1 outputs against q0’s loading demonstration as though it were a qualified action label for q1.

## 6. Interpretation and repair gates

| Pattern | What it establishes | What it would justify next |
|---|---|---|
| **10 reproduces the large approach; 01 remains near the starting response** | The q1 visual block is sufficient to induce the change with q0 proprioception | Qualify preservation under physically valid changed robot-view observations—not freeze old images online |
| **01 reproduces the approach; 10 does not** | Effective proprioceptive conditioning is sufficient locally | Qualify no-action behavior around nearby real wrist states; inspect representation consistency before altering the deployed convention |
| **Only11 produces the strong approach** | The joint image/proprioception combination matters | Real paired-state supervision, rather than an isolated image or pose augmentation |
| Both isolated changes induce approach | Multiple cues can initiate the behavior | Avoid declaring one sole cause |
| All cells remain task-insensitive and the factor contrast is weak | This test does not localize a useful state trigger | Native conditional fitting becomes the next unexecuted question |
| Loading and preservation separate under some cells but not others | Task conditioning depends on the observation state | Target the identified state region rather than add generic language paraphrases |

**None of these outcomes authorizes clamping proprioception, replaying stale images or forcing the gripper open.** Those could suppress valid loading as well as unwanted movement.

### What would justify a narrowly scoped repair experiment?

A data-level repair becomes justified when we can qualify **both desired behaviors at the actual learner-reached precontact state**:

1. Under the feeding request, measured-pose/open-gripper holding preserves the completed arrangement.
2. Under the loading request, a verified teacher can still complete loading from that same state.
3. The two records share correctly aligned initial observations and differ in the intended instruction and subsequent actions.
4. They come only from eligible training-family states, with no development/quarantine import.

That would supply **task-contrasted supervision at the state where T departs**, rather than simply extending a stationary hold demonstration or adding more successful loading motion. A later learner-state collection would also be materially closer to dataset aggregation than INIT01’s expert-terminal-state bank. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v15/ross11a.html)

The factorial test determines which local robustness question to qualify first. **It does not itself create trustworthy labels.**

If no useful factor dependence is found, do not begin a second, larger image/proprioception sweep. Preserve the negative result and prioritize **T-only native-loss qualification** under a separate protocol; R’s restore is not necessary for that.

## 7. Clock and storage

**Separate 40-minute block, after INIT04’s archive verification:**

| Boundary | Required progress |
|---|---|
| **T+12 min** | q0/q1 packets, effective-input contract and protocol frozen |
| **T+20 min** | Original q1 reproduction and isolation gates passed |
| **T+30 min** | All 16 outputs complete |
| **T+40 min** | Analysis and verified compact archive |

Use T in place. Keep R’s partial restore and verified Mint original intact; do not resume it into the remaining scratch. This experiment needs only small packet/output files, not another model copy.

**Proceed with INIT05.** It tests the first amplification of unwanted motion, before contact, crop loss or grasp mechanics obscure the issue. The programme remains active; the result should select a specific repair qualification or the still-open native-loss question—not another indefinite diagnostic matrix.