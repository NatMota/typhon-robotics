## Nat: we now have evidence of task selection—not yet numerical goal control

**The EVAL02-GOAL conclusion is supported by the reported checks.** In this familiar scene, B performs both feeding and loading when the instructions are consistent. When task wording and coordinates conflict, **all four conflict trials complete the wording-associated task**.

That establishes **local precedence of the qualitative task block**. It does not establish that coordinates are globally ignored: they measurably change the first proposal, but do not change the completed destination here. Nor are the conflict outcomes “instruction-following errors”—the instructions deliberately disagree.

Keep this result separate from the 68 unexecuted EVAL02 intervention trials.

## 1. The strongest scientific story

The LOAD sequence now forms a coherent explanation:

| Stage | What the evidence supports |
|---|---|
| **LOAD01: add loading supervision** | The policy acquires loading competence that equal-budget additional feeding training does not produce. |
| **LOAD02–03: test task selection** | The original mixed-task policy follows fixture visibility rather than the requested task in the tested scenes. |
| **LOAD04: counterbalance visibility** | With matched action examples and optimization, counterbalancing makes the requested destination selectable locally, although some acquisitions still produce side-on placement. |
| **EVAL02-GOAL: separate wording and coordinates** | The counterbalanced policy selects the wording-associated endpoint under deliberate conflicts. Independent numerical-goal control remains unestablished. |

**This is stronger than “more data improved the robot.”** It shows that the structure of supervision affects **which input controls behavior**. The important result is the sequence of controlled comparisons—not an assertion that we have identified the model’s internal representation.

Retain **counterbalanced B as the instruction-selectable task checkpoint**, with C and FL preserved as matched/historical controls. It is not yet a generally reliable manipulation policy.

## 2. Strongest next experiment—only in a future authorized session

### Same-task numerical steering, without contradictory instructions

My first choice is **one small test of requested position within a single task**, not another training run or set of prompt paraphrases.

Use the same qualified training-family scene and checkpoint B. Keep the physical scene, feeding wording, required orientation, height convention and numerical settings fixed. Vary only the requested **block-centre XY position within the feeding region**.

Preselect three physically compatible points: the familiar centre and two offsets. The extreme targets must be far enough apart that **one habitual endpoint cannot satisfy both centring tolerances**, while allowing the whole block footprint to remain inside the region. Qualify those geometric conditions before inspecting policy outputs.

**Proposed size: three targets × two paired seeds = six rollouts.** Both stations remain unchanged; no target-marker relocation, hidden fixture, controller intervention or training.

The distinction from tonight’s test is important: **all three requests satisfy the same qualitative feeding task**. We would no longer ask the policy to choose between contradictory task categories. This follows the same principle of using targeted interventions to distinguish cue dependence from apparently successful imitation. [arXiv](https://arxiv.org/abs/1905.11979?utm_source=chatgpt.com)

Measure the actual endpoint displacement as the numerical target changes, alongside strict physical placement—not just small changes in predicted actions.

- If endpoints move appropriately, that establishes **local numerical steering within the tested task**, not arbitrary-coordinate generalization.
- If they remain at the familiar point, numerical steering remains unsupported under that interface.
- If the revised requests cannot be made unambiguous without changing other conditioning, stop at protocol preparation rather than running an uninterpretable test.

**This is a recommendation for later, not a new queue activation tonight.**

## 3. Keep these boundaries intact

| Branch | Current disposition |
|---|---|
| **Recovery teacher / acquisition repair** | Unqualified from the problematic learner state. No failed suffix becomes positive training data. CTRL05’s gradual-descent conditions were **not executed**, so they are not a negative algorithm result. |
| **Observer authority and live intervention** | Blocked. Capture lineage and offline tensor preservation passed; the live paired-parity test failed. Fixed-context capture did not reproduce that failure. |
| **PPO, residual RL, hierarchical/value selection, AlphaNIST prototype** | Remain blocked or parked under their existing prerequisites. No automatic resumption. |
| **Human-video action alignment and metric calibration** | Unresolved. Assisted video-conditioned execution is real; direct human-action recovery and an incremental complete-sequence benefit remain unestablished. |
| **EVAL02 intervention panel** | The 68 trials remain unexecuted; this diagnostic does not count toward them. |

## Tonight’s finish

Seal EVAL02-GOAL, verify its pending archive, and update the private synthesis with this distinction:

> **Counterbalanced supervision produced useful selection between trained feeding/loading behaviors. In the tested familiar scene, qualitative task wording dominates conflicting numerical destinations; independent metric-goal control has not been demonstrated.**

Preserve the imageio packaging workaround as a media-generation detail, with no implication that any rollout was repeated.

**No further experiment tonight. Consolidate the evidence and finish the backups.**