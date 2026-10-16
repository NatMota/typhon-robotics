## EXECUTE NEXT: HIER02-LIVE — continuous feeding→loading, native versus assisted Q1

**Move to task composition. Do not tune clearance or revisit the untouched prior-start failure now.**

INIT14 establishes that the frozen combination works on the remaining source cases and the inspected archived handover. **It does not establish uninterrupted workflow execution.** The next test should replace the supplied feeding prefix with an actual feeding-policy execution, then start loading from whatever state that policy leaves.

One important distinction: **native Q1 also succeeded on the archived demo0012 handover.** Therefore, the new comparison must determine separately whether the updated loading policy composes successfully and whether D5+C adds anything.

## 1. Keep INIT14’s qualification boundaries intact

| Finding | Disposition |
|---|---|
| Remaining source loading and preservation | Passed on the tested panel |
| Fixture contacts | Clearance removes the observed D5 contacts in the matched source cases |
| Prior loading | No new strict losses or contacts, but the existing absolute joint criterion still fails on0001 |
| Prior feeding | Three cases unsupported and unexecuted; no general prior-skill qualification |
| Archived handover | Native and assisted Q1 both succeed; this is not yet an uninterrupted workflow |
| One RGB-first pair | Retain its limitation; do not call every pair an exact causal comparison |

The prior0001 policy never proposes closure, so a closure-triggered acquisition rule cannot address that failure. **Do not add a forced trigger.** Likewise, do not invent a fixture-absent contract to complete the omitted prior-feeding panel.

## 2. Freeze eight continuous workflows

Use two already-inspected physical configurations and two seeds:

| Factor | Fixed choice |
|---|---|
| Physical configurations | Exact **demo0012 and demo0013 fixture-present loading scenes** from HIER01-DEV |
| Policy seeds | **912171 and912172** |
| First task | Feeding, executed by **GOAL-varied** |
| Second task | Loading, executed by **Q1 native** or **Q1+D5+C** |
| Total | **2 configurations ×2 seeds ×2 loading modes =8 workflows** |
| Physical segments | **16**, if every workflow reaches both segments |

Checkpoint identities:

```text
GOAL-varied:
d5449244d96e55e60c3342af9fffe2ae64362b74cdf9c3d41e39b74995c9bcdb

Q1:
26d573eb90f791d9af86bbff30e138f51c5aa658ac7abefe76787ba352363a7a
```

Resolve the archived simulator/reset seed for each configuration from its manifest. Do not assume it equals the policy seed.

### The two compared workflows

| Mode | Feeding stage | Loading stage |
|---|---|---|
| **N-chain** | Native GOAL-varied | Native Q1 |
| **A-chain** | Identical native GOAL-varied | Q1 with the **unchanged INIT14 D5+C** |

D5+C is absent during the feeding stage in both modes. This is **explicit caller-task routing between learned specialists**, with an assisted loading option—not a learned hierarchy or a single end-to-end policy.

This pair isolates the loading assistance while retaining the same intended upstream policy. Sequential composition needs its own test because preceding actions determine the next policy’s input distribution. [arXiv](https://arxiv.org/abs/1011.0686?utm_source=chatgpt.com)

## 3. No supplied prefix between tasks

Execute feeding from the normal scene reset using actual GOAL inference. At the fixed segment boundary, request loading in the **same live simulator**.

| Retain unchanged | Initialize for the new request |
|---|---|
| Object/robot state, velocities, contacts, actuator state, controller references and solver state | Loading instruction, Q1 request bookkeeping, query/noise index and first-query measured proprioception |
| Physical table, both destinations, cameras and collision geometry | New independent scoring window |
| Actual last executed gripper state | D5/C once-per-request flags, under their existing initialization contract |

**No world reset, home motion, archived-action substitution, object relocation, extra settling or success-triggered switch.**

Prior feeding-stage closing events must not arm loading-stage clearance. Do not fabricate an OPEN state if the preceding task left the gripper closed. If the frozen wrapper cannot accept the actual handover state, report an unsupported handover—not a quietly repaired one.

After Q1’s first query, preserve its native issued-target history: first nine pose values as float32, remaining eleven zero.

Use the existing verified continuous-request initialization. MuJoCo’s continuation depends on controller/callback inputs and warm-start information as well as positions and velocities; resetting those could manufacture an easier handover. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html?utm_source=chatgpt.com)

## 4. Fixed timing—not completion recognition

Each task receives:

- **1,436 policy actions**
- **24 original passive settling/scoring ticks**

The first scene reset uses its original 40 ticks. There are **no additional reset ticks** before loading. Where the runner counts those initial ticks globally, the complete two-stage guard is **2,960 ticks**; retain explicit segment and global clocks.

The loading request starts at the fixed boundary, **not when simulator truth reports feeding success**. If feeding ends unsuccessfully but no existing runtime/safety stop has occurred, still execute the scheduled loading task and report both stages. Do not evaluate only successful handovers.

Pause simulation while loading the next specialist if necessary. That is computational latency, not additional robot settling time; record it without a real-time claim.

## 5. Qualification gate: 15 minutes

Before outcomes, freeze:

| Gate | Required evidence |
|---|---|
| Models/runtime | Exact hashes, correct original cache contexts, fresh exact probe for each checkpoint |
| Scene compatibility | Both original physical contexts support both tasks; no fixture-absent case is introduced |
| Request transition | Changes no physical/controller state and performs no physics tick or home action |
| Wrapper scope | D5+C activates only within the loading request; rule constants and code match INIT14 |
| Initial pairing | Same initial physical state, actual actor inputs and paired noise across N/A for each configuration/seed |
| Storage | Existing model paths usable without duplicating weights; space for the full planned traces and verified offload |

If a required checkpoint is offloaded, restore it only from its verified archive within the preparation gate. Do not delete unverified evidence or substitute a different feeding model.

**Do not require every new first-stage image to reproduce an older archived episode.** Instead, audit the two current workflows’ common feeding stage and their actual handover states.

Keep the established RGB-first reporting rule. A pair whose common feeding executions diverge before the handover remains descriptive, rather than a clean matched-state intervention comparison. Do not force the states back together or reseed it. Unexplained nonvisual mutations remain qualification failures.

## 6. Score complete workflows and each stage

For each workflow, report:

| Outcome | Required distinction |
|---|---|
| Feeding stage | Original strict success, acquisition and final object/robot state |
| Loading starts unsatisfied | Confirm offline; no completion credit carried over |
| Loading stage | Pickup, strict destination, orientation, seating, release and stability |
| Workflow success | **Both tasks strictly pass in sequence** |
| Contact qualification | Original forbidden contacts throughout each stage |
| Joint qualification | Existing signed-margin audit and original criterion, unchanged |
| Primitive behavior | D5/clearance triggers, issued actions, timeout or premature-close flags |
| Handover | Object pose, robot posture, aperture, visibility/crop validity and effective first input |

A workflow may complete both tasks while failing a contact or joint qualification. **Keep task competence and engineering acceptance separate.** Do not waive a common native excursion merely because assistance did not worsen it.

Compare **four N-chain/A-chain pairs**, reporting gains and losses. These are two inspected families with paired seeds, not four independent human demonstrations or an unseen-session benchmark.

## 7. Decision branches—no retuning

| Result | What it establishes |
|---|---|
| Both modes complete all workflows | Current Q1 can compose with GOAL here; assistance’s contribution is limited to any measured contact/acquisition differences |
| Assisted chains gain complete workflows without losses or new violations | The frozen assisted loading component has useful local workflow benefit |
| Both fail to approach from the new handovers | Archived-prefix success did not transfer to these live terminal states; task-start conditioning remains limiting |
| Both approach, but native loses support and assistance succeeds | Acquisition repair transfers to a live handover in those cases |
| Assistance gains completion but adds another forbidden contact or failure | A trade-off; do not qualify the package generally |
| Feeding itself fails | Report the upstream failure; do not attribute the entire workflow result to Q1 |
| Common feeding trajectories diverge across the pair | Report that handover-state difference alongside outcomes; avoid clean causal attribution to D5+C for that pair |

**Do not add the reverse loading→feeding sequence, another goal, an automatic home primitive or a recovery teacher in this block.**

If composition fails, use at most **ten minutes of existing traces within closeout** to classify the first failed stage as task-start, acquisition, transport/placement, or initialization. No additional calls or intervention variants.

## 8. Clock and preservation

**Start after INIT14’s verified closeout. One 90-minute block:**

| Boundary | Work |
|---|---|
| **T+15 min** | Protocol, model/interface and storage gates qualified |
| **T+60 min** | Eight workflows completed or explicitly blocked |
| **T+90 min** | Independent scoring, source-linked report and verified archive |

Run serially with one live simulator/render context and one resident policy. Reuse existing weights, offload sealed outputs to Mint and preserve all original scores and quarantines. These executions add **zero** trials to EVAL02’s old intervention arm.

**Proceed with HIER02-LIVE.** This moves from “the loading controller works from a reconstructed finished-feeding state” to the more useful question: **can the actual feeding policy hand the task to the actual loading policy, in one uninterrupted world, and does the frozen assistance improve that workflow?**
