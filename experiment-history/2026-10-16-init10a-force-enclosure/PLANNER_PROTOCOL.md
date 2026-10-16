## INIT10 establishes a local acquisition-retention repair. Next: one short force/geometry audit of these four traces.

**Both selected failures were converted into upright, retained lifts by the fixed 5 mm lower target path.** The intervention was physically realized: the wrist was approximately **4.94 mm lower**, and the first bilateral pad contacts moved farther below the block’s upper edge.

That is stronger evidence than merely observing shallow contacts in failed grasps.

| Measure | Native replay | D5 |
|---|---|---|
| Sustained supported lift | Fails both cases | Passes both cases |
| Object rise | Peaks approximately 30–35 mm before loss | Approximately 140 mm retained |
| Final tilt | 90° | Approximately 0.4–0.5° |
| First recorded bilateral-contact tick | Same as D5 | Same as native |
| Gripper control sequence | Same | Same |
| Full placement/release | **Not evaluated** | **Not evaluated** |

**The supported causal claim is about the fixed vertical-path intervention in these two selected states.** It does not isolate contact depth from all resulting contact mechanics, establish a universal 5 mm correction, or improve Q1’s recorded native scores.

Identical gripper controls also do **not** establish identical squeezing or contact forces. MuJoCo distinguishes actuator inputs, actuator force and transmitted generalized force; their relationship can depend on the actuator model and physical state. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/computation.html)

## 1. Preserve the media defect without weakening the physical result

Your replacement presentation is acceptable **only as explicitly labelled archival replay footage**.

For the affected native case, the comparison must say:

> **Native imagery: archived INIT07 footage, matched to INIT10-N’s reproduced physical trajectory over the displayed ticks. D5 imagery: captured during INIT10.**

Retain the defective `final.png` and any affected video as invalidated presentation evidence. Preserve original and replacement hashes, source paths, camera identities and tick alignment. **Do not overwrite the defective file or imply the replacement was captured during INIT10.**

The trace-based outcome remains supported by the qualified physical replay. Render-context contamination remains a suspected explanation—not an established cause. No new rendering investigation is needed.

## 2. EXECUTE AFTER VERIFICATION: INIT10-A — contact-force and enclosure interpretation

**Twenty minutes maximum, read-only. Zero simulation, model calls, fitting or new videos.** Begin only after INIT10’s raw archive passes verification.

Use exactly the four completed INIT10 branches and the two predefined cases. The question is:

> **What measured contact changes accompany the depth intervention’s retention benefit, and how far can the new force records explain that benefit?**

### First gate: verify the new force records belong to the right branch

The media defect makes this check important, although it does not establish a force-logging defect.

From saved code and records, confirm:

- Force extraction used the **same branch’s model/data object** as the recorded physical state.
- Contact IDs, geom pairs, contact positions, times and counts correspond to that branch’s trace.
- Gripper actuator identities and units are documented.
- Measurements were taken at the recorded native computation point.

**Do not recompute missing forces by loading states into a new simulator.** If provenance or contact-frame information is insufficient, mark the force interpretation unqualified and proceed with the already-supported kinematic result.

## 3. Compare three fixed intervals—not whichever samples look persuasive

For each pair:

| Interval | Definition |
|---|---|
| **Acquisition** | First bilateral sample through the next two control samples |
| **Native support-loss boundary** | Last bilateral native sample and first native loss sample, compared with D5 at the same control ticks |
| **Terminal retention** | Last three recorded samples of the 60-action segment |

Also retain the complete time series so those summaries do not hide transients.

Report:

| Quantity | Purpose |
|---|---|
| Measured aperture and individual finger motion | Whether enclosure differs despite identical control |
| Contact geom identities and object-local locations | How the contact configuration changes |
| Per-pad normal and tangential forces | Whether loading and asymmetry differ |
| Net finger force on the object, where sign/frame metadata qualify | Whether the recorded contacts provide upward support |
| Object motion relative to the gripper and marked-face tilt | When retained contact becomes loss or tipping |
| Actual actuator force, separately from `ctrl` | Distinguishes commands from delivered actuation |

`mj_contactForce` returns a wrench in the **contact frame**. Transform it consistently and account for which contact geom is the object before combining forces. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/APIreference/APIfunctions.html)

**Do not sum unsigned normal-force magnitudes and call that vertical support.** Likewise, contact-point spread is not contact area. If only 20 Hz samples are available, retain that limitation; unrecorded substep peaks and exact loss timing remain unknown.

Avoid a new friction-margin calculation or force-balance fit. The contact model’s friction representation and dimensionality matter, and sparse samples cannot automatically identify friction as the limiting mechanism. [Mujoco Documentation](https://mujoco.readthedocs.io/en/2.3.7/computation.html)

## 4. Acceptance and interpretation

The addendum is complete when it produces **one paired table and one evidence-qualified mechanism statement**.

| Finding | Defensible conclusion |
|---|---|
| D5 retains lower pad contact locations and sustained supportive forces while N loses them | Consistent with improved enclosure/contact support under the lower trajectory |
| Actuator forces differ despite identical controls | Part of the physical response differs; no stronger closing command was supplied |
| Similar force magnitudes, but different contact locations and retention | Geometry/contact organization matters; magnitude alone does not explain the contrast |
| Force records cannot be qualified | The force mechanism remains unresolved; the intervention’s measured retention effect still stands |
| Results differ between the two cases | Report separate mechanisms rather than forcing a common explanation |

Even a clean force contrast would not prove that a particular friction coefficient, pad edge or force component was the **sole** cause. The experiment changed the vertical trajectory, and several contact consequences may mediate its effect.

## 5. Time and next engineering decision

**Finish this addendum by 00:45 UTC at the latest**, leaving at least 15 minutes before the conservative 01:00 cutoff. If verification finishes too late to allow the audit, write the compact intervention/evidence index and leave force interpretation pending. Do not shorten the preservation checks.

No automatic second depth, stronger grip, full-horizon continuation or training follows.

The next **future** qualification—once time and compute are explicitly available—would be whether the acquisition intervention can hand control back to Q1 and complete the original task. That would require a frozen transition back from the offset path and a paired native reference. **The current truncated success is not yet a qualified corrective training suffix or a deployed policy fix.**

**For Nat: this is a concrete positive mechanical result. The failed policy grasps were locally rescuable without changing its gripper command—by placing the fingers about 5 mm lower. Now finish the saved-force interpretation and preservation, rather than dilute that finding with an unqualified last-minute extension.**