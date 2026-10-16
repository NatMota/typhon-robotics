# Current research results — 16 October 2026

**The two policies now complete feeding then loading in the same live simulation on all four tested start/seed combinations, with or without assistance. Assistance helps some earlier loading cases but adds no successes on this continuous-workflow panel. Broader qualification still fails: a prior loading start remains unattempted with a joint-limit excursion, and fixture-free prior scenes are unsupported.**

This page separates model training from supplied controller assistance. It does not replace the frozen reports or combine different evaluation suites.

## What the personal data contributes

Eight phone recordings supply observed object positions/orientations, task goals, categories and source timestamps. Sixteen examples were curated; fifteen had sufficient information for scenario reconstruction. The simulated RGB and Panda action labels come from executed robot trajectories, including geometric-expert controls. They are not recovered human arm commands or validated frame-aligned phone-to-robot action labels.

The original physical scale and block proxy remain declared assumptions. Later loading experiments use states reached by recorded robot prefixes; the world is continued from that state rather than replacing the block pose alone.

## Native models: the actual trade-off

The following rows use the same four inspected task-start families and two policy seeds. A loading request asks the model to move the staged block into the fixture. A feeding request asks it to leave the already-correct staging arrangement untouched.

| Model | What changed | Loading completed | Completed feeding preserved | Earlier feeding | Earlier loading |
|---|---|---:|---:|---:|---:|
| INIT02-R | Matched rehearsal control without the new loading-initiation labels | 0/8 | 8/8 | See original report | See original report |
| INIT02-T | Added supplied loading-initiation supervision | 5/8 | 0/8 | 2/3 | 3/4 |
| INIT07-Q0 | 3,000 further joint updates; original-state rehearsal | 8/8 | 0/8 | 2/3 | 2/4 |
| INIT07-Q1 | 3,000 further joint updates; paired loading/hold examples at learner-reached states | 4/8 | 8/8 | 2/3 | 3/4 |

All these are **native policy** outcomes, without the D5/clearance assistance described below. T is a reused historical reference in INIT07, not a newly executed control. Repeated seeds are not independent recording sessions.

Q0 and Q1 each used 12,000 presentations and about 16.5 minutes for training, saving and their reload probe. The two runs shared 8,000 presentation identities/positions; 4,000 designated slots compared old-state versus learner-reached-state examples. They started independently from T. This was full-joint supervised adaptation, not PPO, LoRA or proof of a complete DAgger procedure.

Q1 repaired preservation but lost two exact loading successes previously achieved by T. Q0 loaded all eight but still disturbed completed work and lost one earlier loading success. **Neither passed its frozen combined gate.** Appropriate behavior on these cases does not establish an internal completion-recognition mechanism.

Sources: [INIT02 report](https://github.com/NatMota/typhon-robotics/blob/c3e41920238b394c08d2a604b9df703c450a6484/experiment-history/2026-10-15-init02-matched-adaptation/INIT02_report.md), [INIT07 report and full subgoals](https://github.com/NatMota/typhon-robotics/blob/c3e41920238b394c08d2a604b9df703c450a6484/experiment-history/2026-10-15-init07-paired-state-repair/INIT07_report.md).

## Controller assistance: what it adds and what it costs

Q1 is the fixed model in every row below; these are not additional trained checkpoints.

| Test | Compared systems | Observed benefit | Limitation / decision |
|---|---|---|---|
| INIT12/A1, 16 source/preservation cases | Native Q1 versus D5 | Loading 4/8 → 8/8; preservation remains 8/8 | D5 adds forbidden finger–fixture contacts in 4/8 loading episodes. Advancement failed. |
| INIT13, four selected cases | D5 versus D5+C | Both retain 3/3 loading and 1/1 preservation; the two reproduced contact cases become contact-free at recorded samples | Passes only the selected development gate. Five overridden pose commands per retreat; gripper commands unchanged. |
| INIT14, remaining source | D5 versus frozen D5+C | Both load 5/5 and preserve 7/7; contact episodes 2/5 → 0/5 | Source gate passes; previously inspected scenes. |
| INIT14, prior skills | Native Q1 versus frozen D5+C | Both load 3/4; no lost success or new fixture/table contact | Same failed start and joint excursion; three fixture-free feeding cases unsupported. Prior gate fails/incomplete. |
| INIT14, archived handover | Native Q1 versus frozen D5+C | Both load 2/2 and preserve 2/2; contact episodes 1/2 → 0/2 | Passes this boundary test; preceding feeding task is an archived supplied prefix. |

**D5:** the first eligible predicted closure triggers a 5 mm reduction in commanded wrist Z for 60 actions. The model still replans normally from the actual resulting images and command history. It does not use object truth or a successful-grasp detector to activate.

**C:** after that acquisition adjustment, a measured-open primitive holds wrist XY/orientation and raises the hand using known robot/fixture geometry. It hands control back after measured clearance persists for three samples. It is supplied motion logic, not learned collision avoidance.

These experiments suggest that precise acquisition height is a local execution bottleneck. They do not establish it as the only reason loading generalises poorly. A rule triggered by proposed closure cannot solve a policy that never approaches or closes.

Sources: [INIT12/A1](https://github.com/NatMota/typhon-robotics/blob/f7eb4fa40981d5c1cdcac61fb385bdd00dfbbf3b/experiment-history/2026-10-16-init12a1-amendment/INIT12A1_report.md), [INIT13](https://github.com/NatMota/typhon-robotics/blob/4851ef323e5484c3498cca335a5afc29f3c0eb5f/experiment-history/2026-10-16-init13-measured-clearance/INIT13_report.md).

## Continuous feeding followed by loading

HIER02-LIVE-R1 executes native GOAL-varied feeding and then Q1 loading without resetting the robot, brick, controller or simulator between tasks. The stage switch is supplied at a fixed horizon. Two inspected personal scenario families and two seeds give four matched cases per condition.

| Condition | Feeding pickup / strict | Loading pickup / strict | Complete workflow | Forbidden contacts / joint violations |
|---|---:|---:|---:|---:|
| Native GOAL → native Q1 | 4/4 / 4/4 | 4/4 / 4/4 | 4/4 | 0 / 0 |
| Native GOAL → Q1 with frozen D5+C | 4/4 / 4/4 | 4/4 / 4/4 | 4/4 | 0 / 0 |

All sixteen scores and eight full-state handovers were independently verified. The four paired feeding executions and loading inputs before assistance were exact. D5 and clearance did activate in assisted loading, but added no successful workflows on this panel. This supports local policy composition; it does not demonstrate unseen-session generalisation, learned task switching or visual numeral recognition.

[HIER02-LIVE-R1 report, per-stage subgoals and comparison images](https://github.com/NatMota/typhon-robotics/blob/ebf36e430bc22dc28f537bdaa0b8b0fe9a9846bb/experiment-history/2026-10-16-hier02-live-r1/HIER02_R1_report.md). Full archive: 2,184 files / 501,543,926 bytes, verified on Mint with zero SHA errors.

## What each score means

- **Supported pickup:** bilateral contact and at least 10 mm lift with tilt ≤10° for three consecutive control samples. Closing the gripper alone is insufficient.
- **Strict completion:** the original combined position, directed yaw, face-up, footprint, height, release and stability criteria; loading also requires seating.
- **Preservation:** the correct terminal arrangement plus no gripper–object contact and at most 1 mm movement throughout the episode.
- **Contact qualification:** a separate condition from terminal task success. A successful placement can still fail this check.

Contact evidence is sampled at 20 Hz; it does not establish hardware safety. Final yaw error is not visual numeral-recognition accuracy. Earlier V1–V4/M3–M6 suites have different models and starts and must not be pooled into the tables above.

## Integrity checks and remaining uncertainty

A wrong cached model implementation caused a reload mismatch and was corrected in a separately recorded runtime investigation. Subsequent runs use the original qualified cache and an exact fresh-process action probe.

One INIT12 feeding pair first differed in secondary-camera RGB before prediction/history diverged. Its original exact-parity failure remains failed. INIT12-A1 explicitly completed the remaining ten trials under an amended RGB-first handling rule; the report does not call all original pairs exact.

INIT14 completed 40 eligible episodes without tuning or retries. All 40 scores and issued-history audits pass; 19/20 pairs have exact preintervention execution. Prior loading 0005 first diverges in secondary RGB at query 2, so that comparison remains descriptive under the existing rule. In failed prior loading 0001, neither condition issues a close command or activates assistance; both have a −0.0035295 rad joint margin. The controller therefore cannot be described as universally qualified. Local continuous feeding→loading composition now passes the HIER02-LIVE-R1 panel above; the untreated prior-context initiation failure remains open. There is no genuinely independent recording-session panel in the supplied bank. Physical calibration, direct human-video action alignment and a generally qualified recovery/observation layer remain separate unfinished work.

This is a private working synthesis. No repository publication or application submission is performed by this update.

Latest evidence: [INIT14 report and per-case evidence](https://github.com/NatMota/typhon-robotics/blob/d6b43bc529eb8fbf29e01a1f93d947b7d7e0028b/experiment-history/2026-10-16-init14-frozen-qualification/INIT14_report.md).
