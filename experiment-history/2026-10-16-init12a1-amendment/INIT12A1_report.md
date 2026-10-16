# INIT12 / INIT12-A1 — command-triggered acquisition adjustment

A fixed 5 mm reduction in commanded gripper height raised strict loading success from **4/8 to 8/8** in this inspected source-family panel. It preserved all eight already-completed feeding arrangements, but introduced forbidden finger–fixture contacts in **4/8 loading episodes**. **The candidate fails its advancement gate and remains quarantined.**

This is an exploratory comparison on four previously inspected source families and two fixed policy seeds. It is not evidence of unseen-workflow generalisation, an end-to-end human-video policy, or a training result.

## What ran

- Frozen Q1 checkpoint: `26d573eb90f791d9af86bbff30e138f51c5aa658ac7abefe76787ba352363a7a`, step3000.
- Native Q1 versus D5, with the same task starts, controller, observations, noise, H30 predictions, 1,436 policy actions and 24 settling ticks.
- Parents: demo_0001, demo_0002, demo_0003, demo_0005. Seeds: 912171 and 912172.
- Loading starts follow the qualified supplied feeding prefix. Feeding trials ask the model to preserve the already-completed arrangement. These are the original INIT07 task-request starts.
- D5 triggers once at the first fresh 30-action chunk containing an active-arm closing probability >0.5 while the last executed gripper command is open. It subtracts 0.005 m from world-Z for that chunk and the next chunk: exactly 60 issued actions.
- Both modified chunks use fresh native inference from the branch's actual camera images and issued-command history. Only active-arm Z is modified; first-nine float32 pose history and eleven-zero masking are preserved.
- No task name, scene ID, seed, simulator object truth, contact, measured lift or completion detector is used by the trigger. It detects a closing proposal, not a successful grasp.
- No training, TTC, forced release, extra dwell, taper, outcome retries or added seeds.

Original execution commit: `772652e11baff435745fe5c335e2e0509f80a392`.
A1 execution commit: `9ef5eed3e5e3f032ba543a4dba12af624cb227de`.
Independent analysis commit: `fb8e9e24366380af1b1bd03ecda5b914491bdf48`.

## Outcomes

| Measure | Native Q1 | D5 |
|---|---:|---:|
| Strict loading success | 4/8 | 8/8 |
| Supported lifts, >=10 mm for >=3 steps | 4/8 | 8/8 |
| Correct final loading footprint, release and stability | 4/8 complete combinations | 8/8 |
| Loading episodes with forbidden contacts | 0/8 | 4/8 |
| Already-completed feeding preserved | 8/8 | 8/8 |
| Feeding episodes with object contact or displacement >1 mm | 0/8 | 0/8 |
| D5 triggers during feeding | — | 0/8 |

The descriptive loading change is +50 percentage points: four gains and no lost native successes. The two selected INIT11-R rescues both succeeded again. The other six loading trials are also inspected conditions, not an untouched confirmatory test.

Strict loading retains the original conjunction of position, yaw, marked face up, footprint, placement height, release, real temporal stability and seating. The separate forbidden-contact criterion can therefore fail while terminal task success passes. A low yaw error alone does not imply success: the four native failures had no verified supported lift, remained outside the fixture and ended tipped by 90 degrees.

### Every loading pair

Position and yaw errors below are final goal errors, not improvements from the starting angle. Complete per-episode fields and all feeding results are in [episodes.csv](evidence/episodes.csv).

| Parent / seed | Native success | D5 success | Native error mm / deg | D5 error mm / deg | D5 forbidden samples |
|---|---:|---:|---:|---:|---:|
| 0001 / 912171 | True | True | 4.960 / 1.049 | 5.643 / 0.819 | 1 |
| 0001 / 912172 | True | True | 3.999 / 0.945 | 5.762 / 1.192 | 1 |
| 0002 / 912171 | False | True | 264.406 / 1.864 | 6.936 / 0.557 | 0 |
| 0002 / 912172 | True | True | 6.516 / 0.651 | 4.568 / 0.839 | 0 |
| 0003 / 912171 | False | True | 263.377 / 2.505 | 4.451 / 0.574 | 0 |
| 0003 / 912172 | False | True | 263.072 / 2.521 | 5.849 / 1.007 | 1 |
| 0005 / 912171 | False | True | 258.529 / 0.424 | 4.724 / 0.564 | 0 |
| 0005 / 912172 | True | True | 7.913 / 0.986 | 5.628 / 1.085 | 2 |

## Contact finding and motion interpretation

All four D5 cases first contact `fixture_wall_1` with `gripper0_finger2_collision` at policy action149 (zero-based). The offset ended at action120; action149 is the final command of the first fully unmodified H30 chunk. One case also records contact at action150. At contact the gripper is open, the object is already inside the fixture, and the measured aperture proxy is about74 mm.

The repeated phase suggests a placement/release clearance problem worth investigating. It does not prove whether the decisive cause is the last raw pose, achieved pose, opening motion or earlier trajectory. These are consequences of the entire D5 branch even though the Z modification has expired.

| Parent / seed | Sampled contact actions | Largest sampled force norm (N) |
|---|---|---:|
| 0001 / 912171 | 149 | 4.721 |
| 0001 / 912172 | 149 | 4.420 |
| 0003 / 912172 | 149 | 11.419 |
| 0005 / 912172 | 149,150 | 6.121 |

These are contact-frame wrenches sampled at20Hz after natural controller steps. They are not contact duration, substep peak force, impulse, or an object-side net wrench. Force values do not waive the original forbidden-contact classification. Actual EEF pose, finger joints, control, object pose and geometry pairs are retained in [forbidden-contact-events.json](evidence/forbidden-contact-events.json).

Existing workspace clipping remains active and is logged: native loading6–17 clipped commands per episode; D5 loading11–15. No joint-limit excursion was added. No undeclared truth-based protection was introduced.

## Technical stop and explicit amendment

The original protocol completed22 physical episodes, of which21 passed all original qualification gates. It stopped on `init-demo_0003-feeding-912171` in D5, where the wrapper never triggered.

Through query17 and the first540 executed actions, the pair matched. At query18, 13 channel values in the secondary RGB image differed by one intensity level. Primary RGB, token/mask/proprio inputs and paired noise still matched. Five prepared-image tensor values differed; maximum difference0.0175071. The first raw prediction then differed by up to1.953125mm in XYZ and0.042315 degrees in decoded rotation, with no gripper threshold change. History and physical trajectories diverged afterwards.

The upstream camera-input difference is established; its origin is unresolved. A visually small background discrepancy is not assumed harmless, and renderer nondeterminism is not claimed as a proven cause. Both executions preserve the brick, but the original exact-parity gate stays **FAILED_UNCHANGED**.

INIT12-A1 was frozen **after these22 outcomes and the failure were known**. It ran only the ten previously unexecuted episodes, in their original order and with the same seeds. It changed only the handling of later RGB-first cross-run divergence: exact within-run arithmetic and history remained mandatory; a nonvisual-first divergence or different prediction from identical effective input/noise would still stop execution. It did not retroactively pass the original failure.

All16 pairs pass exact initial state/input and48-noise-packet checks. Fifteen pairs retain exact pretrigger equivalence, including all eight loading pairs. The one feeding pair lacks that equivalence and remains in the outcome table. All ten A1 executions were exact under their applicable pairing checks. All32 strict scores, command arithmetic and history checks were independently reproduced from saved artifacts.

Do not describe all16 pairs as exact counterfactual trajectories. A1 is an explicitly amended exploratory completion.

## Decision

The main advancement criterion fails because four D5 loading trials add forbidden contacts. **The14 conditional prior-skill trials were not run**, and no substitute panel was used. Improved acquisition is a local result; preservation of an already-completed scene does not show that the wrapper understands completion.

The next question is whether the common post-placement contact can be explained and avoided without giving the policy hidden geometry or sacrificing acquisition. No revised wrapper or new training is evaluated in this report.

## Visual checks

Agent inspection, not an additional human review:

- [Selected loading rescue: demo0003 /912172](visual-review/init-demo_0003-loading-912172.png): native leaves the brick outside the fixture; D5 acquires, transports and seats it.
- [Native success and D5 success: demo0001 /912172](visual-review/init-demo_0001-loading-912172.png): both load; the coarse sheet alone cannot resolve the brief forbidden contact.
- [Selected loading rescue: demo0005 /912171](visual-review/init-demo_0005-loading-912171.png): native leaves the brick at the source; D5 carries and releases it inside the fixture.
- [Completed feeding preserved: demo0003 /912172](visual-review/init-demo_0003-feeding-912172.png): neither condition touches or moves the brick.
- [First differing RGB diagnostic](visual-review/raster-divergence-contact-sheet.png): changed values are in the tabletop/background area. Their downstream effects are quantified, not dismissed.

Frame provenance links every displayed frame to its original episode video and control tick. All original primary and secondary episode videos remain in the verified Mint archive.

## Provenance, archive and reproduction

No frozen M0/M1/M1.5 or earlier experiment output was modified. INIT12's2,908 original files were SHA-checked before and after A1. Full raw artifacts live on Mint's fast drive; the private Git branch contains source, protocol, this report, bounded evidence and review images.


| Archive | Files | Bytes | SHA errors |
|---|---:|---:|---:|
| INIT12 | 2908 | 695422356 | 0 |
| A1 | 1336 | 314459291 | 0 |

Full paths:

- Original: `/media/nm/Fast Drive/Typhon/research-20261015/init12-command-acquisition`
- A1: `/media/nm/Fast Drive/Typhon/research-20261015/init12a1-remaining`

Original block began02:23:58 UTC; physical work ended02:50:18. The original closeout deadline03:53:58 UTC was not reset. No14-case prior panel or new physics was used during analysis.

To reproduce the saved-artifact analysis on Powerglide with its existing paths and environments (choose an unused output directory):

```sh
cd /var/tmp/typhon-worktrees-20261016/init12a1
.venv-sim/bin/python -c "import sys; sys.path[:0]=['scripts']; from pathlib import Path; import init12a1_analyse as a; a.OUT=Path('/var/tmp/typhon-init12-independent-recheck'); assert not a.OUT.exists(); a.main()"
```

Original execution commands were `.venv-sim/bin/python scripts/init12.py` and, after freezing the numbered amendment, `.venv-sim/bin/python scripts/init12a1.py`. Do not rerun either into the frozen output directories. A fresh execution requires its own output namespace, case freeze, clock and corrected cache links as recorded in the execution manifests.

Runtime: RTX4090, driver560.35.03; simulation Python3.11.12/NumPy1.26.4/MuJoCo2.3.7/robosuite1.4.1; model torch2.5.1+cu121/transformers4.51.3/bfloat16. The restored checkpoint code resolves under `Projects/humanoid-m6-joint/.cache-xvla-pt`; using the M2 cache is an invalid reproduction.
