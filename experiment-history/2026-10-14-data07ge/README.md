# DATA07-GE — The reset-successful teacher fails from B's approach state

**The geometric expert transported and released the brick inside the loading fixture, but left it on its side. Strict success: 0/1.** No corrective trajectory was approved for training, and no training ran.

| Execution on family 0001 | Final position error | Yaw error | Upright / seated | Released | Strict result |
|---|---:|---:|---|---|---|
| Archived native B | 427.73 mm | 55.65° | No / No | No | Fail |
| Archived one-chunk C-pose repair, then B | 428.09 mm | 55.63° | No / No | No | Fail |
| Current-state geometric expert | 1.22 mm | 0.016° | No / No | Yes | Fail |

B and C are the LOAD04 counterbalanced and correlated-control checkpoints, not the older V1–V4 versions. The first two references are unchanged DATA07-R outcomes; only the expert continuation is new.

[Expert video](videos/geometric-current-state-demo_0001.mp4) · [Matched-time contact sheet](native-vs-expert-contact-sheet.jpg) · [Detailed score](summary.json) · [Expert phases](expert-result.json)

## What this establishes

The same geometric loading program previously succeeded from this training scene's reset state. Here it was started from B's qualified pre-contact state at tick 460, without resetting physics, object pose, joints, controller or solver state.

It contacted the brick at tick 527 and tipped it beyond 15° at tick 530. The first closing command was only at tick 550: tipping occurred during the open-gripper descent. At the first supported 10 mm lift the brick's tilt was 91.43°. The expert then transported the sideways brick and released it inside the fixture. Final tilt was 90°.

The position, yaw, footprint, release and stability criteria passed. Upright marked-face orientation, placement height and seating failed. Thus a tiny yaw error does not imply the required full orientation.

The expert's internal verified_lift flag was true because it checks lift height, not upright acquisition. Its lift phase also failed its motion-convergence criterion. The independently preserved full-task scorer correctly rejected the outcome. The nearest declared joint-limit margin reached -0.0030 rad during the suffix; that is logged, not established as the cause.

**Reset-state success does not qualify this controller as a recovery teacher from arbitrary learner states.** This failed suffix must not become positive imitation data. It does not prove physical impossibility or establish that the visual model cannot recognise the numeral.

## Was the C-pose patch actually different?

Read-only comparison of the archived DATA07-R runs found different raw targets, requested commands, applied commands and finger-geometry trajectories from tick 461. Finger positions differed by up to 4.90 mm before the first contact at tick 473. There was no additional workspace clipping in that window.

The failed patch therefore was not simply erased by the workspace bounds. This does not identify which approach geometry or timing caused tipping. See saved-command-comparison.json for every compared tick through 478.

## Qualified, bounded experiment

- Exact LOAD04 loading-training family 0001, original simulation seed 131902, visible fixture and original goal/physics.
- Cached DATA07-R B prefix through tick 460/query 14, with exact RGB and proprio/history equality. No new VLA inference.
- Zero-tolerance semantic physics/controller/RNG comparison passed. Only the previously qualified XML statistic/meansize exclusion remained; no new comparator exclusions.
- Unchanged supervisor.feeding_loading.runner.scripted_attempt, calculating its commands from the current state. The original OSC controller, bounds and stepping convention were retained.
- One expert attempt, no alternate grasp, automatic retry, perturbation or reseeding. The expert ended at tick 769; the original 24 passive stability ticks ended at 793, within the original tick-1500 cap. The expert's completion endpoint is earlier than the archived fixed-horizon B endpoint.
- Explicitly privileged object-state control. This is a controller recovery diagnostic, not an RGB-only actor result. No truth channel was added to B.
- Full current-state observations, controller goals, commands, contacts, finger geometry, object orientation and gripper transitions are retained.
- Empty execution/artifact/evaluation error lists. This is a measured task failure, not an infrastructure failure.
- No development or holdout scenes used. AlphaNIST remains Parked.

The benchmark change is a three-line opt-in termination hook after the externally supplied continuation. It preserves the original scoring and passive stability path. The old benchmark and frozen DATA07-R worktree remain unchanged.

## Preserved launch qualification amendment

The first prefix qualification stopped before any expert command. All 86 differences were model-asset path strings introduced by launching through the new worktree's virtual-environment alias; each resolved to the exact same existing asset path. Physics, controller, RNG and observation values matched.

We preserved that preflight, then launched the new experiment code using the exact original DATA07-R environment path. The unchanged comparator passed with zero differences. No meshes, physics, gains, prefix actions, cases or deadline were changed. This is one corrected launch qualification and one expert attempt, not repeated recovery trials. See launcher-amendment.json and the raw qualification-path-mismatch-01 directory.

## Disposition

Stopped after the single attempt, as the planner required. There is no successful correction candidate from this test, and the local donor/teacher search is ended for now.

Before further corrective fitting, review why the approach tips the object and require upright grasp/lift verification from learner-reached states. Additional training cannot be justified by calling this suffix successful. A new controller investigation would need its own explicit scope; no earlier handoffs or longer-donor sweeps were added.

## Reproduction and storage

Branch: research/data07ge-current-state-teacher.
Parent: 8bab3bb2e08c3263d56e7680f9a0c1bf98de8d8c.
New executable: scripts/data07ge_run.py.
Analysis: scripts/data07ge_analyse.py.
Frozen source, function hashes, bounded hook and exact configuration: protocol.json and benchmark-hook-diff.txt.

On the existing qualified Powerglide environment, a fresh run must use the original interpreter alias to preserve asset provenance. The reproduction wrapper creates a fresh output namespace, starts a new bounded clock and refuses overwrite:

~~~bash
/cs/student/msc_rai/2026/ndossant/Projects/typhon-experiments/data07r-training-repairs/.venv-sim/bin/python scripts/data07ge_reproduce.py --output /var/tmp/typhon-data07ge-reproduction-UNIQUE
~~~

This reproduction command was prepared, not executed as an additional trial.

Raw: /var/tmp/typhon-data07ge-20261014.
Mint mirror: /media/nm/Fast Drive/Typhon/research-20261014/data07ge-current-state/raw.
The raw manifest and mirror-verification.json record SHA256 integrity. Code, results and video are on the private GitHub branch; bulk state arrays remain on Mint. Fresh-machine restoration is untested.
