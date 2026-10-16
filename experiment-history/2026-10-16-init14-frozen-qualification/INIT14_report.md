# INIT14 — frozen acquisition and clearance qualification

**The frozen controller assistance passes the remaining-source and archived-handover panels. Full qualification fails: prior loading includes the same uncorrected native failure and joint excursion, and three fixture-free prior feeding cases are unsupported.** No new training or local parameter tuning occurred.

This experiment qualifies the limits of the supplied D5 plus clearance controller around the unchanged Q1 policy. It does not turn an assisted execution result into a better native-model score.

## Fixed comparison and outcomes

| Panel | Executed contrast | Completion / preservation | Contact episodes | Decision |
|---|---|---|---|---|
| Remaining source loading | D5 versus D5+C, five pairs | Both 5/5 strict and 5/5 supported pickups | D5 2/5; C 0/5 | Pass |
| Remaining source completed feeding | D5 versus D5+C, seven pairs | Both preserve 7/7; neither rule activates | Both 0/7 | Pass |
| Prior loading | Native Q1 versus C, four pairs | Both 3/4 strict and 3/4 supported pickups | Both 0/4 | No lost native success; joint-qualified gate fails |
| Prior feeding | Three prescribed pairs | Unsupported; zero of six episodes executed | Not measured | Applicability incomplete |
| Archived handover loading | Native Q1 versus C, two pairs | Both 2/2 strict and 2/2 supported pickups | Native 1/2; C 0/2 | Pass |
| Archived handover completed feeding | Native Q1 versus C, two pairs | Both preserve 2/2; neither rule activates | Both 0/2 | Pass |

Forty eligible episodes were executed from forty-six planned. All outcomes were retained; there were no outcome retries, substituted cases or technical episode failures. The three panels answer different questions and have no pooled treatment effect. Source cases are the exact complement of INIT13's four inspected cases, not a new independent recording session.

The handover is the archived demo0012 feeding-to-loading boundary reached through 1,460 recorded prefix commands. It is a real previously executed robot boundary, followed here by new model decisions. It is not a new uninterrupted two-task policy execution. The prefix and these quarantined outcomes do not become training data.

## What assistance supplies

Q1 weights, O2/domain29/H30 inputs, ten denoising steps, native gripper commands, controllers and geometry remain fixed.

D5 lowers commanded world Z by 5 mm for sixty executed actions after the first eligible predicted closure. C then uses measured robot pose/aperture and known fixture geometry for the unchanged measured-open withdrawal: fully open collision envelope targeted 20 mm above the fixture, handback after three consecutive measured clearance samples of at least 10 mm. History contains the actual issued first-nine float32 values plus eleven zeros. Normal model reobservation continues.

Object truth, task success and contact forces are used only for evaluation. Robot collision geometry and fixture height are supplied control knowledge. This is a controller intervention, not learned avoidance or an image-only result.

All five source C loading cases use five clearance actions from action 147. The three successful prior C loadings use five from 207. Handover C loading uses five from 147 for seed 912171 and four for 912172. No timeout, premature reclose, new later forbidden contact or descriptive drop event was recorded. Path differences are recorded per case in the JSON/CSV; these are complete trajectory differences, not just the vertical target rise.

## The prior failure and the unsupported domain

For prior loading 0001, native and assisted outputs both remain open for all 1,436 policy actions. The largest close probability is 0.000843812. Neither D5 nor C activates, there is no gripper–block contact, and the final errors remain 433.794 mm and 54.481 degrees.

Both executions have 320 recorded negative installed-joint margins; the worst is−0.0035294979rad (about−0.202degrees), first recorded at control step 73. The two summaries match exactly. This is an existing native failure retained by the wrapper, not a new wrapper-induced regression. It still fails the declared absolute joint-qualified gate. Task-end success and controller qualification remain distinct.

The exact INIT07 prior-feeding configurations G0/G1/G2 have no physical fixture. The unchanged clearance implementation has no supported fixture-absent contract. We did not add a fabricated wall height, replacement scene or conditional skip. Cases retain-personal_demo_0002-goal0, retain-personal_demo_0002-goal1 and retain-personal_demo_0001-goal2 remain unsupported/unexecuted.

## Integrity and comparison limits

A fresh exact Q1 reload probe passed. All 40 strict scores were recomputed offline using the original scorer; all 40 raw-to-issued-to-history audits pass. All 20 initial physical/controller states, initial prepared inputs and paired noise packets match exactly.

Nineteen of 20 pairs retain exact preintervention equivalence. Prior loading 0005 first differs in secondary-camera RGB and prepared inputs at query 2; raw predictions also differ there, while history/measured state first differ at query 3. This is retained under the predeclared INIT12-A1 RGB-first treatment, with origin unresolved. Both endpoints succeed, but their difference is descriptive and cannot be assigned exclusively to D5+C. No pixel tolerance, reseeding or historical-output substitution was added. All source and handover pairs are exact up to their intervention.

The original INIT12 failed contact qualification and RGB equivalence failure remain unchanged. This follow-up does not retroactively pass them.

Strict success uses the original position, directed yaw, correct face, footprint, placement height, release, stability and loading seating criteria. Supported pickup requires bilateral contact, at least 10 mm lift and tilt ≤10 degrees for three samples. Preservation additionally requires no object contact and maximum displacement ≤1 mm. All these labels are reported separately.

Contacts and joint evidence are sampled at 20 Hz. No contact at recorded samples establishes neither substep contact absence nor physical-robot safety. No physical robot was used. Final yaw accuracy does not establish visual numeral recognition.

## Per-case outcomes

| Panel / case | Control | Strict control / C | Supported pickup control / C | Contact samples control / C | C final position mm / yaw deg |
|---|---|---:|---:|---:|---:|
| source / init-demo_0001-loading-912172 | D5 | yes / yes | yes / yes | 1 / 0 | 5.762 / 1.192 |
| source / init-demo_0001-feeding-912172 | D5 | yes / yes | no / no | 0 / 0 | 12.068 / 0.338 |
| source / init-demo_0002-loading-912172 | D5 | yes / yes | yes / yes | 0 / 0 | 4.568 / 0.839 |
| source / init-demo_0002-feeding-912171 | D5 | yes / yes | no / no | 0 / 0 | 1.812 / 0.466 |
| source / init-demo_0002-feeding-912172 | D5 | yes / yes | no / no | 0 / 0 | 1.812 / 0.466 |
| source / init-demo_0003-loading-912171 | D5 | yes / yes | yes / yes | 0 / 0 | 4.451 / 0.574 |
| source / init-demo_0003-feeding-912171 | D5 | yes / yes | no / no | 0 / 0 | 2.607 / 0.432 |
| source / init-demo_0003-feeding-912172 | D5 | yes / yes | no / no | 0 / 0 | 2.607 / 0.432 |
| source / init-demo_0005-loading-912171 | D5 | yes / yes | yes / yes | 0 / 0 | 4.724 / 0.564 |
| source / init-demo_0005-loading-912172 | D5 | yes / yes | yes / yes | 2 / 0 | 5.628 / 1.085 |
| source / init-demo_0005-feeding-912171 | D5 | yes / yes | no / no | 0 / 0 | 2.279 / 0.499 |
| source / init-demo_0005-feeding-912172 | D5 | yes / yes | no / no | 0 / 0 | 2.279 / 0.499 |
| prior / retain-load01-training-demo_0001 | N | no / no | no / no | 0 / 0 | 433.794 / 54.481 |
| prior / retain-load01-training-demo_0002 | N | yes / yes | yes / yes | 0 / 0 | 1.288 / 0.102 |
| prior / retain-load01-training-demo_0003 | N | yes / yes | yes / yes | 0 / 0 | 3.218 / 0.557 |
| prior / retain-load01-training-demo_0005 | N | yes / yes | yes / yes | 0 / 0 | 4.996 / 1.100 |
| handover / dev-demo_0012-loading-912171 | N | yes / yes | yes / yes | 0 / 0 | 5.207 / 0.170 |
| handover / dev-demo_0012-loading-912172 | N | yes / yes | yes / yes | 1 / 0 | 6.723 / 0.569 |
| handover / dev-demo_0012-feeding-912171 | N | yes / yes | no / no | 0 / 0 | 4.553 / 1.428 |
| handover / dev-demo_0012-feeding-912172 | N | yes / yes | no / no | 0 / 0 | 4.553 / 1.428 |


The seven source and two handover feeding cases in this table are already-completed preservation requests. Their absent pickup is intended behavior. The full component scores, contacts/wrenches, joint margins, paths, triggers and paired-input audits are in [evidence/episodes.json](evidence/episodes.json) and [evidence/episodes.csv](evidence/episodes.csv).

## Saved visual evidence

Six cases were selected before outcomes; frames come from their saved 5 fps camera videos, with original frame/tick and SHA references in [frame-provenance.json](visual-review/frame-provenance.json).

- [Remaining source0001/912172](visual-review/init-demo_0001-loading-912172.png)
- [Remaining source0005/912172](visual-review/init-demo_0005-loading-912172.png)
- [Uncorrected prior loading 0001](visual-review/retain-load01-training-demo_0001.png)
- [Prior loading 0005](visual-review/retain-load01-training-demo_0005.png)
- [Archived handover loading](visual-review/dev-demo_0012-loading-912171.png)
- [Archived handover preservation](visual-review/dev-demo_0012-feeding-912171.png)

Agent inspection of all six fixed contact sheets agrees with the recorded endpoints: four pictured loading examples finish in the fixture under both conditions, the failed prior case leaves the brick at its original left-side start, and completed feeding stays untouched. The prior 0005 middle frames show a lifted block. These are agent observations, not additional human labels. Sheets select the nearest saved frame to the requested time; the source0001 middle frame is two control ticks earlier, as recorded in frame-provenance.json. Five-fps sheets do not establish absence of brief contacts or isolate the clearance motion.

## Decision

Keep the controller frozen as a local assisted candidate for the supported fixture-present domain. The source comparison supports incremental contact reduction from clearance. The handover comparison supports the combined system at one inspected boundary; native Q1 already completed those loading tasks. The prior comparison preserves measured successes while exposing an uncorrected initiation failure and an unsupported physical context.

Do not claim universal applicability, broad reliability, completion recognition, numeral understanding or new human-session generalisation. The next experiment should address initiation/composition or another explicit unfinished OST question without further local clearance tuning.

## Reproduction and archive

Frozen execution commit: 800ab86fc2f2f50ee0ea67128474f08b00cdec79.
Offline analyser: 09191987eead74543ad39b79d8bbb5da5b534136.
Saved-video script: 8818c4d84c4128f53214d1efff7c0308e4d6dd0e.
Q1 SHA-256: 26d573eb90f791d9af86bbff30e138f51c5aa658ac7abefe76787ba352363a7a.

Original commands:

~~~sh
cd /var/tmp/typhon-worktrees-20261016/init14
.venv-sim/bin/python scripts/init14.py
.venv-sim/bin/python scripts/init14_analyse.py
.venv-sim/bin/python scripts/init14_media.py
python3 /var/tmp/typhon-init14-20261016/analysis/limitation-audit.py
~~~

These exact physical output directories and clock are frozen and deliberately reject reruns. To verify saved evidence, import the analysis module and point its input root at the archive and output at a fresh directory; no simulator steps or model calls are needed. A new physical execution requires a separate namespace and explicit frozen clock.

Verified 5,366 files / 1,240,722,749 bytes with zero SHA-256 errors at 2026-10-16 04:06:34 UTC. Full archive: /media/nm/Fast Drive/Typhon/research-20261015/init14-frozen-qualification. Source-manifest SHA-256: 6be0566f28b8b7ad786a4c9a8c38ad078fabdc1e9d22436fd34bff3aa155c7bb.

Activated 2026-10-16 03:38:56 UTC; physical completion 2026-10-16 03:59:43 UTC; original archival deadline 2026-10-16 05:08:56 UTC. No clock reset.

The archive preserves all predictions, issued targets, noise, actual prepared inputs, controller/force traces, boundary states, source manifests and videos. Verified redundant INIT12/A1 scratch episode copies were removed before this run; their complete Mint archives remain unchanged. Bulk data and model weights remain on Powerglide/Mint; six temporary Mac review PNGs (4.06 MB) were deleted after inspection.
