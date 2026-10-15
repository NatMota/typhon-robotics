# LIVE01-R: fixed-state camera repeatability

**Result: the earlier pixel difference was not reproduced within one fixed graphics context.** The corrected, preregistered 24-bundle comparison produced identical native camera bytes and processor tensors in ordinary mode (N) and provenance mode (P). This does not reverse LIVE01-P's failed live paired test.

## Scope and counts

| Item | Observed result |
|---|---|
| Scene | LOAD04-B training configuration, personal demo_0002, loading |
| Boundary | Replay-reconstructed tick190 / query5 |
| Prefix | Original reset plus150 archived requested actions; all150 available physical/controller/RNG snapshots exact |
| Corrected capture matrix | NNPP, PPNN, NNPP, PPNN, NNPP, PPNN |
| Modes | 12 N and12 P bundles;48 native512×512 RGB arrays |
| Differences within corrected matrix | 0 changed pixels; maximum channel difference0 |
| Processed inputs | Exact equality for all24 bundles, including mask, text, command-history proprioception and domain |
| Available active render fields | Exact per-camera equality across24 bundles |
| Physical/controller/RNG read-only checks | 24/24 before/after pairs exact to qualified boundary |
| Physics after boundary / model calls | 0 /0 |
| Actual P capture identities | 24 unique IDs;24 nonpadding processor slots resolved |
| Common diagnostic acquisition identities | 48 unique IDs despite the same simulation tick |
| Temporal monitor | One successful update at tick190, then explicitly cached output |
| New training / holdout use | None /none |

Secondary-camera pixel(x24,y417) was [195,186,177] in every corrected capture. Both native cameras and all prepared arrays were compared byte-for-byte, not merely this point. No tolerance was widened.

The primary-image orange crop found no plausible component at this boundary. The actual O2 path returned two native views, with image mask [true,true,false]; the third slot was padding. This test does not establish numeral recognition. The contact sheet shows the primary view obscured by the wrist and the brick visible in the secondary view.

## Failed first composition and frozen amendment

Attempt01 reconstructed the state exactly in187.49seconds and retained one ordinary bundle. The next temporal-observer update used the same tick190 and divided by zero in its aperture-rate calculation. This was a test-composition error, not a pixel-repeatability result or evidence that the normal advancing-clock observer fails.

The planner approved exactly one corrected attempt under the original40-minute clock. The observer implementation, denominator and clock were unchanged. Attempt02 updated it once and cached the estimate, preserving the estimate's original bundle0 input hashes, acquisition references and timestamp. Subsequent captures do not claim to be the source of that cached estimate. Both modes used identical treatment.

Attempt02 qualified the original150-command replay at446.05seconds after the original start and finished capture at448.29seconds. The initial attempt was not used to fill any of its24 slots. Original failed-attempt files and hashes are preserved in attempt-01, alongside original root-level copies. No third attempt was made.

A read-only unit check reproduces the duplicate-timestamp error and confirms an advancing timestamp works. No OBS06 repair was made.

## Exact tested path

Each corrected bundle used the original env.observe() and native primary/secondary rendering and copy operations. P used real RGBLineage capture/copy/publication and PreprocessLineage around actual O2 and AutoProcessor calls; N used the ordinary path. Both used the actual telemetry publisher with a common adapter that deferred JPEG encoding and filesystem publication until after capture. The image processor ran without model weights.

One graphics context was retained for all24 corrected bundles. No added step, forward, reset, scene restart, overlay, synchronization command, numerical setting change, renderer change or package installation occurred after the boundary. Arrays were copied losslessly; image encoding and the contact sheet occurred afterwards. Instrumentation recorded active scene geometry/lights/cameras, available framebuffer/viewport/options and relevant model inputs, not unused buffer capacity or addresses.

The reconstruction gate used available archived physical/controller/RNG fields. It is not a reconstruction of all historical GPU details. Earlier prefix camera queries0–4 also matched; the new query5 pixel value was never an eligibility gate.

## Secondary observation, kept separate

The single ordinary capture from failed attempt01 differs from corrected attempt02's first ordinary capture by exactly one secondary pixel at(x24,y417), by one level in each channel; primary bytes and recorded active render fields agree. These were different processes and test compositions. This observation is outside the24-bundle matrix. It neither establishes a sidecar effect nor localizes the graphics/context cause.

## Decision

Close fixed-state repeatability for this overnight session. Preserve three distinct findings:

1. LIVE01-P component capture lineage and offline tensor parity passed their checks.
2. Its original live paired test failed exact parity.
3. LIVE01-R did not reproduce variation within one qualified fixed context with cached temporal-monitor output.

The monitor gains no controller authority from this result. No new controller, policy-performance, task-success or generalization claim is made.

## Evidence and reproduction

Raw evidence on Powerglide: /var/tmp/typhon-live01-r-20261015

Corrected evidence under attempt-02: capture-matrix.csv, arrays/, captures.pkl.gz, state-pairs.pkl.gz, graphics-settings.json, lineage-events.json, lineage-*.json, publications.json, validation-summary.json, evidence-validation.json, preservation-validation.json and contact-sheet.png.

The actual command, interpreter alias, working directory, clock, source hashes and planner amendment are retained. Do not replay capture into these frozen output directories. Future replication requires a fresh output root and clock with the same interpreter alias, source and existing dependencies.

Historical capture command (not rerun after closeout):

~~~bash
/cs/student/msc_rai/2026/ndossant/Projects/typhon-experiments/data07r-training-repairs/.venv-sim/bin/python scripts/live01_r_capture_v2.py
~~~

Read-only reproduction from retained arrays:

~~~bash
/cs/student/msc_rai/2026/ndossant/Projects/typhon-experiments/data07r-training-repairs/.venv-sim/bin/python scripts/live01_r_analyse.py /var/tmp/typhon-live01-r-20261015/attempt-02
/cs/student/msc_rai/2026/ndossant/Projects/typhon-experiments/data07r-training-repairs/.venv-sim/bin/python scripts/live01_r_validate.py /var/tmp/typhon-live01-r-20261015
~~~

The private branch and Mint FastDrive archive preserve scripts, reports, native arrays and failed-attempt evidence. Earlier frozen M0/M1/M1.5 and policy results are unchanged.
