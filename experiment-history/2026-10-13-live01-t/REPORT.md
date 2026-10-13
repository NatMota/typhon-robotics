# LIVE01-T: behavior preserved, frame provenance failed

**Verdict: failed metadata qualification.** Two actual-inference executions preserved the original V4 behavior exactly, but all 1,254 checked image-source references were unresolved. The candidate remains diagnostic and was not repaired or rerun.

## Experiment

Original V4-M6 checkpoint `c884e5052bc2734c1edf7d33498497cb26b27d88db1b32e3289fc46ea2e453dc`, film-demo_0006, seed 7417. Original telemetry versus append-only source metadata, with the same passive OBS06 observer, H30, 20 Hz, 40 reset ticks, 1,436 policy execution ticks and 24 settling ticks (1,500 total).

Executed code: `291caa9f9cbf3345592d8020ef76f9f22927f102`. Fixture expectations were frozen at `18e29a2`, qualified patch at `25de6e5`. See the saved patch manifest and protocol. Original actor input, including its privileged simulator-derived held marker, is unchanged; this is not qualification of an entirely camera-only actor. The observer and telemetry consumer have no control authority.

| Check | Actual result |
|---|---|
| Four consumer fixtures | 4 passed |
| Existing telemetry tests | 3 passed |
| Prepared tensors, worker RNG/noise, model outputs and query schedule | Exact equality across 48 paired queries |
| Applied actions and recorded physical/controller fields | Exact equality across 1,460 paired ticks |
| Passive observer values | Exact equality |
| Final position error, both runs | 11.4245908809 mm |
| Final yaw error, both runs | 85.3315887842 degrees |
| Strict task success, both runs | False |
| Image references resolved | 0 / 1,254 |
| Source frame events recorded | 4,540 |
| Query / observer / publication records | 48 / 366 / 52 |
| Command attribution | 1,436 policy execution; 24 settling; 0 unknown |
| Publisher errors | None |

The unused parent Python RNG states differ initially and remain unchanged within each run; relevant worker RNG/noise and executed behavior match. Legacy wall-clock publication times are not required to match.

## Failure and its demonstrated cause

The metadata registry associates render records with array identities. The existing `policy_observation()` constructs each RGB observation with `np.asarray(frame, dtype=np.uint8).copy()`. Pixel content survives, but array identity does not. The registry cannot associate these new arrays with their render records, and that gap propagates to current and historical model inputs, observer inputs and published images.

A CPU probe invoking the real observation helper demonstrates identical pixel hashes, different arrays, no shared memory and failed provenance lookup after the copy. The first live observation independently contains unresolved references immediately after two properly recorded render events. The 1,254 failures comprise 96 current-query, 322 history-query, 732 observer and 104 publication references.

Unknowns were retained. Acquisition timestamps were not substituted with publication time, and cached kinematic fields were not assigned invented computation timestamps. This is a concrete implementation failure in the new provenance layer; the paired runs do not show changed policy behavior.

The four passing fixtures tested settling attribution, repeated publication, out-of-order delivery and derived-versus-integrated phase interpretation. They missed the real render-to-observation copy boundary. Their pass is therefore insufficient to qualify this candidate.

## Preservation and interpretation

Both videos fully decode. Read-only reruns of the analysis and CPU counterexample reproduce their three output files byte-for-byte. Tracked runtime files remain identical to the executed commit. See `closure-verification.json`.

The execution and failure analysis occurred in the original morning window. Final read-only verification, report and backup resumed after Nat's evening "Keep going"; this administrative closure is outside the original 45-minute cap. It adds zero rollouts and zero training. No repair was folded into the failed experiment.

A separately versioned follow-up could bind copied observations to render records captured during the same existing observation call, with a copy-boundary integration fixture before one frozen parity pair. This remains a proposal, not an implemented fix. The planner is choosing the next bounded OST task.

Monitoring parity does not qualify held-state recognition, orientation, autonomous phase advancement, candidate ranking, historical archive reproduction or the 68 outstanding intervention trials.

## Reproduction and artifact locations

Repository branch: `research/live01-t-telemetry`.

Powerglide raw evidence: `/var/tmp/typhon-live01-t-20261013/`.

Mint archive target: `/media/nm/Fast Drive/Typhon/research-20261012/live01-t-telemetry/`; the separate backup receipt records actual transfer verification.

With the original raw directory mounted and the existing simulation environment:
```sh
.venv-sim/bin/python scripts/live01_t_analyze.py
.venv-sim/bin/python scripts/live01_t_copy_probe.py
```

These commands reanalyse saved evidence without inference or physics. The frozen execution driver retains its expired hard deadline and refuses a new run. A new execution requires a new protocol/output directory/version; changing that deadline is not part of this result.
