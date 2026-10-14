# DATA07-R — Does the approach repair work on eligible training cases?

**B completed 3/4 native loading trials. The one qualifying failure remained a failure after the one-chunk C-pose substitution. No successful correction examples were qualified, and no model was trained.**

This is a small training-family diagnostic, not a held-out estimate. B is LOAD04's counterbalanced checkpoint; C is its correlated-supervision control. These names do not denote task A/B or the older V1–V4 runs.

| Training family | Execution | Strict loading | Final position error | Final yaw error | Tilt at first supported lift |
|---|---|---|---:|---:|---:|
| 0001 | Native B | Fail | 427.73 mm | 55.65° | 89.98° |
| 0002 | Native B | Pass | 10.91 mm | 0.71° | 1.64° |
| 0003 | Native B | Pass | 1.52 mm | 0.79° | 0.47° |
| 0005 | Native B | Pass | 5.22 mm | 0.58° | 1.28° |
| 0001 | One C pose chunk, then B | Fail | 428.09 mm | 55.63° | 89.05° |

[Matched video](videos/native-vs-pose-repair-demo_0001.mp4) · [Contact sheet](comparison-contact-sheet.jpg) · [Per-episode records](episodes) · [Validation](validation.json)

## What happened

In native 0001, the first finger contact was at control tick473, the first tilt above15° at478, bilateral contact at505 and a supported10mm lift at507. The robot lifted the brick approximately sideways and still held it near its starting location at the final tick1500. It did not place or release it in the loading fixture.

The predeclared repair boundary was tick460, query14: the last stable query boundary before first contact. We generated C's proposal from exactly that B state. Only the first chunk's position and6D wrist channels0:9 were substituted; B's gripper and unused channels9:20 remained exact. B controlled subsequent chunks.

The repaired execution still contacted at473 and tipped at478. Bilateral contact occurred at503 and supported lift at513. It likewise ended holding the brick outside the fixture. C's substituted position targets differed by7.10mm on average,23.46mm maximum; wrist targets differed by0.23° on average,0.45° maximum. These are proposal differences, not claims about executed displacement.

The earlier LOAD06 rescue on development family0012 therefore does not establish a general correction teacher. This one training-family counterexample rejects automatic promotion of the same intervention into positive recovery data. It does not establish why C failed here, or whether an earlier/longer intervention or a verified recovery controller would work.

## Frozen design and controls

- Exact LOAD04 loading training configurations0001/0002/0003/0005, in that order. Original simulation seed131902 and predeclared policy seed912171.
- Four native trials; at most the first two failures that tipped above15° before bilateral acquisition. Only0001 qualified. No perturbations or additional seeds were added to obtain another failure.
- Visible fixtures, O2 observations, domain29, native generic-interleaved rotation representation,30 actions per prediction,10 denoising steps, bfloat16. No harness, candidate selection or training.
- Unmodified benchmark horizon: reset ends at control tick40;1436 predicted actions execute to1476;24 passive stability steps end at1500. There is no automatic release, correction or truth-driven stopping.48 chunks are requested, with the final chunk partially executed.
- The exact benchmark scorer requires position within15mm, yaw within10°, upright marked face within10°, footprint within the target, correct loading height within2mm, release, stability and seating. The10mm supported-lift diagnostic is distinct from full success and from the legacy attempt logger's70mm lift threshold.
- Identical initial camera pixels, float32 policy proprio, qpos, qvel, mocap and truth were required against each archived training initial state.
- Native action prefixes were replayed, not restored from an incomplete snapshot. The repair boundary passed the qualified semantic physics/controller/history/RNG comparison at zero tolerance, retaining the existing explicitly documented exclusion of the rendering-related XML statistic/meansize.
- Native RGB/proprio history and all prepared input hashes matched at the branch. Recomputed B output exactly matched its archived chunk. C used identical captured generation noise. Subsequent B queries used the corresponding native per-query noise.
- Simulator truth was used to select/verify the diagnostic intervention and score outcomes. The actor received only RGB, robot proprio/history and the supplied task text. This is an offline causal diagnostic, not a deployable truth-free recovery trigger.
- Development families0012/0013 and their hybrid traces were not imported into this experiment or training.

The raw archive preserves native predictions, input camera arrays, exact sampled noise, controller targets/torques, contact records, gripper transitions, semantic boundary states and full episodes. All five evaluated episodes have empty execution, artifact and evaluation-error lists.

## Interpretation and data decision

The measured native score is3/4 on these four training families at one policy seed. The repair converted0/1 selected failures. Treating that result as another four-case autonomous model score would be misleading.

**No successful repaired trajectory enters training.** These executions remain diagnostic evidence. The three native successes are observed native rollouts, not newly approved correction supervision. Larger data collection, another repair protocol and any subsequent fitting require separately frozen experiments. AlphaNIST expansion remains Parked.

Visual review of the saved comparison confirms that both0001 executions keep the brick near the source and fail to load the fixture. The contact sheet and video align by control tick, with labels added outside the original camera images. The paired video has367 frames at5fps.

## Qualification correction

One initial check stopped before any prediction or robot action: archived proprio was float64, whereas the production actor casts it to float32. The maximum difference was2.74e-8. The check was corrected to compare the exact float32 representations; no numeric tolerance was introduced. The failed preflight, original script and amendment are preserved under raw qualification-failure-01 and protocol-amendment-01.json. Cases, noise seed, controller, horizon and45-minute deadline were unchanged. This was not an outcome-based retry.

The reused worker's legacy initial_isolation_checked flag is not cited as qualification evidence because expected_nontext was empty. DATA07's independent exact initial-state and RGB checks provide that evidence. The repair additionally checks every prepared input hash.

## Reproduction and provenance

Branch: research/data07r-training-acquisition-repairs.
Base: f6638f99fd0958f79c9a208ccd466983ea6d3b68.
Execution: Powerglide RTX4090; no GPU process left running.

The executable driver is scripts/data07r_run.py. scripts/data07r_analyse.py regenerates tables, event summaries and paired visuals from the completed raw traces. The frozen protocol contains full scenes, source filenames, checkpoint hashes and split membership.

On the existing Powerglide checkout with its qualified simulator/model environments and LOAD04 inputs:

~~~bash
.venv-sim/bin/python scripts/data07r_reproduce.py --output /var/tmp/typhon-data07r-reproduction-UNIQUE
~~~

This reruns the frozen four-native/up-to-two-repair design into a new directory and starts a fresh bounded clock. It refuses to overwrite an existing directory. Do not point it at a frozen result directory. This command was prepared, not executed as an extra set of trials.

Raw source: /var/tmp/typhon-data07r-20261014.
Mint mirror: /media/nm/Fast Drive/Typhon/research-20261014/data07r-training-repairs/raw.
See raw-file-manifest.json and mirror-verification.json for byte counts and SHA256 verification. The private repository contains code, protocol, quantified results and videos; bulk camera/noise/state arrays remain in the verified Mint mirror. No checkpoint weights are copied into Git.

Fresh-machine restoration remains untested. Frozen M0/M1/M1.5 and earlier research branches were not changed.
