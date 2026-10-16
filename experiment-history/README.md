# Typhon experiment history

This is an account of what was tried, what was measured, and what remains unproved. It covers the 7–9 October 2026 experiments and the final evaluation evidence available by 22:55 UTC on 9 October. The documentation was finalized after midnight BST on 10 October; no new experiment or retrospective scoring change is implied by that later writing time. The 9 October cutoff was the original planning target for this evaluation. The official Humanoid announcement dated 7 October, supplied by the user and corroborated by [Humanoid's company Updates page](https://uk.linkedin.com/company/humanoidai) and the recruiting email dated 9 October, extended the internship deadline to **16 October 2026 at 23:59 BST**. This 10 October documentation was therefore written before the extended deadline; the experimental timestamps and results are unchanged.

The final learned-policy evaluation comprised **128 scored episodes: twelve common eight-case suites and four separate eight-case diagnostics with a four-second action horizon**. No initially unfinished case met every success criterion. Every counted success preserved a goal that was already complete. The strongest partial result was meaningful manipulation: V4-M6 grasped, lifted about **161.4 mm**, transported about **246.3 mm**, and released stably inside the target. Its **11.36 mm position error passed**, while its **85.06° yaw error failed**; orientation was the only failed final criterion.

All robot executions described here were simulations. There was no real-robot demonstration or validated factory deployment. Human phone recordings show a real task, but they do not contain measured robot actions.

This account distinguishes a learned task selector, geometric-expert executions, visual completion checks, learned robot actions, and preservation of an already-correct arrangement. Lower training loss, a valid action tensor, a completed training job, and successful manipulation are different observations.

## Reading the evidence

Experiment names identify frozen protocols. Original personal M3–M6 runs are distinct from later M7 data versions V1–V4 and their M3–M6 adaptation branches. “M6” without a version in the completion-harness section means the earlier frozen M6-FT3000 checkpoint, not one of the later H100 models.

Source identifiers in parentheses correspond to SOURCE_INDEX.json. The index records original relative evidence locations, public evidence locations where known, and exact SHA-256 hashes retained during review. Missing hashes or unrecovered source copies are explicitly marked. EVIDENCE_GAPS.json separates limitations of the experiments from limitations of this documentation recovery.

The first complete history was written and verified on the experiment host before that host became unreachable around midnight BST. This publication copy was reconstructed from the already-reviewed evidence notes, then updated with the final evaluation report. It is not claimed to be byte-identical to that first draft. Its original draft hash is retained in the source index. The model, rollout and report hashes below identify experiment artifacts independently of the prose.

## 1. The initial task-selector pilot

The 7 October pilot used ten public MimicGen Sawyer demonstrations, 6,288 frames at 20 Hz, to derive placement-order labels. Six whole episodes supplied training, two validation and two test. A 68,613-parameter temporal transformer selected a part, waited, finished or reported unavailability; a shared motor executor performed the movement. Training requested one or two placements, while three- and four-part goal compositions were held out. RGB-derived features, supplied goals, history, engineered event labels and disturbances were part of this setup. This was not end-to-end robot motor learning from the recorded task dataset. (MINT, CODE-MINT-PPO)

This early pilot **did run behavioural cloning followed by clipped on-policy PPO**. Four conditions each ran 16,384 PPO steps with one training seed: clean-source, disturbed-source, a retrained history ablation, and a shuffled-label control. Behavioural cloning took about four to five seconds. Recorded PPO times were 306.678, 182.022, 161.088 and 159.093 seconds respectively on a GTX 1060. These measurements concern the small selector and cannot be used as X-VLA training estimates. (MINT-CLEAN, MINT-DISTURBED, MINT-NOHISTORY, MINT-SHUFFLED)

The 400-outcome RGB macro-surrogate evaluation was separate from the 240 MuJoCo physical-simulation trials. On held-out macro compositions, the rule reference scored 20/20 clean and 17/20 disturbed. The clean-source selector matched those counts; the disturbed-source selector scored 10/20 and 8/20. The no-history condition scored 10/20 and 11/20. The shuffled-label model recovered to 20/20 and 17/20 after PPO. These controls did not establish a demonstration-specific advantage or a benefit from temporal history. (MACRO)

The physical-simulation results were:

| Condition | Familiar clean | Familiar disturbed | Held-out clean | Held-out disturbed |
|---|---:|---:|---:|---:|
| Rule reference | 19/20 | 15/20 | 18/20 | 15/20 |
| Clean-source BC plus PPO | 20/20 | 15/20 | 19/20 | 17/20 |
| Disturbed-source BC plus PPO | 19/20 | 15/20 | 9/20 | 7/20 |

These trials included explicitly recorded interventions or assistance and false completion claims. They were simulation results, not physical robot demonstrations. The macro surrogate and the MuJoCo trials should not be pooled. (PHYSICAL)

A separate 7 October simulator/GPU readiness exercise passed eight tests, ten nominal expert trials and one learned two-part smoke case that began already complete. The general gate remained false because it required at least ten applicable trials. A one-case preservation smoke test did not establish general task completion. (UCL)

## 2. M0: personal recordings and provenance

The personal recording set contained eight original phone videos arranged as four paired views. Curation produced sixteen clips: seven normal, four awkward, three correction and two already-completed examples. The initial “six normal” description was corrected to seven in the selection record; no extra observation was fabricated. Fourteen examples involved manipulation and two were no-ops. A further candidate, personal_demo_0017, was excluded. (M0)

The primary view was canonicalized by a 90° counter-clockwise rotation while originals were preserved. Annotation was sparse, approximately 2 Hz. Unknown observations remained unknown instead of being silently interpolated into measured labels. Tracking coverage was 916/990 frames, or 92.53%; retained-frame coverage was 499/526, or 94.87%. Primary-view orientation coverage was 408/488, or 83.61%, and retained orientation coverage was 229/264, or 86.74%. These are coverage figures, not pose-estimation accuracy. (M0)

All sixteen curated clips visibly ended correctly, but “visible correct endpoint” was a human-video observation, not a robot success label. The target-plane homography supplied dimensionless coordinates. It did not establish physical calibration, block mass, contact forces or a mapping from phone pixels to robot joint commands.

Later synchronization work improved the pairing of the two phone views. It did not turn those videos into robot-action demonstrations; that distinction remains central to the experiment.

## 3. M1: the simulator and geometric-expert reference

M1 established a Panda task in MuJoCo 2.3.7 and robosuite 1.4.1, with a provisional 50 × 35 × 30 mm, 80 g block proxy. Control ran at 20 Hz with a 1,500-step budget. These were declared simulator parameters, not measurements of the real object. (M1)

The geometric expert completed nominal feeding with 4.261 mm position error and 0.165° yaw error in 209 execution steps. Nominal loading finished at 3.8 mm and 0.421° in 200 steps. A failed-grasp retry finished at 4.443 mm and 0.060° in 278 steps; a 45° loading case finished at 3.723 mm and 0.294° in 192 steps. Exact-repeat checks supported reproducibility of these simulated references. (M1)

The scorer required 15 mm XY tolerance, 10° yaw and tilt tolerance, full footprint inclusion, the declared support-height condition, an open gripper with release and no block contact, and stability over at least five samples spanning 0.4 seconds. The stability bounds were 2 mm translation and 2° yaw. A later image classifier did not replace all these physical evaluator criteria.

Startup repairs included moving an unsupported table-offset setting into XML and restoring fixture visibility through the appropriate geometry group. Affected executions were rerun; success thresholds were not relaxed. Seventeen new tests and eight existing tests passed. The expert’s results showed that particular simulated tasks were executable under the declared controller, not that a learned policy could perform them. (M1)

## 4. M15: reconstruction from personal endpoints

M15 imported fifteen of the sixteen curated examples; personal_demo_0004 was not reconstructable. Personal reconstructions passed 13/15 and synthetic controls passed 8/15, including two cases that were already complete. The personal failures were personal_demo_0011 and personal_demo_0015. These were geometric-expert reconstruction outcomes, not X-VLA learning outcomes. (M15)

The virtual target mapping used a 100 mm region. A separate catalogue-informed appearance configuration named DUPLO 31110pb074, with a nominal printed-face-up size of 32 × 43 × 32 mm and mass 12.7 g. Those catalogue values were not physical measurements of the filmed brick, and the older M1 proxy was not retroactively changed.

Raw endpoints required two supported stationary observations within 1.75 seconds. Reconstructions did not initialize from a later, already-placed state to make manipulation easier. Side-by-side human and robot videos compared task behaviour; they did not establish framewise temporal alignment or direct imitation of every human intermediate motion.

One visual defect mattered for later success checking: the original seven-segment-style “2” was symmetric under a 180° image rotation. The physical evaluator still checked directed yaw, but the rendered glyph made that distinction ambiguous to image-only methods.

## 5. M16: making directed orientation visible

M16 introduced a second visual version with an asymmetric curved numeral and body-local stud appearance. This changed visual appearance only; the collision geometry, physical parameters and command sequences were preserved. The 0°/180° blue-mask intersection-over-union changed from 1.0 for the legacy glyph to 0.506 for the revised one. (M16)

A conservative visibility gate passed all four checks at 512-pixel resolution and none at 336, 256, 224 or 128 pixels. This was a renderer/visibility test, not a measured orientation-classification accuracy curve. Replaying thirty M15 executions with the new visual appearance reproduced the personal 13/15 and synthetic 8/15 outcomes with the same actions and physics. It therefore isolated the visible orientation cue without claiming improved learned control.

The legacy visual version and the revised version remained identifiable. Results obtained with them must not be silently merged, especially for 180° image-based completion claims.

## 6. M2: unchanged pretrained X-VLA

The first frozen general-policy comparison used forty scenes, twenty feeding and twenty loading. The geometric expert passed 19/20 feeding and 17/20 loading. Unchanged X-VLA LIBERO passed 0/20 in each family. The failed learned-policy rollouts were runnable executions, not missing-model placeholders. They recorded no block contact and no block movement above 1 mm. (M2)

The pinned LIBERO revision was 129e71460678b7236cee6fc9707f09d9fa0c3590, with 879,482,456 model parameters. The configuration used domain 3, a predicted horizon of thirty actions and ten executed actions per receding-horizon call. A prompt-length issue was corrected to the 128-token limit before the primary run. The simulator supplied overhead and oblique views, while the pretrained camera convention was external and wrist; that mismatch was a limitation, not an isolated causal finding.

Across 1,920 predicted chunks, recorded mean inference time was 112.84 ms. Batch wall times were 84.48 seconds for the expert and 546.79 seconds for the pretrained policy. These were descriptive measurements under the recorded resource conditions, not a controlled sole-model hardware benchmark.

The four expert failures replayed as joint-limit transfer failures. They were retained and did not imply that the tasks were physically impossible. An optional RGB-controller comparison was not run.

## 7. M30: action contracts, cropping and training plumbing

M30 examined observation and action representations before claiming learning progress. The O2 observation retained full 512-pixel primary and secondary images plus a 96-pixel crop as a third view, processed at 224 pixels. Missing crops were marked unknown; the crop path did not receive object truth, persistence fabricated from future frames or a later successful pose. Fourteen of thirty-eight inspected trajectory frames had a valid crop. Warm inference was approximately 117–118 ms, versus about 112 ms for the simpler O0/O1 representations. (M30)

Two traces containing 745 commands round-tripped through action encoding/decoding with maximum numerical error about 1.33 × 10^-15. A render-skipping optimization failed exact equivalence and was not adopted. A checkpoint reload mismatch of about 9.42 × 10^-6 was traced to mismatched gradient-enablement flags; matching the flags gave exact reload behaviour.

A native hundred-update training smoke test reduced loss, but its training and development rollouts still failed. Reading compressed arrays once reduced that hundred-update wall time from 230.1 seconds to 14.18 seconds. This removed redundant I/O; it did not change the learning objective or demonstrate manipulation.

These experiments used the original 50 × 35 × 30 mm, 80 g physics proxy. The catalogue appearance configuration was not a retrospective correction to the M2/M30 physical model.

## 8. Original M3: prompt adaptation and personal/synthetic controls

Original M3 trained only the domain-3 prompt entries, 32,768 effective trainable parameters. Six personal and six synthetic trajectories each supplied forty-eight complete thirty-step windows. Each adaptation used 1,000 updates with batch size four and seed 17 or 29: 4,000 window presentations, not 4,000 independent demonstrations. (M3)

The five conditions were frozen baseline F, personal P17, synthetic S17, personal P29 and synthetic S29. Across the same forty scenes, every condition scored 0/20 feeding and 0/20 loading: two hundred rollouts in total. None recorded block contact or movement above 1 mm. Training losses and action RMSE improved without producing task success.

Recorded optimizer-loop times were 145.62, 145.81, 145.68 and 146.17 seconds for the four trained variants. The fixed personal training goal was 177–204 mm from feeding test goals and 260–284 mm from loading goals. This evaluation did not qualify an in-distribution personal feeding result. Personal and synthetic conditions also differed in layout and shared the personal appearance; they were not a clean single-factor ablation of data origin.

## 9. Original M4: correction exposure and completed-state preservation

Original M4 compared ordinary continuation PN, using personal_demo_0001/0002, with correction continuation PR, using personal_demo_0007/0009. Both started from P17 and used 500 updates, batch size four, forty-eight windows and fresh optimizer state. Recorded loop times were 72.02 and 72.55 seconds. (M4)

The recovery set contained sixteen unfinished cases per model: four wrong-orientation, four misplaced, four engineered post-empty-grasp and four scripted-displacement cases. None of F, P17, PN or PR completed any of those sixteen cases. F, P17 and PN preserved all four already-completed cases; PR preserved two and disturbed the others, with a maximum recorded displacement of 39.08 mm.

PN and PR also scored 0/20 feeding and 0/20 loading. The report recorded 160 new trials across these comparisons. The engineered empty-grasp condition followed a fixed 98-step prefix, not the learner’s own discovered failure. The displacement was a simulator pose injection, not a measured human intervention.

There was no learned completion head in these policies. “False completion” was therefore not a meaningful zero-valued policy metric; the relevant measurements were physical evaluator outcomes and disturbance of already-completed arrangements.

## 10. Motion diagnostics and temporal execution

Eight reachable-pose tests passed with approximately 0.25–0.53 mm position errors. Expert action decoding agreed to about 2.22 × 10^-16. This made a universal decoder arithmetic fault unlikely, while leaving model-output quality and control interactions open. (MOTION)

An exact 1,436-command PR trace was replayed. Fixing the wrist orientation reduced one raw-target position error from 460.169 mm to 10.998 mm; a holding variant reached 0.476 mm. A 145.744° rotation jump and poorly conditioned rotation columns were also recorded. These were useful causal diagnostics of a particular trace. They did not establish that every failure had a single orientation bug or justify removing the declared joint limits.

Thirty rollouts compared execution horizons H30, H10 and H5 for two checkpoints across five cases; none of the eighteen unfinished trials completed. A fresh H10/H30 comparison added twelve rollouts and again scored 0/8 unfinished cases. A PR preservation gain did not reliably repeat.

History was corrected to track the last executed target rather than advancing through the discarded future tail of a predicted chunk. A later one-intensity-level RGB difference explained some historical P17 discrepancies, so new exact references were preserved. A symlink/setup failure remained a setup failure, not a scored policy episode.

These diagnostics separated predicted chunk length from executed prefix length. They did not demonstrate an effective learned temporal abstraction, a learned subgoal controller or recovery simply by increasing or decreasing the open-loop execution horizon.

## 11. Original M5: action-head and LoRA adaptation

M5 deliberately restricted supervision to personal_demo_0001: forty-eight windows from one source trajectory. It compared frozen F, 3,000-update prompt adaptation C3000, and A3000, with 1,000 prompt/action-head warmup updates followed by 2,000 LoRA updates. LoRA used rank eight, alpha sixteen and 266 adapted modules. The data windows were identical across the trained comparison. (M5)

The twenty-four core evaluations gave each model 0/3 source and 0/3 development completions. Strict already-completed preservation was F 2/2, C 1/2 and A 2/2. All three preserved the object arrangements in both completed cases; C’s strict failure involved the gripper/stability condition. That distinction prevents object non-disturbance being confused with passing the complete evaluator.

A further W1000 evaluation failed. Optional loading was not run because the declared gate was not passed. C took 532.31 seconds and A took 759.73 seconds, with recorded peak memory around 15.004 GB. Because action-head and LoRA changes occurred together, this was not an isolated estimate of the effect of LoRA.

## 12. Earlier M6: full joint adaptation

Earlier M6 used the Pt base checkpoint separately from the LIBERO baseline, a custom domain-29 initialization, and an interleaved rotation permutation at the unchanged OSC controller boundary. It again used only personal_demo_0001 and forty-eight windows. The 3,000-update schedule comprised 1,000 prompt/head updates, 1,000 ramp updates and 1,000 joint updates. Parameters, gradients and Adam state were FP32 with BF16 autocast. (M6)

The reported 978.51 seconds included the final save, probe and hash, not just the optimizer loop. Peak memory was about 17.782 GB and exact reload was verified. The primary sixteen trials compared custom initialization with FT3000. Each scored 0/3 source and 0/3 development completions. Already-completed preservation was 1/2 and 2/2 respectively. Eight additional untouched-Pt/domain-3 diagnostic trials scored 0/3 source, 0/3 development and 0/2 completed preservation.

No supported lift was established in these original M6 trials. The trained approach stalled. Differences in base checkpoint, domain, loss and adaptation scope prevent this from being a clean “full tuning versus LoRA” comparison.

The earlier frozen checkpoint used later by the completion harness was:

SHA-256: 03a55fa46088368da0b8d92091ec3a4a5a539ae2772bf2cfd4c592b92b745a85.

Its provenance is the original personal M6 training/final.pt. It is not one of the new V3/V4 H100 checkpoints.

## 13. Data expansion: what V1–V4 actually contained

The coverage audit found fourteen older training sources, grouped as P6/R2/S6, plus six validation sources: twenty NPZ files in total. This did not change the fact that original M5 and M6 trained on only personal_demo_0001. New correction examples personal_demo_0018/0019 and two synthetic counterparts added four C sources, giving eighteen training sources across P6/R2/S6/C4. (COVERAGE)

Visual enrichment covered nine episodes and 726 frames. Runtime grasp status could remain unknown; privileged oracle annotations were identified separately. The full human set contained thirty-two enriched clips but **zero native twenty-dimensional robot-action labels**. Annotated pixels, apparent contact and a held-state graphic were not measured robot pose, grasp force or an end-to-end human-to-robot action mapping. (ENRICHMENT)

The first V1 catalogue contained 1,007 samples and 2,014 camera arrangements with dense supervision and no held-state cue. V2 had 316 semantic samples and 632 camera arrangements with dense supervision and the held-state cue. Early V3/V4 placeholder catalogues also had 316 sparse samples, but did not yet implement the requested four-second physical horizon and control loop. Renaming a catalogue did not establish temporal abstraction. (CATALOG)

An exporter update enabled the green held-state cue for future V3/V4 exports without changing the action targets or overwriting old exports. This was an annotation/export change, not a validated deployable grasp estimator. Later reports identify immutable catalogue and exporter versions so that dense versus sparse, cue presence and source coverage are not silently treated as one controlled ablation. (CATALOG-STEERING)

The actual temporal Run 3 data, followed by its v3_1 correction, contained 316 semantic parents expanded into 388 windows and 776 camera candidates. The eighteen underlying sources comprised ten personal and eight synthetic sources. Scope counts were 116/98/84/18 parents, becoming 118/105/117/48 windows. Windows spanned at most four seconds, or eighty 20 Hz ticks, and used at most thirty exact action anchors, retaining endpoints and gripper transitions. There were 10,064 valid action entries and 1,576 masked entries. (RUN3)

The history input was causal: sparse observations at approximately 1 Hz over the preceding four seconds, at most five images, without future frames. Robot command timing remained 20 Hz. A balanced training epoch contained 1,664 presentations drawn from 776 unique camera candidates, with 832 presentations per camera. Seeds 17 and 29 changed ordering rather than changing those group counts.

The timing audit measured a maximum interpolated target deviation of 15.0704 mm and exact gripper-transition preservation. Two geometric-expert timing replays passed paired lift/full-intent checks. These results qualified the data/timing machinery; they were not learned-policy success. (TIMING)

In the implemented trainer, history vision features were mean-pooled with explicit time/camera information. This compresses temporal information. Frozen-vision M3/M4 could cache history features; joint-adaptation M5/M6 recomputed them. The runtime request remained externally supplied. An online learned subgoal selector, learned-value search and future-aware goal oracle were not demonstrated. (CODE-M7-TRAIN, CODE-M7-TIMING)

## 14. Video synchronization and action alignment

The phone-view audit recovered offsets of +4.58, -2.44, -1.41 and -1.56 seconds for the four paired originals. Motion-based uncertainty was about ±0.25 seconds and static/audio checks about ±0.05 seconds. Variable-frame-rate random seeking could be wrong by about 0.1 seconds. Sequential decoding repaired frame selection; selected-frame residuals of 20, 24.4, 17.8 and 6.8 ms were decoding/selection checks, not sub-25-ms proof of physical audio/video synchronization. (SYNC)

The robot dataset had eighteen simulated episodes, 1,134 paired RGB instants and 4,519 robot commands. Images were sampled at 5 Hz and commands at 20 Hz. RGB was rendered before the selected command, following forty warmup steps. The first proprioceptive entry used measured state; subsequent entries represented the previous command under the recorded training convention.

Those simulated RGB/action pairs were genuine robot-observation/action examples from the simulator and geometric executor. They were not phone frames paired with measured human or robot actions. The human demonstrations informed task reconstruction; human intermediate motions, pauses and exact frame times were not transferred one-for-one. Side-by-side video similarity cannot establish that missing correspondence.

Training was held while these semantics were reviewed. An initial five-update H100 smoke had finite losses but no verified full checkpoint flush and was not a completed training matrix. A subsequent three-update native joint-training smoke exercised all phases, twelve examples and all scopes/views/splits. Mean update time was 1.878 seconds, peak memory 17.934 GiB, and a roughly 10.5 GB checkpoint passed exact readback. A previous roughly 3.4 GB quota-limited partial checkpoint was preserved as a failed save; the successful replacement used the intended local disk. Maximum observed token length was 172 under the 256-token cap. (TRAINING-HOLD, NATIVE-SMOKE)

## 15. Completed training and V4 aggregation

At the final reviewed training snapshot, V1, V2 and V3 had completed their M3/M4/M5/M6 schedules of 1,000/1,000/3,000/3,000 updates. V4 completed 500/520/520/500 updates. A briefly observed V2-M5 count of 2,995 was an earlier live status, not the final count. All sixteen trained version/branch combinations therefore had a completed training status, but only twelve received the common evaluation described below. (V12-STATUS, V34-STATUS, RUN4-STATUS)

The later runs used one four-H100-80GB allocation. Reported V3 training durations were approximately 10.5 minutes for M3, 8.7 for M4, 52.5 for M5 and 43.7 for M6; M4 began after M3. These are run-specific measurements with the recorded scheduling. They are not a matched throughput benchmark against the original 4090 experiments.

V4 was one round of matching-student geometric-expert aggregation. For each branch, the matching learned checkpoint ran a 120-command, six-second prefix on the predeclared scenarios. The expert continued from that resulting state without resetting it. Original sources were preserved, learner-prefix actions were not converted into positive expert labels, and unsuccessful or nonconverged teacher continuations were quarantined. (RUN4-DATA, RUN4-TRAINING)

| V4 branch | Original sources retained | Clean teacher additions | Windows | Camera candidates | Updates |
|---|---:|---:|---:|---:|---:|
| M3 | 18 | 3 | 470 | 940 | 500 |
| M4 | 18 | 4 | 497 | 994 | 520 |
| M5 | 18 | 4 | 496 | 992 | 520 |
| M6 | 18 | 3 | 459 | 918 | 500 |

Every listed candidate was visited in the recorded continuation. Process elapsed times derived from start/end timestamps were 361.37, 361.31, 661.34 and 561.26 seconds respectively. Those include job overhead and differ in scope from older optimizer-loop timings. (RUN4-STATUS)

Accepted additions followed an unfinished six-second student prefix. Budget expiry did not prove irrecoverable failure or failure under the full episode budget. Three explicit empty-grasp attempts across M5/M6 produced no clean accepted empty-grasp correction: two teacher recoveries failed, while one final-successful recovery had nonconverged primitives and was quarantined.

This was **one-round DAgger-style supervised aggregation**, not PPO, Q-learning or a repeated autonomous failure-discovery loop. The earlier small task-selector PPO experiment remains a separate branch of the history. Neither completing V4 training nor accepting teacher suffixes established a policy improvement before independent evaluation.

## 16. Final frozen-policy results

The common protocol used eight frozen cases: six unfinished/correction cases and two already-completed cases. It supplied the declared full-intent request with a 1.5-second predicted action horizon. Changing the request wording and horizon relative to native training was itself an evaluation condition, especially for V1. Models shared scenes, so the rows are comparisons under one small frozen suite, not independent samples of general real-world performance. (M7-PILOT, M7-FOLLOWUP, DEADLINE)

| Version | Branch | All cases | Unfinished/correction | Already-completed preservation |
|---|---|---:|---:|---:|
| V1 | M3 | 0/8 | 0/6 | 0/2 |
| V1 | M4 | 2/8 | 0/6 | 2/2 |
| V2 | M3 | 0/8 | 0/6 | 0/2 |
| V2 | M4 | 2/8 | 0/6 | 2/2 |
| V3 | M3 | 0/8 | 0/6 | 0/2 |
| V3 | M4 | 2/8 | 0/6 | 2/2 |
| V3 | M5 | 0/8 | 0/6 | 0/2 |
| V3 | M6 | 2/8 | 0/6 | 2/2 |
| V4 | M3 | 1/8 | 0/6 | 1/2 |
| V4 | M4 | 2/8 | 0/6 | 2/2 |
| V4 | M5 | 0/8 | 0/6 | 0/2 |
| V4 | M6 | 2/8 | 0/6 | 2/2 |

V1-M5, V1-M6, V2-M5 and V2-M6 were trained but not included in this evaluated set. “Sixteen trained branches” must not be reported as sixteen common-protocol evaluation suites.

The separate, predeclared four-second diagnostic used V3 and the same eight-case structure:

| Branch | All cases | Unfinished/correction | Already-completed preservation |
|---|---:|---:|---:|
| M3 | 0/8 | 0/6 | 0/2 |
| M4 | 1/8 | 0/6 | 1/2 |
| M5 | 0/8 | 0/6 | 0/2 |
| M6 | 2/8 | 0/6 | 2/2 |

The twelve common suites contribute 96 scored episodes and the four diagnostics contribute 32, totaling **128**. Their horizons remain separate; pooling them as one success-rate estimate would conceal an experimental condition. All full successes in both tables were preservation. No unfinished task passed the complete evaluator. (DEADLINE)

The first thirty-two V1/V2-M3/M4 trials showed target positions outside the workspace while clipping/decoding arithmetic remained correct. Mean action-call latency was about 254 ms in that recorded setting, and requests occupied 152–155 tokens within the 256-token cap. This identified a model-output problem upstream of the checked arithmetic in those traces; it did not establish that all later failure modes had the same cause. (M7-PILOT)

Native simulator aborts interrupted an initial V3-M6 common run after one valid scored case and the initial V4-M5 suite after one valid row. A native abort was an infrastructure outcome, not a 0/8 model result. V4-M5 retained the valid original row and completed the remaining seven exact frozen scenes in fresh isolated processes with the same seeds, request and scoring, producing the final 0/8 aggregate. The two completed arrangements were displaced by approximately 43.47 and 124.16 mm. V3-M6 completed a separate full retry, producing 2/8. Its original partial row was retained but is not an additional independent trial. (H100-V4M5, DEADLINE)

The final V3-M6 retry pushed the source object about 20 mm without a bilateral grasp and did not complete a correction. Decoder arithmetic checks passed. In contrast, V4-M6 supplied real grasp/transport progress in personal_demo_0006: thirty-six bilateral-contact steps, forty-four steps with any contact, a verified lift above the 70 mm detection threshold, maximum lift about 161.4 mm, and maximum displacement about 246.3 mm. Final displacement from the initial pose was about 210.9 mm. The measured maximum lift should not be replaced with the lower 70 mm verification threshold. (H100-V4M6-DIAGNOSTICS, H100-V4M6-EPISODE)

That V4-M6 episode ended released and stable inside the target. Position, marked-face, footprint, support-height, release and stability criteria all passed. Position error was 11.3555 mm; yaw error was 85.0581°. **Only directed orientation failed, so overall task success remained false.** The episode contained 1,436 executed actions and forty-eight predicted chunks. Checked decoding arithmetic was correct, and no target or measured-position workspace violation was recorded in that episode.

H100 results were centrally transferred with SHA verification. The final local media index contained sixty-four videos. The earlier public media release was intentionally frozen at fifty-six and omitted the final two local suites; it is a selected media release rather than the complete final evaluation inventory. The final 128-episode summary hash is 389d4dfd9ee276d2628fc1d66067f0512aff73f04803230d16316f871a9caf26. (DEADLINE)

## 17. Qwen planning and the completion problem

A separate text-planning bridge tested Qwen3-VL-2B-Instruct with the earlier frozen M6-FT3000. It ran three decisions in each of two cases, with two candidate policy chunks per decision; two malformed planner schemas were recorded overall. The source case failed without contact or movement. The already-correct case was held correctly, but the bridge did not establish a reliable completion claim. Planner latency ranged from about 0.506 to 2.469 seconds, with median 0.646 seconds. This was a small wiring/diagnostic exercise, not proof of semantic instruction following. (QWEN-BRIDGE, QWEN-MANIFEST)

Proposed alternatives and estimates were kept separate from measured runs. Thor/Cosmos access was not obtained in the recorded attempt, which returned an authorization failure; no training or evaluation result for those models was established. (MODEL-ADVICE)

In the later Qwen service, the model confidently judged a 180°-wrong object as complete. That concrete false positive led to a fixed-camera RGB numeral-reference veto and a stateful completion gate. Completion then required three distinct confident observations, agreement with the reference image and measured gripper opening; workspace, joint and bounded-candidate guards constrained execution. This repaired the service logic for those tested cases, not Qwen’s underlying general visual judgment. (QWEN-SERVICE)

With the original M6, the bounded source diagnostic ran four decisions and twelve executed actions without finishing. An already-correct case was recognized without movement and the wrong-180° case was held. In a separate new V1-M4 assisted diagnostic, all twelve source candidates were rejected, producing zero executed actions; the correct case was recognized with zero actions. The wrong-90° case produced truncated, malformed Qwen JSON under the 192-token response limit and therefore became unknown. Independent RGB evidence mismatched; this was not successful Qwen orientation recognition.

Contradictory Qwen prose and malformed outputs were retained. Assisted-service outcomes were never added to the unassisted policy tables. Rejecting an unsafe or malformed proposal is a useful supervisor result, but it is not learned manipulation.

## 18. A small image classifier as the primary visual check

The next service mode replaced Qwen on the completion-critical path with a CPU image classifier. Its scope was the revised DUPLO feeding appearance in the fixed overhead camera. Unsupported loading and unobservable inputs return unknown. It combines a learned numeral/direction/occlusion component with visible target geometry, preserving the declared 15 mm position tolerance and full-footprint requirement. It accepts allowed placement variations rather than requiring exact reference pixels. It does not receive the current simulator object pose at inference. (CLASSIFIER, CODE-CLASSIFIER)

The model has 111,107 parameters, a 1,732-feature input, a 64-unit hidden layer and three outputs. The saved model is approximately 414 KiB. Its 0.8 abstention threshold was selected from validation evidence. Softmax confidence is uncalibrated; no calibrated-probability claim is made.

The first conservative fit produced no completions. Investigation found that an image-augmentation mask could include robot-blue pixels and assign an unknown label to an intact object numeral. The correction localized the augmentation to the orange-object region and preserved the full-object blue mask. That first attempt remains part of the history; its threshold was not silently treated as the final model.

Exploratory development used 1,380 training, 280 validation and 420 test images. The final exploratory test had 124 completions among 141 valid cases, with seventeen valid boundary cases unknown. Among 189 invalid cases, 178 were incomplete and eleven unknown. All ninety occluded cases were unknown; all thirty 180° and thirty 90° wrong-orientation cases were rejected. These development results informed the final choice and are distinct from the later confirmation set.

After freezing, a separate 336-image confirmation set contained 288 renders plus augmentation cases:

| Ground truth | Complete | Incomplete | Unknown |
|---|---:|---:|---:|
| Valid placement | 101 | 0 | 10 |
| Invalid placement or orientation | 0 | 144 | 9 |
| Occluded | 0 | 0 | 72 |

All twenty-four reversed and twenty-four quarter-turned bricks were rejected. There were no false-complete predictions in this finite confirmation set, with ninety-one total abstentions. This does not establish a zero false-positive rate in deployment. Fine angular boundaries, real-camera lighting, unseen objects and loading remain unvalidated. (CLASSIFIER-CONFIRM)

Final model SHA-256: bc15b4519f7787aecb5078df08029984ba3e9297ca4af48caba1620b0530ae0b.

Classifier source SHA-256 at freeze: 609dd72b1f21b6fc2fc8e535b9cbb279ce66b311473a294d6a4f5d7abe8ffa61.

## 19. Actual Typhon harness hookup with the earlier M6

The classifier was connected to the real service and tested with the earlier M6 checkpoint identified above. This was not a stub-only exercise and did not substitute a newly trained H100 model. Old Qwen mode remained available separately; the new mode uses the small classifier as the primary visual authority without a Qwen dependency. (HARNESS, HARNESS-AUDIT, CODE-TYPHON)

The completion latch required five fresh, distinct observations over at least 0.4 seconds, visible object-position drift no greater than 2 mm, and measured finger opening with sum of absolute finger positions greater than 0.055 m. The wrapper used image-derived table coordinates and scale for the translational stability check. It did not pretend that a coarse orientation class measured continuous yaw stability to 2°. The independent physical evaluator remained separate.

For unfinished visible cases, bounded test-time computation sampled three unmodified M6 candidates, rejected candidates violating the existing guards, ranked admissible paths by shortest/smoothest end-effector motion, executed at most three actions, and reobserved. Guards were checked before each executed step. This was bounded candidate filtering, not learned-value tree search, a guarantee of whole-arm collision avoidance or geometric-expert rescue.

The live test outcomes were:

| Case | Visual/service result | Executed policy actions | Independent task outcome |
|---|---|---:|---|
| Correct at target centre | Complete after five observations | 0 | Complete |
| Correct with +8 mm X offset | Complete after five observations | 0 | Complete |
| Correct with +8 mm Y offset | Complete after five observations | 0 | Complete |
| Wrong 180° orientation | Incomplete; eight decisions, twenty-four proposals | 24 | Failed |
| Wrong 90° orientation | Incomplete; eight decisions, twenty-four proposals | 24 | Failed |
| Opaque sensor-fault image | Unknown on all eight observations | 0 | Underlying object was correct |
| Original unfinished source | No completion within the bounded attempt | 49 | Failed |

The opaque-image case was a deliberately full grey sensor-fault diagnostic. It was not an independently validated model of every real occlusion. Its underlying simulated arrangement was already correct, making abstention rather than a visual completion claim the intended response.

The source attempt ran 500 service decisions in 84.205 seconds, generated seventy-five candidates and executed forty-nine policy actions. There were twenty-five admissible-prefix decisions, forty-five measured-joint-limit holds and 430 visual-uncertainty holds, usually with numeral_unobservable. Fifteen mid-prefix execution rejections were recorded. The first recorded joint-limit hold involved joint 4 at approximately -3.07520 rad against a -3.0718 rad lower bound. Final position error was 478.312 mm and yaw error 54.481°.

This was a full execution of the declared 500-decision bounded harness attempt. It was not 1,500 executed robot control steps, and the supervisor’s holds should not be described as completed policy work. The source remained unfinished.

Across forty-one decisions containing policy candidates, median decision latency was 188.713 ms, p95 200.962 ms and maximum 301.587 ms. Across 123 candidate inferences, median was 61.727 ms and p95 68.820 ms. Fifty-six visible complete/incomplete classifier calls had median 0.919 ms and p95 2.211 ms. The overall 494-call median of 0.500 ms included many quick unobservable branches. These were isolated-4090 measurements with the simulator paused during inference, not real-time robot throughput.

Seventeen prior service tests, eight Typhon service tests and seven CPU replays using saved actual simulator RGB passed. The actual live run additionally verified the frozen model hashes, current joint positions and velocities, end-effector state, measured finger positions and simulated actuator-torque dimensions. Per-case frames, raw classifier outputs, proposed/rejected candidates, executed prefixes and evaluator outcomes were retained. A preload-only first attempt was stopped before any cases and preserved with its reason; the scored run loaded the final classifier source and weights. (HARNESS-AUDIT)

The publisher separates commanded action from measured q, dq, end-effector pose, gripper position and MuJoCo actuator torque, with explicit source timestamps and scenario/model identities. Torque is simulator actuator torque, not a physical force sensor reading; unavailable temperature remains null. This makes failures inspectable without presenting simulated telemetry as measurements from a real robot. (CODE-TELEMETRY)

## 20. What the combined evidence supports

The work established a reproducible simulated task, explicit observation/action provenance, validated portions of the action contract, a sequence of preserved training/evaluation protocols, and a working guarded completion service for a narrow rendered setting. It also established repeated negative learned-policy results under the frozen tasks, including regressions in preservation after some adaptations.

Later V4-M6 produced a stronger partial outcome than the original contact-free/stalled runs: actual grasp, lift, transport and stable release near the target. The final orientation failure remains part of that result. The evidence does not support claiming completed learned manipulation, successful correction, real-robot transfer, or a benefit attributable solely to personal data, LoRA, full tuning, temporal abstraction or DAgger.

Broad randomization was discussed and some variation was implemented through source/layout differences, paired camera candidates, balancing and evaluation scenes. There was no isolated result establishing robustness to general lighting, camera, object, dynamics or contact randomization. Likewise, a held-state graphic, sparse causal history and physical-time action anchors were implemented components, but not independently proven sources of task success.

The most direct next scientific questions arise from observed failures: whether the grasp/transport result can acquire the required directed orientation; whether action targets stay within the intended task distribution without degrading completed-state preservation; and whether corrected supervision and temporal representations improve held-out execution under a matched protocol. Those are questions left open by this history, not results already achieved.

## Compact experiment ledger

| Stage | Actually measured or implemented | Boundary of the claim |
|---|---|---|
| Initial selector | Four BC/PPO fits, 400 surrogate outcomes, 240 physics trials | Small selector with shared executor; no personal motor learning |
| M0 | Eight originals, sixteen curated clips, coverage and provenance | Human observations, no native robot-action labels |
| M1/M15 | Expert task execution and fifteen imported reconstructions | Simulator feasibility, not learned completion |
| M16 | Asymmetric visible numeral; physics/action replay preserved | Better observable direction, not general visual accuracy |
| M2/M30 | Unchanged policy baseline, action/crop/reload audits | Valid execution and lower loss still failed manipulation |
| Original M3/M4 | Prompt/correction adaptation; explicit controls | No unfinished success; some preservation regression |
| Original M5/M6 | LoRA/head and full joint adaptation on one source | Confounded adaptation comparisons; no completed task |
| V1/V2 | Expanded data/catalogue versions and full training | Distinct supervision/cue conditions, not clean single-factor ablations |
| Actual V3 | Causal sparse history and physical-time anchor windows | Timing checks pass; learned temporal advantage unproved |
| V4 | One-round matching-student expert suffix aggregation | Budget expiry is not proven irrecoverable failure; no PPO/Q-learning |
| Final common pilots | Twelve eight-case suites | All unfinished 0/6 per model; successes are preservation |
| Four-second diagnostic | Four separate eight-case V3 suites | Different horizon; no unfinished completion |
| V4-M6 partial | Grasp, 161.4 mm lift, 246.3 mm transport, stable target release | 85.06° yaw failed; overall task failed |
| Qwen service | Real schema/semantic failures and added completion veto | Assisted diagnostics remain separate from policy scores |
| Small classifier | Frozen synthetic confirmation, explicit unknown | Fixed overhead feeding scope; confidence uncalibrated |
| Typhon harness | Earlier M6 plus real classifier, guarded live cases | Three precompleted placements recognized; source attempt failed |

## Evidence available in the public snapshot

The existing public evidence includes [M0](../evidence/m0/M0_report.txt), [M15](../evidence/m15/M15_report.txt), [M3](../evidence/m3/M3_report.txt), [M4](../evidence/m4/M4_report.txt), [M5](../evidence/m5/M5_report.txt), [M6](../evidence/m6/M6_report.txt), [M7 pilot](../evidence/m7/M7_pilot_report.txt), [M7 follow-up](../evidence/m7/M7_followup_report.txt), [harness report](../evidence/harness/SERVICE_REPORT.txt) and [classifier evidence](../evidence/goal-classifier/). The source index distinguishes these published copies from original report hashes and from final evidence not yet mirrored in that earlier snapshot.

Code for the classifier, service, harness, telemetry and training is retained under the repository’s supervisor/ and scripts/ directories. Reproduction still requires the declared environment, appropriate model weights and data; the small public export is not a claim that every private video, checkpoint or bulk rollout is bundled.

See [SOURCE_INDEX.json](SOURCE_INDEX.json) for provenance and [EVIDENCE_GAPS.json](EVIDENCE_GAPS.json) for unvalidated claims and publication-copy limitations.
