# VIDEO09 — Full-sequence video-to-Panda execution

**The personally recorded correction sequence drove physical Panda manipulation, but the stronger result was not established: all five executions missed the unchanged placement criteria, and ordered video did not show a reliable overall advantage over simpler references.** This is assisted video-conditioned retargeting, not X-VLA training or recovered human motor commands.

Ticket: **#11033**. Parent: VIDEO01 **#11015**. Planner: [Hm challeng p2](https://chatgpt.com/c/6ac8f7df-fef0-83eb-98c3-48bd796980a1). Code branch: `research/video09-sequence-adapter`. Bulk evidence: Mint Fast Drive `Typhon/research-20261012/video09/`; execution set: `landmark-v2/`.

## Question and frozen comparison

Does intermediate, ordered motion from a complete human RGB take add useful robot behavior beyond endpoints and a shared acquisition/release routine?

One previously inspected **demo_0007 / R_train** correction take supplied 228 sequentially decoded native frames per camera. Both raw-video SHA-256 values were verified. Primary video supplied the blue numeral's projected centroid and directed orientation; the paired secondary recording retained event/provenance evidence. No calibrated human 3D, force, or joint-action labels were asserted.

The four conditions were frozen before execution: **SEQ** (ordered take), **END** (smooth endpoint interpolation with shared event assistance), **SHUFFLE** (one fixed permutation of four interior blocks), and **STATIC** (final descriptor during manipulation). An exact **SEQ repeat** followed. These are four conditions on one development take, not five independent statistical trials.

The robot supplied RGB-based acquisition, a downward parallel-jaw grasp, fixed clearance/lowering/release, calibrated robot-camera geometry and native OSC control. The nominal source clock was slowed by four. Acquisition/lift occurred before the source clock and is separately logged. Agent-reviewed event intervals are not represented as new human approvals. Object truth/contact entered only independent scoring. No old geometric-expert trajectory, action arrays, VLA weights or training were used.

The new experiment explicitly used the 50×35×30 mm, 80 g provisional proxy and M1.6 visual asset `duplo_31110pb074_v2`. Historical catalogue-sized M1.5 configurations were left unchanged.

## Physical results

| Condition | Supported grasp/lift | Footprint in square | Centre error | Yaw error | Released and stable | Strict success |
|---|---:|---:|---:|---:|---:|---:|
| SEQ | Yes | No | 35.475 mm | 3.826° | Yes | No |
| END | Yes | No | 34.425 mm | 7.685° | Yes | No |
| SHUFFLE | Yes | No | 33.673 mm | 7.397° | Yes | No |
| STATIC | Yes | No | 35.387 mm | 3.597° | Yes | No |
| SEQ repeat | Yes | No | 35.475 mm | 3.826° | Yes | No |

Every execution completed its horizon. All passed orientation, marked-face-up, placement-height, release and settling checks. All failed 15 mm centring and full footprint containment. No forbidden static contacts or joint-bound violations were logged. SEQ and its repeat have **byte-identical decompressed action and observation traces**.

The batch took 14.1 seconds wall time with three parallel processes on Powerglide. Each simulated execution contained its supplied prefix, the 30.4-second scaled source timeline and settling. Fast simulation wall time is not real-robot throughput.

## What the audit established

1. **The endpoint registration conflicts with the task.** Under this declared proxy and robot-camera model, exact matching of the final human glyph descriptor would place the robot block centre at [0.121558, −0.093118] m, **27.101 mm from the unchanged goal** [0.125, −0.12] m. Thus perfect visual-reference following would already violate the 15 mm centring requirement. This analytic diagnostic does not measure the physical phone scene. The goal was not moved.

2. **The initial feature contract also disagrees.** Initial block reconstruction used the visible orange centroid, whereas the revised controller followed the blue glyph. The first simulated and human projected glyph locations differ by **0.1896 target units**. The initial “human stillness” can therefore contain robot corrective motion. This limits any claim about faithfully reproduced pauses.

3. **Post-release observation scoring is invalid.** All valid runtime pose observations used secondary-view rectification because the overhead view was occluded. The fallback retained an EEF-relative depth plane after release, although the block rested on the table. Raising the empty gripper then moved the reported glyph projection by several target units. This was logged as observable. It did **not** change post-release XY/yaw commands, because that servo was inactive, and it does not change the independent physical outcomes. It invalidates an overall visual-position benefit claim from those observer metrics.

4. **Sequence order changed motion, without a universal benefit.** Independent scorer-only projection gives the following whole-timeline errors. These diagnostics never fed the actor:

| Condition | Mean projected position error, target units | Mean directed error |
|---|---:|---:|
| SEQ | 0.0987 | 4.73° |
| END | 0.0928 | 6.74° |
| SHUFFLE | 0.1046 | 13.78° |
| STATIC | 0.0923 | 3.58° |

SEQ's rotation followed the original ordering better than the shuffled control. STATIC performed better than SEQ on these whole-timeline means. This does not establish that intermediate frames are generally unnecessary; it rejects the stronger advantage claim for this take and adapter.

The raw observer medians remain saved: SEQ position 0.04794 units and direction 4.195°. Its position median improved 17.9% over END and 29.6% over STATIC, missing the proposed 20%-over-both gate even before the observation flaw. **Do not advertise these as validated gains.** `metric_validity_audit.json` explicitly limits their use.

All 228 native references are archived; feedback sampled 227. Only the last stationary-tail frame was missed by feedback cadence. No manipulation interval was truncated. No unknown source observations were interpolated. Reported 100% source detection coverage is not validated measurement accuracy.

## Qualification and revisions

The first qualification grasped and lifted successfully; X/Y response passed, but a 5.14° physical turn produced no template-angle change. It also exposed partial-orange-centroid bias. Before any comparison outcomes, the reference changed to the matched printed glyph and the qualification turn increased to 0.30 rad. Both versions and the failed qualification remain archived.

The second local qualification passed: X/Y response ratios 1.045/1.026 and yaw ratio 0.904, correct response directions, independently verified lift 152.5 mm. Analytic projection derivatives matched numerical derivatives within 3.1×10⁻⁶. Missing-observation timeout, forbidden actor channels and shuffle boundaries were checked. This local response check did not qualify post-release support assumptions, as the later audit demonstrates.

## Decision and evidence

**Close this experiment as mixed/negative.** Real intermediate video observations can participate in actual robot control with explicit embodiment assistance. The experiment does not show strict task completion, calibrated human motion recovery, useful training action labels, generic retargeting, X-VLA numeral understanding or a robust advantage over static goals.

Keep these generated trajectories out of policy training. The next experiment should resolve the feature/endpoint contract and released-object support assumption before scaling data or changing model weights. A shared goal-constrained terminal correction is a separate intervention and would need its own control.

Preserved artifacts include original-source provenance, native frame clocks/masks, both qualifications, all five raw executions, independent scores, four synchronized human-versus-Panda videos, visual contact sheets, exact-repeat checks and checksum manifests. M0/M1/M1.5 and previous experiment outputs remain unchanged.

- [Reproduction commands](commands.txt)
- [Frozen protocol](../../configs/experiments/video09/protocol.json)
- [Comparison table](comparison.csv)
- [Metric validity audit](metric_validity_audit.json)
- [Independent validation](independent_validation.json)
- [Representative comparison sheet](SEQ-contact-sheet.jpg)

Main videos on Mint:
`/media/nm/Fast Drive/Typhon/research-20261012/video09/landmark-v2/comparison_videos/{SEQ,END,SHUFFLE,STATIC}-human-vs-Panda.mp4`.
