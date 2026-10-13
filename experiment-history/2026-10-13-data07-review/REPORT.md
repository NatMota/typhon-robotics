# DATA07 — What the models were actually taught

Completed 13 October 2026. This is an existing-evidence review, not another training run. Frozen datasets, splits, checkpoints and evaluation results were not changed.

**Conclusion:** we have real recovery training, but its coverage is narrow. The later recovery model received four qualified correction trajectories from two training families. Its only already-held start was nearly nominal; the difficult off-axis repair remained a quarantined diagnostic. This supports a coverage gap, but does not establish that more data alone fixes the model, or certify the contents of V4's unavailable original recovery arrays.

## 1. Keep the datasets separate

| Dataset / run | Underlying generated episodes | What was supplied by the human recordings | Robot labels and exposure |
|---|---:|---|---|
| V1–V3 shared catalog | 18: 6 P_train, 2 R_train, 6 S_train, 4 C_train | Personal scene endpoints/category; none for synthetic cases | Generated expert robot commands; the four completed cases use generated open-gripper hold commands |
| Broad Pt, 12,000 updates | The same six P_train episodes inside those 18 | Same personal scene information | 8,000 logged draws per take; 48,000 total. These are not six additional demonstrations |
| Original V4 | 18 inherited + 3 declared recovery additions | Personal parent scenes where applicable | V3 checkpoint followed by 500 joint updates. Original three recovery arrays and per-episode draw evidence remain unavailable in the inspected records |
| Later recovery continuation | Six nominal parent episodes + four new correction suffixes | Parent scenes; failures are learner-generated robot states | Broad Pt12k followed by 3,000 joint updates. Four corrections actually sampled 2,000 times in 12,000 draws |
| Older recovery oracle | 12 suffixes: six diagnostic starts × two controller variants | Parent scene only | Quarantined; never admitted by this review |
| Human footage | 16 curated takes, plus excluded demo_0017 retained separately | Actual observed manipulation and static goal evidence | No validated phone-frame-to-native-robot-action alignment |

All 18 current catalog NPZ files match their frozen hashes. Their original robot-action files and episode summaries also match their recorded hashes; all 18 saved physical outcomes were successful. That verifies the saved teacher material, not learned-policy competence.

The broad six match the catalog **byte for byte**. The later six goal0 representations are recorded under their existing underlying parent episodes, rather than counted again. Four C_train episodes were explicitly admitted by a separate historical protocol: two personal completed scenes and two independent synthetic completed scenes. Static human footage remains observation evidence; the robot hold commands were generated separately.

The review contains **38 generated-episode/correction rows**: 22 verified existing training items, 12 quarantined diagnostics, three missing-evidence V4 additions and one rejected correction state. The rejected state's duplicate alias is retained without counting it twice. This is a scoped ledger of the requested training/recovery lineages, not an inventory of every historical simulator attempt.

## 2. Does recovery training include off-axis acquired grasps?

For the **later four corrections**, only one starts with an acquired bilateral grasp:

| Admitted correction | State at handoff | Saved teacher outcome | Actual training draws |
|---|---|---|---:|
| 0005-last_supported | Brick already held near goal, nearly nominal wrist/object yaw relationship | Strict pass | 372 |
| 0005-final_failed | Open gripper, no brick contact, failed learner rollout ended | Strict pass | 628 |
| missed0003-7417 | Open gripper, no contact; learner failure on source0003 yaw+15° development variant | Strict pass | 505 |
| missed0003-7418 | Same source family/variation, separate learner seed and trajectory | Strict pass | 495 |

Each of these four has a saved exact-prefix replay qualification, a verified source correction NPZ, a verified converted training NPZ, action conversion error below 8.75e-7 and positive joint margin. The policy history resets at intervention. Targets are privileged expert commands, not human actions.

Using **saved simulator telemetry**, the inherited grasp's planar object-to-nominal-wrist residual is **0.07493°**. The quarantined diagnostic named offaxis_held has a residual of **−84.13318°** at handoff. The formula and frame convention are in the ledger and reproduction script: nominal wrist rotation is Rz(yaw) multiplied by the controller's fixed downward-facing basis. This is a planar relationship diagnostic, not a visual orientation estimate; it should not be applied as a full 3D grasp-quality score.

The successful off-axis expert suffix exists and is hash-verified, but remains excluded because it comes from a quarantined diagnostic state. Another diagnostic held state at approximately −24.13° failed under both tested experts. Side-face diagnostics also failed. Thus the qualified later training set does **not** demonstrate diverse difficult held-grasp coverage.

The near-nominal held-start suffix received 372/12,000 draws, or **3.1%** of this continuation's sampled examples. These are draws from that suffix, not necessarily 372 held-state windows. This review audited episode starts; it did not exhaustively measure every later window's grasp relationship. Therefore it establishes narrow verified start-state coverage, not the absence of every other relationship anywhere in the trajectories. The four correction suffixes together account for 16.67% of presentations.

Missing V4 training arrays do not erase its saved checkpoint behavior. They prevent fully reproducing or attributing its correction-data recipe.

For **original V4**, the answer remains unknown: its three declared recovery episode IDs are recorded, but their arrays and physical qualification were not recovered in this review. Do not substitute the later four corrections when describing V4.

## 3. Does it include learner approach or empty-close failures?

**Partially.** The two source0003 correction suffixes begin after failed learner rollouts, with an unheld brick and open gripper; the source0005 final-failure suffix also begins unheld. Their teacher continuations provide successful reapproach/grasp/placement targets from those states.

This is more than nominal expert-only training. It is still only three unheld starts from two families, including two seeds of the same yaw-perturbed source0003 family. It does not establish coverage of arbitrary bad approach poses, the exact instant of an empty close, repeated failed closes, loss of contact while carrying, or broad off-axis acquired grasps. A source's category name alone is not evidence of those states.

The later continuation's 3,000 logged updates are contiguous; all losses checked are finite. Its 12,000 sample IDs match the saved sampling plan exactly, and all 660 existing window IDs were visited. **10,000 draws were nominal, 2,000 were correction draws.** The source-family schedule was balanced; the number of distinct failure states was not enlarged by that balance.

Its existing quarantined recovery evaluation was **0/4 strict successes**. No evaluation was rerun here. This does not prove why it failed, and is not a V4 result.

## 4. What has observable inputs and qualified targets?

| Decision / evidence | Inputs and targets we actually have | Boundary |
|---|---|---|
| Nominal approach, close, lift, transport, place, release | Paired simulator RGB, robot command-history input, supplied goal, physically successful expert command trajectories | Privileged geometric teacher; no native human robot-action labels |
| Preserve an already completed state | Four physically successful simulated hold episodes, with linked personal/synthetic provenance | Not a robot action trajectory recovered from static phone footage |
| Four later admitted repairs | Learner start state, exact replay receipt, simulated RGB, converted native action-history input and successful expert correction targets | Only two source families; one nearly nominal acquired grasp |
| Off-axis diagnostic repair | Saved physically successful expert suffix and paired robot inputs | Diagnostic-only; action-history convention differs from M3; cannot silently reuse it |
| Original V4's three recovery additions | Episode IDs, catalog/checkpoint hashes and training configuration | Missing arrays and detailed qualification; coverage cannot be certified |
| Human pickup/release, pause-resume, correction and completed-goal footage | Original recording identity/hash, paired reviewed views, existing category/uncertainty and observed motion evidence | No qualified causal per-frame grasp state or calibrated human-to-Panda command alignment |

Human lineage is preserved in human_observation_ledger.json, including recording IDs, source hashes, camera transforms, reviewed timestamps, existing split and exclusion reasons. No train/validation admission was changed. demo_0017 remains excluded.

V1/V2 training evidence has saved per-episode aggregate draw counts, but not per-update sample IDs. V3/V4 checkpoint metadata establishes declared membership and training configuration; it does not establish exact per-episode exposure. Those fields remain null.

## 5. Decision

Do not launch another unchanged full training run on the assumption that recovery coverage is now broad. Keep the existing four-correction model as a completed narrow experiment. Before a new correction-training claim, qualify new training-family learner states that cover the missing grasp relationships and failure phases, while leaving the existing diagnostic and validation states excluded.

In parallel, retain the previous VIDEO01 conclusion: the representation-to-action link is unqualified. This audit does not establish that the model cannot see the numeral, or that perception is the sole bottleneck. Better visibility, action calibration, grasp-aware control and appropriate supervision remain distinct hypotheses.

Original V4 source-array recovery is a provenance gap, not permission to regenerate the data and pretend it is the original dataset.

## Reproduction and evidence

On Powerglide, run scripts/data07_collect_powerglide.py with its standard Python. It reads existing artifacts and writes only /var/tmp/typhon-data07-existing-audit. Copy that small evidence directory to the review's inputs folder on Mint, then run:

```bash
python3 scripts/data07_coverage_review.py \
  --root "/media/nm/Fast Drive/Typhon" \
  --output "/media/nm/Fast Drive/Typhon/research-20261012/data07-coverage-review"
```

No NumPy, simulator, model inference or package installation is required for this review. The collector hashes existing arrays; the audit inspects NPZ headers and saved logs without creating labels.

- episode_ledger.json / .csv: one row per requested underlying generated episode or correction suffix.
- human_observation_ledger.json: separate human lineage, 16 curated + one excluded.
- source_overlap.json: exact six-of-18 overlap.
- file_verification.json and input_manifest.json: hashes and traceable input files.
- archive_search.json: scoped lookup of the original V4 additions in four saved pod inventories; absence here is not proof of absence everywhere.
- summary.json: actual counts and sampling checks.
- inputs/: copied source manifests and collector output.
- protocol.json: frozen 45-minute review scope, start and deadline.
