# VIDEO04 — Training-only hand-present negatives

Adding twelve reviewed negative frame-views improves the observation probe's withdrawal handling, but does not resolve false grasp recognition during approach. The extra transition panel still has one failing physical event out of three. No robot-control signal is qualified.

## What changed

Six linear observation heads were fitted on frozen X-VLA-Pt phone-image descriptors: three representations, each with two training-weight choices. The original three VIDEO03 heads remain unchanged as baselines.

This is direct learning from personal video and visual annotations. It is not motor-policy training, a geometric-expert action dataset, a new VLA checkpoint, or a Panda execution. No backbone, action head, soft prompt, simulator, controller or frozen experiment output was modified.

The experiment is OST #11021, under VIDEO01 #11015. VIDEO03 #11020 remains the preceding immutable experiment.

## Frozen selection and labels

- Protocol and score-blind selection committed at `6b4b922`.
- Agent visual labels committed before fitting or viewing individual scores at `0af1082d8ed7f22b9f56e91753b46c1be878b01f`.
- Fitting implementation committed before execution at `4571e8d`.
- Reused all 2,644 verified unmarked dense-v2 frame descriptors, source timestamps and original source hashes.
- Training takes: demo_0001, 0002, 0003, 0005, 0012 and 0013. All views from other takes remain excluded.
- Existing validation: demo_0006, 0014, 0016. Its previously inspected transition challenge is reported as a reused diagnostic.
- Extra event evaluation: demo_0004 and 0015, excluded from fitting. These share the known recording sessions, and their conservative interiors already had automatic VIDEO03 scores. They are not new sessions or an independent confirmatory dataset.

For each of the eight training/extra takes, select the earliest hand-detected ambiguous approach and latest hand-detected ambiguous withdrawal between the old reviewed intervals. There were fourteen eligible events; demo_0003 and 0004 had no eligible approach frame. Selection never uses a model score.

The coding agent reviewed four saved contact sheets with paired views at -0.3, 0 and +0.3 seconds. Labels apply only to the central frames. Future context assists offline annotation and is never a predictor input. This is agent review, not new human confirmation or contact-sensor ground truth.

Six training withdrawal events have a clear gap between hand and resting block: twelve negative camera views. All five selected training approaches remain unknown because contact/support cannot be resolved confidently. The three extra events are clearly not held: demo_0004 withdrawal, demo_0015 approach and withdrawal, six views.

**Training coverage therefore remains asymmetric: all new negatives are withdrawals.** No approach, failed grasp, light-contact or supported-pause label was invented. The training set grows from 451 to 463 known frame-views; ten reviewed training approach views remain unscored.

## Models and controls

Three representations are compared:

1. Current full image plus current object crop.
2. Those image features plus observed hand/object features.
3. The same inputs plus a causal 0.5-second feature history.

The two new fitting conditions either retain VIDEO03's equal take/class/view weighting or allocate half of each negative take/class/view weight to the new transitions and half to old negative interiors. The twelve new negatives receive 1.91% and 25.0% of total fitting weight respectively. The second arm is explicitly a sampling intervention.

All conditions use the same fixed sigmoid threshold 0.5, L2 0.01, training-only standardisation, LBFGS limit 200 and split. No threshold or regularisation was selected on the evaluation panels. The scores are uncalibrated classifier outputs.

The earlier four-second intention windows are unchanged. A 0.5-second contact history is an observation diagnostic, not an action horizon.

## Results

The discrete results below hold for **each of the three feature representations**; all nine model/condition combinations are retained in metrics.json.

| Fitting condition | Held interiors found / 90 | False held on negative interiors / 155 | False held, extra transitions / 6 views | False held, reused challenge / 8 views |
|---|---:|---:|---:|---:|
| Frozen VIDEO03 baseline | 90 | 0 | 2 | 2 |
| Add negatives, original weighting | 90 | 0 | 1 | 1 |
| Add negatives, balanced negative regimes | 90 | 0 | 1 | 1 |

The 90 labelled positives consist of 56 old-validation and 34 extra-take frame-views. The 155 negatives consist of 99 and 56 respectively. Another 541 interior-panel frame-views remain unknown and are excluded from accuracy.

These are correlated frames and cameras from five takes, not 245 independent demonstrations. Perfect agreement on conservative interiors does not establish physical grasp sensing.

### Count physical events as well as camera views

| Panel | Physical negative events | Baseline events with any false-held view | New heads |
|---|---:|---:|---:|
| Extra transition panel | 3 | 1 | 1 |
| Reused VIDEO03 challenge | 4 | 2 | 1 |

The extra-panel view count improves 2 to 1, but it is the **same unresolved approach event**. Calling this a 50% improvement in physical-event recognition would be misleading. The any-view summary is an error audit, not a deployed two-camera fusion rule.

The old challenge's withdrawal error disappears, leaving its approach error. More weighting and the history features provide no additional discrete benefit on these small panels.

### Concrete remaining failure

On demo_0015's approach, the hand is visibly separated from the block. For the current full+crop head:

| View | Frozen held score | New score, original weighting |
|---|---:|---:|
| Primary | 0.9893 | 0.9497 |
| Secondary | 0.7906 | 0.2111 |

The secondary view becomes correctly not held, while the primary remains confidently wrong. The reused demo_0006 approach remains wrong in the primary view too (0.9331 to 0.9133). Its secondary withdrawal error is corrected (0.6490 to 0.0256).

All per-event scores, including unknown labels, are retained. The report does not select only improved views.

## Decision

**Proceed with targeted observation-label repair, not deployment or another large motor-training run.** Personal video supplies useful supervision to an observation head, and six clear negative events repair a withdrawal failure without losing the available positive interiors. They do not establish reliable contact or approach recognition.

The next useful additions are training-only pre-contact frames with visible gaps, held pauses, light contact and failed grasps, followed by separate event evaluation. Those require explicit visual review or better evidence; unknown frames cannot become negatives by assumption. Validate onset/release timing and missed-held cases as well as false positives.

This experiment does not justify a new camera-fusion guard. Metric depth, physical calibration, the generic Panda action adapter, correction-positive labels and simulator-side observer qualification remain unresolved under #11015, #10989, #10991 and #10984.

No task B/loading result or robot success rate changed.

## Validation and reproduction

- Every source-frame JPEG hash was checked again against dense-v2.
- All twelve changed labels are previously unknown rows from P_train; no evaluation label enters fitting.
- All six new heads reload independently in NumPy within 1.20e-6 of the CUDA scores.
- The three frozen baselines reproduce their earlier saved probabilities within 5.37e-7.
- A fresh six-head refit reproduces all twelve saved prediction/label arrays bit-for-bit.
- Training-only scalers, complete-take exclusion, known/unknown handling, fitted indices and weight sums pass.
- The original fits took 2.89 seconds using precomputed image descriptors; this excludes VIDEO02 extraction/encoding and is not a motor-policy training-time claim.

On Powerglide, from branch `research/video04-contact-negatives`, use a new output root:

~~~sh
VIDEO_PYTHON="$HOME/Projects/typhon-harness-20261010/.venv-xvla/bin/python"
VIDEO_OUTPUT=/var/tmp/video04-reproduction
"$VIDEO_PYTHON" scripts/video04_fit.py \
  --protocol configs/experiments/video04/protocol.json \
  --output "$VIDEO_OUTPUT/fit-v1"
"$VIDEO_PYTHON" scripts/video04_fit.py \
  --protocol configs/experiments/video04/protocol.json \
  --output "$VIDEO_OUTPUT/fit-reproduction"
"$VIDEO_PYTHON" scripts/video04_verify.py --root "$VIDEO_OUTPUT"
~~~

The committed selection and labels are fixed fitting/evaluation inputs. The score-blind contact sheets can be regenerated separately with scripts/video04_select.py and a fresh --output directory. Do not relabel them from prediction scores.

Restore dense-v2 at the protocol's input_data path, or change only that location in a copied protocol and record the path change. The frozen VIDEO03 heads and reviewed source manifests are versioned in the preceding experiment directory. Bulk raw recordings remain in their existing Mint archive.

Outputs: `/var/tmp/typhon-video-contact-20261012`. Private Git contains scripts, protocol, labels, review sheets, heads, scores, validation and this report. The complete experiment directory is copied to Mint Fast Drive with a SHA-256 manifest and a separate verification receipt.
