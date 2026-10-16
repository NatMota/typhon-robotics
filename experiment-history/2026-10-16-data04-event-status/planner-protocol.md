## EXECUTE NEXT: DATA04-EVENT — does the existing video representation align human and robot task phases?

**Park the return-teacher geometry branch.** RETURN02-Q rejects the fixed candidate at the sampled nominal configurations; it does not establish that all return motions are impossible. No further offset, aperture or wrist search follows.

The next useful question returns to Nat’s personal footage:

> **Can the existing frozen visual representation locate corresponding manipulation events in human and Panda videos better than simply matching their relative elapsed times?**

This advances the unresolved human-video/robot alignment dependency **without calibrated human 3D, new motor labels, another controller or VLA training**. Earlier cross-view time matching and take retrieval did not establish this cross-embodiment correspondence.

**One 75-minute experiment: three existing take pairs, three fixed alignment methods, zero physical rollouts and zero optimizer updates.**

## 1. Use three training-source pairs—not another correction take

Use:

| Take | Role |
|---|---|
| **0001** | Ordinary feeding |
| **0002** | Second ordinary feeding example |
| **0012** | Awkward/pause-resume example |

For each, pair the **complete original human take** with its corresponding **original verified canonical Panda feeding trajectory** from the audited catalogue.

Do not use VIDEO10’s goal-registered execution or display-time-warped comparison video. Those already contain supplied temporal/geometric adaptation and would contaminate this test.

All three are P_train sources and historically inspected. **No new generalization claim, no P_validation examples, no demo0017, and no loading demonstrations are implied.** The robot actions remain expert-generated; the question is whether the images support meaningful temporal correspondence.

Use the primary human view in its verified **90° counterclockwise orientation**, and the corresponding primary robot view for the scored alignments. The secondary views may resolve annotation uncertainty, but do not silently add them to the descriptor.

## 2. Freeze one existing representation

Use the **same frozen full-image-plus-RGB-crop feature pipeline already evaluated in VIDEO03–07**, with its actual encoder, preprocessing and fusion settings pinned from the saved manifest.

Before feature work:

- Verify the resolved encoder/extractor identities and existing feature provenance.
- Reuse a stored feature only when its exact source frame and extraction contract match.
- Use raw robot images without the simulator-derived held crosshair.
- Keep crop failures/missing observations explicit.
- Do not use simulator pose, contact, phase, action commands, elapsed-time tokens or episode IDs as feature inputs.

If robot features need extraction, run the **same frozen encoder** on the archived images. No new weights, projection head, fine-tuning or normalization fitted to these three pairs.

Time-Contrastive Networks explicitly distinguish viewpoint invariance from recognizing visually similar but functionally different states. Temporal Cycle-Consistency work evaluates action-phase information in addition to alignment. Those are useful distinctions here—not methods we can claim to reproduce with an unchanged pretrained descriptor. [arXiv](https://arxiv.org/abs/1704.06888?utm_source=chatgpt.com)

### Resource gate

Process at most **128 actual frames per stream**, selected uniformly by timestamp over the complete clip, including endpoints: **six scored streams, at most768 feature vectors**.

Use sequential decoding or qualified saved frames, not the superseded seek-based exporter. Retain actual selected PTS; do not pretend the two embodiments have synchronized clocks.

With limited scratch, stream small batches from verified archives and save only features, frame references and selected evidence images. **No full video-bank restore or checkpoint duplication.**

If the existing extractor or unannotated robot images cannot be recovered within **15 minutes**, record the specific dependency and stop. Do not substitute another encoder or annotated render.

## 3. Freeze semantic event intervals before examining alignment outputs

Use existing reviewed intervals where available. Otherwise inspect only enough of the paired original views to bracket these three events in each human and robot sequence:

| Event | Operational meaning |
|---|---|
| **Lift/held-transfer onset** | First interval where the block is visibly carried off its resting support |
| **Arrival over the destination** | First interval where the carried block reaches the destination before final placement/release |
| **Release** | Transition from visibly carried/controlled object to the object remaining while the hand/gripper withdraws |

**Maximum18 event intervals: three events × six sequences.** Cap this review at **20 minutes**. Labels must be saved before displaying feature distances or alignment paths.

Important boundaries:

- Human labels are **agent-reviewed visual intervals**, not measured contact times, forces or metric lift.
- Use both phone views to bracket uncertainty, preserving the approximate synchronization error.
- Robot contact/height logs may corroborate the robot event offline; they cannot enter the aligner.
- If an event is occluded or ambiguous, mark it unknown. Do not assign a precise midpoint merely to make the score complete.
- Do not turn the pause in0012 into a required robot pause. Its differing timing is useful for testing alignment, not a new task requirement.

Require at least **two independently identifiable interior events in each pair**. If a pair fails that gate, retain it as unscorable; do not replace the take.

## 4. Run exactly three alignments per pair

| Method | Information used | Purpose |
|---|---|---|
| **TIME** | Start/end timestamps only; linear normalized-time mapping | Tests what endpoints and elapsed progress already provide |
| **VISUAL** | Frozen descriptor distances with monotone dynamic time warping | Proposed cross-embodiment alignment |
| **SHUFFLED** | Same descriptors, with one fixed permutation of interior human temporal blocks | Tests whether the visual sequence’s actual ordering contributes beyond the aligner’s constraints |

That is **nine alignment outputs**, not a hyperparameter search.

### Fixed alignment rules

- Use cosine distance between the existing normalized descriptors.
- Use the standard monotone DTW path with permitted diagonal/horizontal/vertical moves.
- Fix start/end correspondence for all methods.
- Add **no event anchors, learned phase labels or fitted timing penalty**.
- For SHUFFLED, keep the endpoint frames fixed and permute the interior human descriptors in **one-second blocks**, using seed **161006**, frozen before scoring.
- Do not let the aligner recover the original order from filenames or timestamps attached to shuffled features.
- Report path compression/stretching and how much of either sequence maps to a small interval.

A monotone path is ordered by construction. **Its lack of backwards steps is not evidence that it discovered manipulation semantics.**

Keep all complete-take frames in the inputs. Event labels serve evaluation only; they do not segment or steer the alignment.

## 5. Evaluate against event intervals—not the alignment’s own objective

For each known human event interval:

1. Map its midpoint through the alignment, using a fixed median rule where a source frame maps to multiple robot frames.
2. Measure distance to the independently labelled robot event interval: zero when inside it, otherwise distance to its nearest boundary.
3. Report the mapped interval span as well, so a broad or collapsed alignment cannot hide behind one midpoint.
4. Report error both in **robot seconds** and as a fraction of robot-clip duration.

Use the reverse mapping as a diagnostic where defined, but do not present forward/reverse agreement alone as independent correctness.

| Required result | Why |
|---|---|
| Per-event errors for all three methods | Prevents one easy endpoint-like event dominating |
| Per-take median and worst interior-event error | Exposes local timing failures |
| Known/unknown event counts and label widths | Makes uncertainty and coverage visible |
| Path stretch/collapse statistics | Reveals degenerate correspondences |
| Actual frame pairs at each mapped event | Allows source-grounded visual inspection |

**Do not use lower DTW cost as the principal success criterion.** VISUAL is optimized for that cost; its advantage there would be largely built in.

## 6. Frozen acceptance and failure branches

Treat these as engineering screens, not statistical claims:

| Outcome | Decision |
|---|---|
| VISUAL improves median event-interval error over both TIME and SHUFFLED in at least two of the three takes, with no take’s worst error worsening by more than5% of clip duration | Evidence that the current representation supports useful cross-embodiment temporal correspondence on this small panel |
| TIME performs similarly or better | No demonstrated benefit over elapsed-progress matching; do not promote the representation to action alignment |
| SHUFFLED performs similarly | The result may be driven by endpoint/path constraints or static appearance rather than meaningful temporal order |
| Alignment looks plausible but event errors are poor | Visual presentation is not sufficient; phase correspondence remains unqualified |
| Event intervals are too ambiguous or sparse | Evaluation-label limitation, not proof that the encoder lacks information |
| The feature pipeline cannot be reproduced | Concrete representation-provenance blocker; no substitute-model experiment |

A zero-error TIME baseline leaves no room to claim improvement for those events. Do not change the event definitions or choose another take to create headroom.

**Even a pass does not produce robot-action labels from human video.** Matching a human frame to a robot frame establishes a candidate correspondence, not that the recorded Panda command is the uniquely correct action for the human observation. Published cross-embodiment methods add mapping and embodiment assumptions beyond temporal correspondence; those remain separate work here. [arXiv](https://arxiv.org/abs/2509.10952?utm_source=chatgpt.com)

## 7. Deliverable, split protection and stop

**Clock:75 minutes.**

| Boundary | Requirement |
|---|---|
| **T+15** | Source pairs, representation and storage qualified |
| **T+35** | Independent event intervals frozen |
| **T+55** | Nine alignments and scoring complete |
| **T+75** | Private report, small evidence package and verified Mint archive |

Deliver one table, nine saved paths, source/feature hashes, the event-interval ledger and a small set of mapped-frame evidence sheets.

No altered trajectories, training admission, VLA calls, simulator runs, metric human poses or new controller authority. Keep all original banks and holdouts unchanged.

**Proceed with DATA04-EVENT after RETURN02-Q closes.** This returns the programme to the personal-video contribution: determine whether the representation can recover *when corresponding manipulation events occur*, rather than continuing to engineer an increasingly privileged return controller.

A positive result would justify a narrowly defined representation-to-action bridge later. A negative result would tell us that the currently improved retrieval/alignment metrics have not yet supplied the semantic timing needed for that bridge. Neither outcome requires another round of geometry tuning.
