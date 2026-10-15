# EVAL02-RET-C: the matched visibility control has a different error

LOAD04's correlated-visibility C retains the original feeding goal (2/2) and succeeds on one of two relocated-goal trials. Its other relocated trial overshoots the centre by 23.70 mm while remaining inside the square. B's earlier matched trials instead returned near the old destination and failed both relocated trials. This is a mixed result, not evidence that both continuations forgot the goal in the same way.

Ticket #11064. No training, new scenes, holdouts or controller changes. Four new physical rollouts and two new saved-input proposals; prior GOAL/FL/B outputs are archived references, not rerun controls.

## Matched training qualification

C is LOAD04 correlated-visibility step3000, SHA256 `6fbdaeaae3ff3510a4288cac1ad4e7f73989bd9ab948ed343d30c2b615426f94`. It is not M5's prompt-only C. C and B have the same exact LOAD01-FL parent, update budget, optimizer settings, parameter counts and training code. Their configurations differ only in variant. All 12,000 sampled windows and per-step underlying trajectory/action indices match after removing visible/hidden suffixes. The actual logs match their frozen plans. C correlates fixture visibility with task; B counterbalances visibility. Neither latest continuation rehearses relocated feeding goals G1/G2. See `lineage-qualification.json`.

## Physical results

| Goal | Policy seed | Strict success | Position error mm | Yaw error deg | Final XY m |
|---|---:|---|---:|---:|---|
| G0 | 912171 | Yes | 6.11 | 0.70 | 0.12629, -0.12597 |
| G0 | 912172 | Yes | 6.01 | 0.80 | 0.12713, -0.12562 |
| G1 | 912171 | Yes | 8.65 | 0.92 | 0.12985, 0.03717 |
| G1 | 912172 | No | 23.70 | 1.34 | 0.13040, 0.05308 |

G0 is [0.125,-0.120] m; G1 is [0.125,+0.030] m. All four acquire a contact-supported lift and transport the block. The failed C trial violates only the 15 mm centre tolerance; footprint, yaw, face-up, placement height, release and stability criteria all pass. The final contact sheet was visually inspected. Whole-context Y gains are 1.088 and 1.191: both image and numeric goal change here, so this is not isolated numerical steering.

Original EVAL02-RET configurations, native interfaces, 48 queries and 1,460 executed records per episode are unchanged. Initial available simulator/controller/random state matches archived RET byte-for-byte; initial images and effective proprioception match the original training archives. Actual generated noise and CPU/CUDA RNG arrays match the archived RET seed references at all 48 queries. These checks do not assert cross-session identity of all subsequent frames. Every strict score was independently recomputed. Four native videos and one side-by-side goal comparison have 367 decodable frames each.

## Same-input post-lift predictions

C receives the identical FIT frame35/action140/tick180 packet and its two noise draws. All actual prepared inputs, including text, and generated noise arrays match FIT exactly.

| Seed | Mean XYZ error mm | Final XYZ error mm | Final predicted EEF Y m |
|---|---:|---:|---:|
| 912171 | 25.86 | 35.17 | 0.064976 |
| 912172 | 25.94 | 35.08 | 0.064976 |

The expert's final EEF Y is 0.030 m. C predicts past that Y; B predicted -0.00818/-0.00901 m on the same inputs. This differs from B's direction of bias. Conditional prediction error is not final object-placement error and does not explain the full rollout quantitatively. The post-lift window has constant closed-gripper labels, so it does not test release timing.

## Decision and limitations

The predefined mixed-result branch selects GOAL-varied versus B for EVAL02-NUM (#11065). The next test changes only numeric coordinates in this fixed image/proprioception packet. No additional C seeds or scenes are selected.

Absent rehearsal alone does not distinguish C and B. The matched visibility intervention is associated with different local outcomes, but two noise draws from one training family cannot establish a general mechanism or success rate. This is a training-family retention diagnosis, not held-out generalisation. The existing engineering block proxy is unchanged and is not a claim of measured physical DUPLO geometry.

## Reproduction and artifacts

Branch `research/eval02-matched-c-retention`; preparation freeze `381ddace25072c5ccfd7b09095ba0cdc3961ab88`. Use the qualified Powerglide environments and checkpoint paths in `protocol.json`. Use a fresh output directory and matching archived initial packets; do not overwrite sealed outputs.

```bash
.venv-sim/bin/python scripts/eval02_ret_c_qualify.py
.venv-sim/bin/python scripts/eval02_ret_c.py freeze
.venv-sim/bin/python scripts/eval02_ret_c.py physical
.venv-sim/bin/python scripts/eval02_ret_c.py proposals
.venv-sim/bin/python scripts/eval02_ret_c_analyse.py
```

The physical runner is reused unchanged from RET; its internal legacy provenance tag still says EVAL02-RETENTION. The distinct output root, protocol, ticket, model SHA and source hashes identify this C extension. The standalone C analyzer uses the dated default root; change its output constant for a new reproduction root.

Raw package: `/var/tmp/typhon-eval02-ret-c-20261015`. Verified archival copy: `/media/nm/Fast Drive/Typhon/research-20261015/eval02-matched-c-retention`. Reports/source are committed privately; model weights and bulky traces are not put in Git. This ticket closes a bounded diagnosis; the wider research programme stays active.
