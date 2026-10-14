# LOAD05-R — Deterministic-prefix qualification

**Stopped at the independent-prefix gate. Both native diagonals reproduce; no crossed model action was generated.**

Executed commit: `3ddcc6c5840ea7aed6b7df134dc0a0fa717b5604`. Protocol and semantic field manifest were frozen before simulator execution, within the ten-minute setup allowance. All runs use the unchanged archived demo_0012 visible-fixture loading case, seed 912171, query boundary tick100, H30 and C/B checkpoints from LOAD04.

## Observed results

| Prefix → policy | Post-boundary steps | First contact | Bilateral contact | Supported lift | Tilt at lift | Maximum tilt |
|---|---:|---:|---:|---:|---:|---:|
| C → C | 90 | 131 | 135 | 147 | 1.57° | 2.71° |
| B → B | 90 | 107 | 129 | 144 | 62.76° | 72.70° |
| C → B | 0 | — | — | — | — | — |
| B → C | Not started | — | — | — | — | — |

Both native branches exactly reproduced saved model inputs, noise and raw predictions, and reproduced all recorded dynamic/controller traces under the existing tolerances. Six model queries ran in total. These are diagnostic repetitions of the original two behaviors, not new independent evidence of task success. No final placement score was computed.

The third independent C-prefix reconstruction reached tick100. Its semantic comparison differed in **exactly one field**:

```
/xml/children/4/attributes/meansize
original C diagonal: 0.24438
new C prefix:       0.244345
```

All retained physical arrays, active constraints/contacts, controller context, actor history, RNG and derived camera/kinematic fields agreed exactly. The mismatch was recorded before the assertion stopped execution; no crossed query ran. Raw snapshots, full XML, both diagonal videos, predictions, noise/input audits and incremental traces are preserved.

## Interpretation and next decision

This establishes that deterministic prefix reconstruction can reproduce both original native action/trajectory sequences. It does not yet establish a crossed-policy outcome.

MuJoCo documents `statistic.meansize` as the unit used to scale visualization decorations, without other side effects. The earlier LOAD04 audit separately identified this attribute and preserved its raw values. LOAD05-R intentionally retained it in its stricter frozen XML comparison, so the attempt stopped rather than silently changing that rule. [Pinned MuJoCo 2.3.7 reference](https://mujoco.readthedocs.io/en/2.3.7/XMLreference.html#statistic)

The planner has been asked to specify a separate, explicit follow-up that can retain these qualified diagonals and test the missing crossovers with this single source-justified comparison exception. No simulator value, camera, control, state or tolerance has been changed.

## Evidence and reproduction

- `scripts/load05r_run.py`: deterministic reset/prefix/continuation driver; no physical-state copying.
- `scripts/load05r_state.py`: semantic comparator.
- `state-manifest.json`: solver-dependent field rules and exclusions, fixed before any continuation.
- `pinned-source/`: versioned upstream code with hashes.
- `/var/tmp/typhon-load05r-20261014`: original raw outputs, including STOPPED.json.
- `/media/nm/Fast Drive/Typhon/research-20261014/load05r-prefix`: designated verified archive.

Command from its worktree:

```sh
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv-sim/bin/python -u scripts/load05r_run.py
```

The driver has a frozen namespace and clock. Reproduction requires a new namespace and recorded clock; never rerun over this attempt. Source/destination archive verification is reported by the archive receipt, not assumed by this document.
