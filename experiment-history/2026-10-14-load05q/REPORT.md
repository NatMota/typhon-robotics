# LOAD05-Q — Restart verification qualification

**Result: STOPPED; restart not qualified. No model inference or crossover ran.**

This separate follow-up repaired the state comparator and added failure evidence. It did not change the state-copy implementation or reinterpret either failed attempt as a model result.

## Fixed scope

- Branch: `research/load05q-restart-qualification`.
- Executed code: `0cc33d95085f713484f316054b85d2ae315d6f3d`.
- Parent comparison: LOAD04 C/B, demo_0012, visible fixture, loading, noise seed 912171.
- Branch boundary: tick 100, before either archived first gripper–brick contact.
- 30-minute cap; at most two C/B qualification prefixes, no repair retries.
- Frozen original LOAD05 and LOAD04 results remain unchanged.
- Planner protocol and the stop update are preserved alongside this report.

## What ran

| Check | Result |
|---|---|
| XML deep-copy equality | Passed |
| XML attribute, child-order and text sensitivity | Passed |
| Numerical-change sensitivity | Passed |
| Unsupported opaque object detection | Passed |
| Simulator instances in copy-only tests | 0 |
| C archived prefix | 60 saved actions, reaching tick 100 |
| Prefix RGB/proprio and executed trace against archive | Passed |
| Native copy sharing check | No shared NumPy buffers reported |
| State after restore | Failed |
| B qualification / resumed 30-action chunk | Not run |
| VLA queries / crossed continuations | 0 / 0 |
| Outcome retries | 0 |

Run-clock start 1791998041.5938852; stop 1791998182.9889743. Approximately 141 seconds elapsed before the hard stop on mismatch.

## Actual mismatch

The independent before/after records agree on XML structure, Python/controller context, qpos, qvel, act, ctrl, qacc_warmstart, applied forces, and contact/constraint counts. Twenty-five exposed native constraint arrays differ after restoring the copied MjData. Numerous previously populated arrays, including efc_J, efc_D, efc_R and efc_force, become zero. Every numerical difference, selected element examples, both raw arrays archives, model XML and signatures are retained in `C/`.

Four fields (efc_AR and its colind/rowadr/rownnz arrays) also change between the independent before record and the snapshot signature, without a physics step. Their values include apparent uninitialized memory. They may be inactive dual-solver arrays. This is **not proof that the source physical state was corrupted**, and an all-public-array comparison can include undefined or inactive storage. These fields were not silently excluded. The qualification stopped as prescribed.

## Source inspection and limits

The installed MuJoCo version is 2.3.7. Its Python copy wrapper delegates to mj_copyData. The versioned C implementation copies fixed-buffer arrays and rebases arena pointers; the inspected function does not copy the arena contents. This is consistent with the copied constraint-array mismatch. It does not establish downstream trajectory divergence because no post-restore physics step was permitted.

Primary sources: [MuJoCo Python wrapper](https://raw.githubusercontent.com/google-deepmind/mujoco/2.3.7/python/mujoco/structs.cc), [MuJoCo engine copy implementation](https://raw.githubusercontent.com/google-deepmind/mujoco/2.3.7/src/engine/engine_io.c). The serialization path explicitly handles contact/primal and solver-conditional dual arrays, but native pickle was not tested.

The exact restored-state comparison remains unqualified. Simulator reconstruction by replaying the archived prefix is a proposed alternative requiring a separately frozen protocol; it was not run in this attempt.

## Reproduction and evidence

The executed qualification driver is `scripts/load05q_qualify.py`; serializer and failure evidence are in `scripts/load05q_state.py`. The CPU prepared-input probe in `scripts/load05q_prepare.py` was never reached. The qualification driver targets its frozen output namespace and must not be rerun over it. Reproduce in a new worktree and a new namespace with a new recorded clock, preserving this attempt.

Executed entry point from the worktree:

```sh
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv-sim/bin/python scripts/load05q_qualify.py
```

Raw output: `/var/tmp/typhon-load05q-20261014` on Powerglide.
Archive destination: `/media/nm/Fast Drive/Typhon/research-20261014/load05q-restart` on Mint.
The source/destination manifest and verification receipt are added to the archive after this report. A completed copy is not claimed until that receipt passes.

## Decision

Close this bounded qualification as a failed restart check. Keep LOAD05 blocked. Do not infer whether C or B causes the acquisition difference from this attempt. Retain the evidence and request one smaller, separately recorded follow-up from the planner.
