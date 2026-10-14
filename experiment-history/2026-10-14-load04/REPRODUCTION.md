# LOAD04 reproduction

The full raw archive is required for result replay. New training also requires the pinned LOAD01/LOAD03 inputs and installed private X-VLA/simulation environments. This private research package is not a claim of a verified clean-machine install.

## Recompute the recorded results without running a robot

Restore the raw LOAD04 directory and the experiment worktree. The existing scorer and dependencies must be installed.

```bash
export TYPHON_LOAD04_ROOT=/path/to/restored/typhon-load04-20261014
.venv-sim/bin/python scripts/load04_analyse.py
.venv-sim/bin/python scripts/load04_instruction_audit.py
.venv-sim/bin/python scripts/load04_diagnose_failures.py
.venv-sim/bin/python scripts/load04_media.py
```

These read the saved episodes, tensors, controller traces and videos. They do not step the simulator or run policy inference. Derived analysis/QA files are regenerated; preserve a copy if comparing hashes.

## Repeat the frozen training experiment in a new directory

1. Restore the immutable LOAD01 and LOAD03 input directories at the paths recorded in `scripts/load04_common.py` and `load04_prepare.py`. This includes the parent checkpoint, original loading action logs, reset fixtures and protocols. Preserve the parent SHA check.
2. Restore/qualify `.venv-sim`, `.venv-xvla` and `.cache-xvla/checkpoint` with the archived environment. The original run used Powerglide's RTX4090. New hardware/rendering requires requalification.
3. Choose a **new, nonexistent** output directory. Never point fresh execution at the archived result.

```bash
export TYPHON_LOAD04_ROOT=/var/tmp/typhon-load04-reproduction-NEW-ID
.venv-sim/bin/python scripts/load04_prepare.py prepare
# Commit and inspect the generated protocol before starting the clock.
.venv-sim/bin/python scripts/load04_prepare.py start
.venv-sim/bin/python scripts/load04_data.py
.venv-sim/bin/python scripts/load04_data_qa.py
.venv-xvla/bin/python scripts/load04_train.py gate
```

All eight trajectories must pass; failed attempts are retained and are not replaced. Before training, inspect the recorded initial/final camera observations and visible/hidden comparisons, use the generated unreviewed template as a starting point and then write the **new run's** `qa/visual-review.json` with an honest `passed` value, reviewed files and limitations. The archived passing review must not be copied as a new review.

```bash
.venv-sim/bin/python scripts/load04_pipeline.py
.venv-sim/bin/python scripts/load04_instruction_audit.py
.venv-sim/bin/python scripts/load04_diagnose_failures.py
.venv-sim/bin/python scripts/load04_media.py
```

The pipeline enforces the original relative time caps: data qualification +30 minutes, both fits/reloads +85, evaluation +110 and analysis +120. It preserves fixed data membership, sample schedule and 32 evaluation cases. It does not extend the budget automatically.

The final checkpoint files contain inference weights and RNG, **not optimizer state**. They are exact endpoint inference checkpoints, not exact training-resumption checkpoints.

## Executed versus reproduction code

The historical run stopped after its first pair, then resumed the remaining six after the recorded qualification addendum. The exact executed scripts are retained under raw `executed-source/` with hashes; the Git artifact contains `executed-source-manifest.json`.

After evaluation, the entry point was clarified: `load04_data.py` starts fresh, while `--resume-qualified` is explicitly restricted to the historical two-trajectory qualification state. The output root is now configurable through `TYPHON_LOAD04_ROOT`. These reproduction-only edits were syntax-checked after execution; they did not trigger another fit or rollout. A complete new training reproduction has not been run.

## Storage and validation

Use `SOURCE_MANIFEST.json` and the Mint verification receipt to check the delivered raw package. Record the checkout commit, source manifest and checkpoint SHA-256 with any rerun. A copied file, a successful rsync and a completed hash check are recorded separately. The broader machine/lab archive is a separate operation.
