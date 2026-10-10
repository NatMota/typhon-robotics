# Geometric-expert movement-guard sanity check

All 36 successful geometric-expert paths passed the unchanged movement guards. All 36 also completed when their recorded targets were replayed with those guards actively enforcing each 30-step proposal and each executed command.

The 40 starting scenarios are exactly the original 20 feeding and 20 loading cases. These are privileged expert diagnostics, not learned-model successes.

| Task | Expert successes / attempts | Successful paths vetoed | Successes with guards enforced |
|---|---:|---:|---:|
| Feeding | 19/20 | 0/19 | 19/19 |
| Loading | 17/20 | 0/17 | 17/17 |

Across successful paths: 7,552 single-command checks and 2,526 overlapping 30-step windows (stride 3); zero rejections. Across all 40 expert attempts: 9,529 commands and 3,187 windows. The enforced replay executed all 7,552 recorded commands, with identical initial states and no runtime errors.

Successful expert paths reached at most 8.66 cm between checked targets (12 cm limit) and 2.025 rad/s measured joint speed (3 rad/s limit).

The four baseline failures were feeding-varied-07, loading-standard-03, loading-standard-07 and loading-varied-07. They failed while guards were observational only. All four encountered joint-limit violations; feeding-varied-07 also crossed workspace limits. They are retained as expert failures and are excluded from the successful-path false-rejection denominator.

## Method and limitations

- The original expert executes unchanged bounded OSC commands. The same absolute_target encoder used by training converts each command into the 20-value X-VLA target format. Decoding is checked against the original command at every step; maximum difference is below 1e-6.
- Guard configurations are copied from the completed evaluation logs, not reconstructed from assumed defaults. All 200 original harness trial configurations were checked against the per-scenario diagnostic configurations.
- The observational audit records both the production first-rejection reason and all independently evaluated geometric violations. It checks one action, three-action prefixes and full 30-action windows. The final partial window repeats its terminal target.
- Enforced replay uses recorded expert targets, checks the full 30-step window, executes at most three steps, and checks measured state again before each command. It uses the same native action adapter and physical task scorer. An extra 24 zero-motion steps provide terminal stability scoring.
- The image classifier, visual-uncertainty holds, completion stopping and 30-second wall cutoff are outside this movement-guard check. The result does not clear those other harness components.
- This is a finite regression over these scenarios. Task completion alone does not certify whole-arm collision safety.
- No production guard, model checkpoint, original scenario or existing experiment output was changed. All eight source hashes frozen by the 400-episode evaluation still match.

## Reproduce on Powerglide

From ~/Projects/typhon-harness-20261010, choose fresh output directories:

```sh
.venv-sim/bin/python scripts/typhon_expert_guard_audit.py \
  --evaluation /var/tmp/typhon-ab40-30s-parallel-20261010-v2 \
  --output /var/tmp/typhon-expert-guard-audit-reproduction --workers 3

.venv-sim/bin/python scripts/typhon_expert_guard_replay.py \
  --audit /var/tmp/typhon-expert-guard-audit-reproduction \
  --output /var/tmp/typhon-expert-guard-replay-reproduction --workers 3
```

The enforced replay exits unsuccessfully if any included positive path is rejected, loses task success, changes its initial state, misses commands or records runtime errors. The 29 existing guard/service unit checks also passed, including negative cases that must be rejected.

## Evidence

- Observational audit: /var/tmp/typhon-expert-guard-audit-20261010-v3
- Enforced replay: /var/tmp/typhon-expert-guard-enforced-20261010-v1
- Each case retains scenario, guard configuration, physical score, action/robot-state logs, initial/final images and a rollout video. Audit logs add per-command and per-window violation details; replay logs record each enforced check.
- Compact manifests, summaries and validation are copied into this repository folder.
- The initial fork-based renderer startup attempt failed before executing any robot command. Those infrastructure errors remain under /var/tmp/typhon-expert-guard-audit-20261010-v1; successful runs use isolated spawn workers.

## Interpretation

The current movement thresholds did not suppress these known successful executions. Keep them unchanged on this evidence. The next separate harness check should examine image-classifier uncertainty during valid grasps, lifts and transport, because a movement-guard pass does not test that hold logic.
