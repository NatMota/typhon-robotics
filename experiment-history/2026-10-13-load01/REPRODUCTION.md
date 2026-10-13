# LOAD01-v2 — Loading-specific supervision

Question: does adding verified fixture-placement examples establish loading competence beyond an equally long feeding-only continuation, and does feeding ability survive?

This is a supervised continuation experiment. It does not add a harness, candidate ranking, RL, new grasps, changed control limits or new image preprocessing.

## Three model identities

| Label | Exact role |
|---|---|
| P | Existing varied-goal model, SHA-256 d5449244d96e55e60c3342af9fffe2ae64362b74cdf9c3d41e39b74995c9bcdb |
| F | P plus 3,000 full-joint updates using feeding examples only |
| FL | P plus 3,000 full-joint updates using a 50/50 feeding/loading mixture |

P is the previous 3,000-update varied-goal continuation of broad Pt12k. These labels do not rename V1–V4.

Both continuations use a fresh identical optimizer, seed 131901, effective batch 4 through four microbatches, and 12,000 presentations. Of these, 6,000 feeding presentations are identical between F and FL. The other 6,000 match source-family order and within-family temporal-window quantiles: feeding in F, loading in FL. Every complete window is included. No partial terminal window is admitted.

Full VLM/core weights and the active domain-29 prompt/action-head rows are updated. Other domain rows remain fixed. Rates are prompt/VLM 1e-5 and heads/core 1e-4; AdamW betas 0.9/0.95, epsilon 1e-8, weight decay 0.01, gradient norm limit 1, bf16 autocast with float32 parameters. This is continuation from an adapted model, with no prompt reset or second warm-up. The final update 3,000 is the only selectable endpoint.

## What the data represents

Twelve existing simulated feeding episodes retain the six personally derived initial-state families. Four new simulated loading episodes use initial states derived from demos 0001, 0002, 0003 and 0005 and the supplied fixture goal. The unchanged privileged geometric expert generates robot actions.

These are human-informed scene reconstructions paired with generated robot commands. They are not direct frame-aligned human-video action supervision. The fixture goal, cameras, robot controller and provisional simulation block geometry are supplied. We do not claim metric reconstruction of the physical tabletop.

All four loading teacher episodes passed the original placement, orientation, release and stability scorer. Canonical action replay has zero measured object-position discrepancy. The qualified set contains 16 episodes and 1,165 complete 30-action windows; model observations use the unchanged O2 pipeline.

## Frozen evaluation

| Panel | Per model | Interpretation |
|---|---:|---|
| Loading source | 4 | Taught loading configurations, seed 7417 |
| Loading development | 2 | Loading combinations 0012/0013 absent from loading training, seed 912171 |
| Feeding retention | 6 | Existing new feeding-goal combinations, seed 912171 |

The total is 36 policy rollouts, with no harness or test-time candidate selection. All three models finish the source panel before any remainder panel starts. The original H30 native controller, 20 Hz execution, ten denoising refinements and strict scorer remain unchanged.

The four new loading-training configurations were historically inspected as evaluation scenes. Their role change was identified and approved before this experiment's data generation and training. Historical outcomes and action records remain unchanged. Families 0012/0013 can occur in feeding training; they are loading-combination holdouts, not unseen human families. No pooled success/12 headline or significance test is appropriate.

For every episode retain raw predicted actions, actual robot commands and joint/contact telemetry, exact RGB policy inputs, initial/final states, videos and original score components. Supported pickup means at least 10 mm lift with bilateral contact for at least three consecutive 20 Hz samples. Stable released placement is reported separately from correct yaw and strict success. Loading outcomes include distance to both the fixture and old feeding square.

## Execution and reproduction

Run from this branch in the qualified Powerglide environment. Existing parent/data assets must be restored at the paths recorded in the manifests. The scripts deliberately refuse to overwrite result directories. A full reproduction uses a clean machine or isolated filesystem namespace with the original paths; no fresh-host retraining is claimed merely from saved commands.

```bash
.venv-sim/bin/python scripts/load01_prepare.py prepare
.venv-sim/bin/python scripts/load01_prepare.py start
.venv-sim/bin/python scripts/load01_prepare.py generate
.venv-xvla/bin/python scripts/load01_train.py gate
.venv-sim/bin/python scripts/load01_run_training.py
.venv-sim/bin/python scripts/load01_run_evaluation.py
.venv-sim/bin/python scripts/load01_analyse.py
```

The original frozen configuration is in protocol.json. Raw outputs are under /var/tmp/typhon-load01-20261013; the independent archive is on Mint at /media/nm/Fast Drive/Typhon/research-20261012/load01-loading-supervision. Existing parent and feeding dependencies remain under the adjacent ost-sweep-20261012/goal-grounding archive, with exact checksum verification.

The evaluation driver can be started while training runs; it waits for both final checkpoints and their exact fresh-process probes. It then uses three concurrent inference lanes on the 4090. No paid pods are used.

T0 is 2026-10-13 19:11:40 UTC. Fixed boundaries: qualified data by 19:51:40, both trained models and reload checks by 21:01:40, evaluations stop by 21:46:40, archive/report closeout by 22:11:40. Failed qualifications are retained and stop the relevant stage; no outcome-based retries or checkpoint selection.

One file-reading correction was made before the first optimizer update: immutable NPZ archives are decompressed once instead of once per frame. The empty pre-update attempt and exact before/after hashes are retained. This changes I/O only, not the sampling or learning recipe. An earlier gate import failure was also retained and corrected before learning.

Final checkpoints retain model weights and RNG states, with optimizer state omitted to conserve scratch storage. Restarting the fixed continuation requires rerunning the frozen recipe from P.
