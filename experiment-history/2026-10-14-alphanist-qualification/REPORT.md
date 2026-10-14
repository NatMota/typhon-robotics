# ALPHA02 — does a longer horizon add manipulation skill?

**Disposition: park additional AlphaNIST-inspired PPO training in this prototype.** On this fixed diagnostic panel, opening the gripper with zero commanded arm displacement explains all four near-goal completions. Neither trained policy completes any full-start task in fifteen seconds or acquires a new grasp. The extra time lets the untrained policy eventually finish its fourth near-goal case; it does not reveal new full-task ability.

This is a diagnostic of our small MuJoCo prototype, not a reproduction or rejection of [AlphaNIST](https://arxiv.org/html/2610.13354v1). It does not test X-VLA, phone-video learning, or visual recognition of the blue numeral.

## Frozen comparison

| Frozen condition | Full starts: success by 4s / 15s | Already-held starts: success by 4s / 15s | Newly acquired grasps and lifts |
|---|---|---|---|
| U0: untrained small actor | 0/4 / 0/4 | 3/4 / 4/4 | 0 |
| A32: adaptive-reset PPO, 32,768 steps | 0/4 / 0/4 | 4/4 / 4/4 | 0 |
| U32: uniform-reset PPO, 32,768 steps | 0/4 / 0/4 | 4/4 / 4/4 | 0 |
| OPEN-only: zero joint deltas, open gripper | 0/4 / 0/4 | 4/4 / 4/4 | 0 |

32 new evaluation rollouts, eight fixed states per condition, two feeding and two loading source trajectories. Full starts are the same settled initial states used in ALPHA01. For each trajectory, the last saved bilateral-held state was selected by timestamp, without looking at policy outcomes. The held panel is deliberately near-goal and is not representative of all 32 development resets.

All three actors used deterministic tanh-mean actions and their original normalization. Their saved first-four-second action and object-position prefixes match ALPHA01 exactly. The only execution change is a 375-step cap instead of 100, at 25 Hz. Episodes stop on strict success or a velocity/torque limit violation. No safety violations occurred in the panel.

OPEN-only still uses the joint servo's impedance and gravity compensation. It has no learned network, transport target, expert correction or commanded joint displacement. Small physical motion while opening can arise from inherited velocity, gravity and servo/contact dynamics.

## What changed after four seconds

Only U0 on loading reset 54 changes from failure at four seconds to strict success at 14.80 seconds. The brick begins grasped above the destination. A32 and U32 finish that state in 1.44 seconds; OPEN-only does so in 1.28 seconds.

Training can change release timing, but this panel does not demonstrate a learned transport or pickup advantage. All four full starts remain incomplete under every condition at fifteen seconds, with no newly acquired bilateral contact. Contact-conditioned path lengths are retained in the episode logs; they include inherited dynamics and must not be interpreted as intentional learned transport.

## Reset and exposure audit

All 64 source resets are accounted for: 33 without bilateral holding or a goal-like released pose, 18 held away, 11 held near, and two unheld goal-like poses. “Held near” uses horizontal error <=30 mm and directed yaw error <=15 degrees; vertical distance is retained separately in the raw initial state and may be substantial.

No reset satisfies all endpoint criteria without a fresh stability interval. Reset clears terminal and history state. The eight reset states from the failed loading source rollout (IDs 32–39) remain in training and in the ledger.

All 190 adaptive-training successes and all 42 uniform-training successes occurred from just six training resets: 6, 7, 14, 15, 46 and 47. Five start held near the goal. Reset 47 has no bilateral contact but is already aligned just above the goal, with the gripper still classified closed; it is not evidence of a new pickup.

The adaptive sampler allocated 202 visits to five held-near training resets, versus 55 under uniform sampling. Held-away states received 81 versus 123 visits, with zero successes in either condition. This is evidence of exposure concentrating on finishing opportunities, not general manipulation competence. Different episode lengths and reset distributions make aggregate training success rates unsuitable for an equal-distribution comparison.

The ledger preserves task, split, parent seed, original source step, failed/successful expert parent, initial truth and score, and per-reset visits/successes/costs. Remaining source duration is measured to the stored final verified expert score; the first source success time is unavailable. It is not a minimum time or a feasibility guarantee for the joint controller.

## Observation, reset and reward contracts

- The actor receives object, **selected destination**, and robot surface point clouds plus joint/gripper state and the last action. Both stations remain visible in both tasks. Feeding and loading goals produce different goal-cloud inputs for identical physical states.
- “Residual” means neural-network residual blocks. No expert action is added to the policy.
- There is no remaining-time observation. Initial observations and ten replayed action observations were identical at the two horizon settings; the full saved prefixes also match.
- The original four-second timeout is treated as external truncation in PPO: the next-state value is bootstrapped, while the GAE recursion is stopped at the reset. Success and safety violations are terminal.
- Success requires correct position, directed yaw, top face, full footprint, height, release and fresh stability (at least 0.4 seconds). No inherited stability samples are accepted.
- The objective is binary strict success minus a normalized quadratic control cost. The median training penalty per episode was 0.00179 adaptive and 0.00231 uniform, versus a unit undiscounted success reward. The largest was 0.00361. This audit does not suggest an overpowering control penalty.
- Discount gamma=0.99 at 25 Hz discounts a terminal reward at four seconds to about 0.370 and at fifteen seconds to 0.0233. Extending evaluation does not retrain the value function for longer behavior. Any future longer-horizon training must specify its temporal objective explicitly.
- Old training logs preserve aggregate costs, not exact per-step discounted returns. ALPHA02 saves each reward and effort term; its discounted returns are exact for the recorded trajectories.

Initial audit categories were refined to identify two unheld goal-like states. Both audit versions are retained; panel membership is identical. This affected descriptive labels only, not actors, data selection, goals or execution.

## What this supports

The reset bank supplies a strong geometric-controller prior. Our current sparse-reward learner can sometimes finish a placement already prepared by that controller. It has not demonstrated pickup, full-task success, or an adaptive-sampling advantage on the common development panel.

Do not add PPO steps blindly. A future AlphaNIST effort would first need a separately qualified route from earlier states, appropriate coverage/exploration and a horizon-compatible objective. Those are new experiments, not repairs quietly applied to this comparison. The higher-priority video-grounded and counterbalanced loading work remains separate (LOAD04).

## Evidence and reproduction

See comparison.csv, per-state-results.csv, reset-exposure.csv, validation-summary.json, contract-audit.json, reset-ledger.json, source-sha256.json, and the saved visual comparison. Six rendered replays are checked against saved final positions and quaternions. Raw actions, rewards, truths, per-step criteria and videos are archived on Mint.

Worktree: ~/Projects/typhon-experiments/alphanist-qualification
Branch: research/alphanist-qualification-20261014
Raw Powerglide directory: /var/tmp/typhon-alphanist-qualification-20261014
Mint archive: /media/nm/Fast Drive/Typhon/research-20261014/alphanist-qualification

Use the exact ALPHA01 environment recorded in experiment-history/2026-10-14-alphanist/environment.json. No packages, pretrained weights or old experiment outputs were changed.

~~~bash
cd ~/Projects/typhon-experiments/alphanist-qualification
ALPHA_PY=~/Projects/humanoid-task-supervisor-m2/.venv-xvla/bin/python
SOURCE=/var/tmp/typhon-alphanist-20261014
OUTPUT=/var/tmp/typhon-alphanist-qualification-reproduction
"$ALPHA_PY" scripts/alphanist_qualification.py audit --source "$SOURCE" --output "$OUTPUT/audit"
for condition in U0 A32 U32 OPEN; do
  "$ALPHA_PY" scripts/alphanist_qualification.py evaluate \
    --source "$SOURCE" --audit "$OUTPUT/audit" \
    --output "$OUTPUT/evaluation" --condition "$condition"
done
"$ALPHA_PY" scripts/alphanist_qualification_render.py --source "$SOURCE" --output "$OUTPUT"
~~~

Use a fresh output path. The original reset bank and frozen ALPHA01 checkpoints are required and are SHA-256 identified. Do not interpret these eight correlated diagnostic states as a general reliability estimate or conduct a significance claim from them.
