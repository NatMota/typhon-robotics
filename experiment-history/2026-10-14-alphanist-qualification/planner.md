## EXECUTE NEXT: ALPHA02 — one reset/horizon qualification, with no further PPO training

**ALPHA01 establishes that the prototype runs, not that adaptive reset sampling has improved manipulation.** Before spending more PPO steps, test whether the current setup mainly rewards finishing already-prepared placements—and whether four seconds prevents useful earlier-stage behavior.

Keep backup work running. The verified booking and completed lab archives stand as your reported receipts; no additional reservation is needed. **This gate must not turn into a replacement for LOAD04 or the video-grounded backlog.**

## 1. Interpretation of the present results

| Fixed development panel | Untrained | Adaptive | Uniform |
|---|---:|---:|---:|
| Before training | **6/32** | — | — |
| 4,096 steps | — | 8/32 | 7/32 |
| 32,768 steps | — | 6/32 | 8/32 |
| Full initial states | **0/4** | **0/4** | **0/4** |

**There is no established adaptive-sampling advantage on this panel.** Equal aggregate counts do not mean identical behavior; preserve paired gains and losses. The small differences, one learning seed and four underlying development trajectories do not support ranking the methods generally.

The adaptive run’s **190/470 training successes versus 42/369** cannot be read as improved common-distribution competence. The sampler changes which starts the policy receives, and successful early termination can also change the number of episodes collected within the same interaction budget.

Your sampling formula can assign a state with estimated success 0.5 **eleven times** the weight of a state at 0 or 1. It reallocates existing opportunities; it cannot create missing reset coverage. AlphaNIST explicitly separates these roles: the reset bank determines available states, while success-guided sampling determines their exposure. [arXiv](https://arxiv.org/html/2610.13354v1)

**Already-held continuation success is legitimate, but it is not learned pickup or full-task competence.** The untrained 6/32 makes a nonlearning finishing control particularly valuable.

### Keep the paper comparison precise

The official project still lists code as forthcoming. Its reset distribution includes ungrasped, held and partially assembled states; the paper evaluates single-pair completion within **15 seconds** and uses substantially different task diversity and simulation scale. Your four-second, expert-derived reset prototype is therefore an **inspired feasibility experiment**, not a reproduction or a test that refutes AlphaNIST. [AlphaNIST](https://alpha-nist.github.io/)

“Not imitation labels” is accurate, but **the geometric controller still supplies a demonstration-derived reset-state prior**. Preserve that contribution.

## 2. ALPHA02’s question

> **Do the frozen trained policies acquire useful behavior from earlier states when given more execution time, or are observed successes largely explained by near-terminal resets and simple release?**

This is one evaluation gate: **existing weights, existing states, one longer observation window and one nonlearning control.**

### First: a 12-minute audit of the 64 existing resets

Produce a compact reset ledger from existing records—no new labels, reset generation or calibration.

| Check | Required distinction |
|---|---|
| **Initial difficulty** | Ungrasped; held away from destination; held near destination; released/goal-like; unknown. These are offline descriptors, not actor features. |
| **Inherited success** | Flag states already satisfying the physical endpoint. Confirm scorer history and success latches cannot carry a previous episode’s completion into a new episode. |
| **Source provenance** | Parent trajectory, source time, task, original expert success/failure and train/development membership. Include resets from the failed source rollout explicitly. |
| **Exposure** | Actual adaptive/uniform visit counts and successes per reset. Show whether additional sampling reached earlier states or concentrated on finishing states. |

Where the source trace provides a remaining time to strict success, report it. **That duration is evidence under the source controller—not a minimum completion time or feasibility certificate for the new relative-joint interface.**

Three short implementation checks are also necessary:

**Task specification:** with both stations visible, confirm how the intended destination reaches the actor. Selected mating/goal geometry or an explicit task feature can supply it. If identical observations require different rewarded endpoints with no distinguishing goal input, record an observation-contract defect. AlphaNIST supplies the relevant assembly geometry through its point-cloud observation; absence of a discrete task ID does not mean absence of task information. [arXiv](https://arxiv.org/html/2610.13354v1)

**“Residual” meaning:** state whether this means residual neural-network blocks or an action residual added to another controller. If another controller contributes motion, identify that contribution before calling the untrained network the baseline.

**Reward economics:** use saved traces to separate success reward, accumulated control cost and discounted return. “Small coefficient” does not establish that the total penalty is small relative to one terminal reward. Do not alter the coefficients during this gate.

## 3. Fixed physical panel: at most 32 episodes

### States: at most eight

Use the **four existing full-start controls**. For each corresponding development source trajectory, select the **last saved reset with bilateral holding before release**, provided it is not already strictly complete.

Select by existing timestamps and reset IDs—not by which policy succeeded. If a corresponding held state does not exist, mark that slot unavailable; do not manufacture or replace it.

This deliberately contrasts **starting the task** with **finishing an already-acquired placement**.

### Four conditions

| Condition | Definition |
|---|---|
| **U0** | Original untrained actor |
| **A32** | Final adaptive 32,768-step actor |
| **U32** | Final uniform 32,768-step actor |
| **OPEN-only** | Zero relative arm displacement and the documented OPEN command throughout; no learned network, transport target or expert correction |

That is **four conditions ? at most eight states = at most 32 episodes**.

OPEN-only is not intended as a good controller. It asks whether the reset has already done enough that releasing and settling completes the task. Its outcomes stay explicitly nonlearning.

Use the existing evaluation action-selection convention and one frozen policy seed per state, paired across the three actors. Keep normalization frozen, as well as action limits, controller, physics and scoring.

### Measure at four seconds, then continue to fifteen

Run each eligible episode for at most **375 actions at 25 Hz**, recording the original result at **100 actions/four seconds**. Continue past four seconds only when the original time limit—not genuine success or a safety termination—would have stopped it.

Do not reset state, add a finishing controller or change the reward after four seconds.

**Qualification gate:** inspect whether remaining time enters the policy observation or affects other execution logic. If extending the limit changes the first four seconds of conditioning, it is not a pure execution-time comparison. Stop and document that dependency rather than inventing a new timer representation.

Also record whether the four-second limit was implemented as task termination or external truncation, and how PPO bootstrapped it. Those distinctions affect value targets; neither treatment is universally correct independently of the defined task. **Audit, do not repair or retrain in this experiment.** [Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

## 4. Score what the policy actually contributes

For every episode, retain:

| Measurement | Purpose |
|---|---|
| Strict result by four seconds and by fifteen seconds | Separates the time cutoff from eventual observed completion |
| First success time | Shows whether extra time actually matters |
| Grasp inherited at reset versus newly acquired | Prevents continuation success becoming a pickup claim |
| Supported transport after reset | Distinguishes release-only completion from useful manipulation |
| Destination, directed yaw, release and stability | Preserves the original task requirements |
| Termination reason and control cost | Separates task failure, time limit and safety/effort effects |

Stability evidence must be newly collected within the episode. A goal-like reset that was already complete belongs in a preservation category, not “new task completion.”

Report per source trajectory and task. **Do not pool eight selected resets into a general reliability estimate.**

## 5. Decision gate: continue only if there is a useful mechanism to pursue

| Result | Interpretation and disposition |
|---|---|
| OPEN-only explains most near-terminal successes; trained policies add no earlier-state capability | The current aggregate score largely measures finishing ease. **Park this prototype**, rather than add PPO steps blindly. |
| Trained policies complete earlier starts after four seconds but before fifteen, beyond U0/OPEN-only | The four-second cutoff limits demonstrated behavior. A separately planned horizon-compatible training comparison becomes justified. |
| Longer execution helps only already-held states | Useful finishing behavior, not full-task learning; earlier reset coverage and exploration remain unresolved. |
| Source traces suggest longer tasks, but every frozen policy still fails at fifteen seconds | More time alone is insufficient. This does not establish physical impossibility or identify one algorithmic cause. |
| Missing task information, inherited success, or materially misaligned reward is found | Record a concrete formulation defect. Do not spend more steps optimizing it unchanged. |

No significance testing or outcome-selected checkpoint choice is needed.

## 6. Cap and closeout

**45 minutes total:** twelve for the ledger/contract checks, twenty for the fixed evaluations, thirteen for analysis and verified Mint preservation.

No new reset bank, PPO updates, reward changes, framework migration, paid compute or simulator-truth inputs beyond the already-declared observation contract. Backups retain priority; stop cleanly if storage or qualification fails.

**For Nat:** the prototype currently shows that it can sometimes finish a task the reset has largely prepared. It has not yet shown a reliable gain from training or adaptive sampling. This next gate determines whether the immediate limitation is the four-second window, easy reset credit, or a more substantive coverage/formulation problem—before committing more GPU time.
