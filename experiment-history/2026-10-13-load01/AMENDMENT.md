## APPROVED: execute LOAD01-v2 with 36 policy rollouts

**Your preflight is correct. The contradiction was in my protocol—not your implementation.** Use your proposed amendment: retain the four loading training configurations, evaluate loading transfer on the **two disjoint configurations**, and retain the six feeding-retention cases.

Preserving every historically evaluated layout as permanently untrained is not essential for this exploratory experiment. **Preserving its exposure history—and not misrepresenting newly trained layouts as holdouts—is essential.**

## 1. Explicitly amend the configuration roles

This amendment permits **four previously inspected loading configurations to become training geometry in LOAD01 only**. It does not alter the historical evaluation records or authorize importing their saved action trajectories.

| Panel | Configurations | Cases per model | Interpretation |
|---|---|---:|---|
| **Loading source competence** | `0001/0002/0003/0005`, using newly generated, qualified loading demonstrations | **4** | Loading competence on FL’s training configurations; these layouts were also inspected in historical evaluations |
| **Loading development transfer** | `0012/0013`, first historical seed **912171** | **2** | Loading-task combination holdouts from feeding-trained families; previously inspected development cases, not new humans or untouched tests |
| **Feeding retention** | The six previously specified varied feeding-combination configurations and their frozen seeds | **6** | Retention of feeding competence |
| **Total** | Same cases for P, F and FL | **12 × 3 = 36** | Report the three panels separately |

Retain the previously specified **7417 source-panel seed**, with identical resolved physical initialization across models. Confirm the actual initial-state hashes during evaluation. Different seeds do not turn the four source configurations into unseen geometry.

**Do not pool these panels into a headline “successes out of 12.”** Report loading source **/4**, loading development **/2**, and feeding retention **/6**.

## 2. Initializer and training remain unchanged

Use the locally verified varied-goal initializer:

```text
original goal-grounding/train-varied/step3000.pt

SHA-256:
d5449244d96e55e60c3342af9fffe2ae64362b74cdf9c3d41e39b74995c9bcdb
```

The three conditions remain:

| Label | Condition |
|---|---|
| **P** | Unchanged initializer |
| **F** | Initializer + 3,000 full-joint updates, feeding only |
| **FL** | Same initializer + 3,000 full-joint updates, half feeding and half loading |

Keep the qualified continuation optimizer, learning rates, precision, effective batch four, seed **131901**, domain29 and action/observation conventions fixed.

Each trained condition receives **12,000 presentations**. Preserve the **6,000 identical common feeding draws** and the **6,000 matched-family comparison draws**: feeding for F, loading for FL, with 1,500 comparison draws per designated parent. No additional grasp variants, recovery data, phase oversampling or outcome-selected checkpoints.

## 3. Freeze the amendment before generating trajectories

Commit a **LOAD01-v2 protocol and role manifest** that retains the rejected preflight and states:

> Four configurations used in earlier loading evaluations are intentionally admitted as new loading-training geometry for LOAD01. Their historical evaluation results remain unchanged. Newly generated expert actions supply training labels; no archived evaluation action trajectory is imported. Loading transfer is assessed separately on two disjoint, previously inspected configurations.

Then verify these gates:

| Gate | Requirement |
|---|---|
| **Geometry separation** | Neither `0012` nor `0013` is an exact resolved configuration match to any of the four loading-training configurations |
| **Data boundary** | No loading trajectory, window or label from `0012/0013` enters F or FL |
| **Training qualification** | All four new loading demonstrations pass the original loading scorer and action/observation contract |
| **Frozen evaluation** | Configuration hashes, seeds, model labels and endpoints are saved before training; outcomes cannot change them |

Existing **feeding** data from families `0012/0013` remains permitted as planned. Thus these are **loading-combination holdouts**, not entirely unseen families.

If one of the four training demonstrations fails qualification, **do not replace it with `0012/0013`, reseed until success, or import a historical evaluation trajectory**. Stop at the existing data gate and report the blocker.

Existing matching expert evidence may establish development-case feasibility. If the previously permitted expert feasibility checks are necessary, they remain evaluation-only, outside the 36 policy rollouts and outside training.

## 4. What this smaller comparison can establish

The amendment still answers the main learning question:

- **FL improves source loading over P/F:** loading-specific supervision established competence on the taught configurations.
- **FL also improves on `0012/0013`:** useful transfer to these two loading combinations, despite their absence from loading supervision.
- **FL succeeds only on source cases:** acquisition without demonstrated transfer on this development panel.
- **FL gains loading but loses feeding:** a measured trade-off.
- **F improves similarly:** additional optimization remains an alternative explanation.

With only two loading-development configurations and one training seed, report exact paired outcomes and failure stages. **Do not present the result as a general loading-reliability estimate or add a significance test.**

A positive result would still concern the **loading-data recipe**, in which fixture appearance, task text and demonstrated destination change together—not isolated numeric-language grounding.

## 5. Clock and execution order

After the amended protocol and manifests are committed, record **T0**. Keep the original protected boundaries:

| Boundary | Limit |
|---|---:|
| Qualified data and launch gate | **T+40 min** |
| Both training runs and final reload checks complete | **T+110 min** |
| Stop all evaluation | **T+155 min** |
| Report and verified archive closeout | **T+180 min** |

The reduction from 48 to 36 policy rollouts provides margin, **not permission for additional conditions**.

**Proceed with LOAD01-v2.** The overlap is now an explicit prospective design decision, not hidden leakage. Historical artifacts and quarantined action data remain untouched.