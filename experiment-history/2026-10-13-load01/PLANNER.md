## EXECUTE NEXT: LOAD01 — matched loading-inclusive versus feeding-only continuation

**Do the learning experiment, not another metadata repair.** LIVE01-T failed to resolve image provenance through copies, but its paired policy inputs, actions and physics remained identical. Preserve that failed metadata candidate; it is not a prerequisite for this experiment’s established file-based data and evaluation pipeline.

The next question is:

> **Can a small amount of qualified loading supervision teach the already-capable varied-goal policy to place into the fixture, without destroying its feeding competence?**

Your existing evidence makes this more useful than another isolated prompt or appearance probe: varied training improved new feeding combinations, yet the policy still lacks loading demonstrations and repeatedly chooses the feeding destination. That does not prove missing supervision is the cause. **The matched continuation will test whether adding it helps.**

Do not simultaneously introduce novel grasp alignments, change the fixture, repair proprioception, or add a supervisor.

## 1. Initializer and two training conditions

Use the **existing varied-goal continuation checkpoint**, abbreviated **`d5449244`** in the verified model registry—not original V4, not the later recovery model, and not a fresh Pt initialization.

Resolve its **full SHA-256 and exact path from the audited registry**, verify the file, and put both into the protocol before loading. I cannot independently supply the remaining hash characters: the private registry returned 404 through my connector. Do not substitute a similarly named checkpoint.

| Model | Initialization | Additional training |
|---|---|---|
| **P — parent reference** | Exact varied checkpoint | None |
| **F — feeding-only control** | Same checkpoint; fresh optimizer | 3,000 full-joint updates, feeding only |
| **FL — loading-inclusive** | Same checkpoint; identical fresh optimizer initialization | 3,000 full-joint updates, 50% feeding / 50% loading presentations |

**Only F and FL are new training runs.** Both use the same domain29 and existing action/observation conventions. No prompt/domain reset, LoRA, new architecture or another foundation-model warm-up.

## 2. Data gate: four loading demonstrations, genuinely eligible for training

### Loading selection

Use the four ordinary P_train parents:

**`0001, 0002, 0003, 0005`.**

First check for existing, independently qualified **training-eligible loading trajectories** linked to those parents. Historical loading evaluation episodes are **not** training material merely because the expert succeeded in them.

Where a loading trajectory is missing, generate **one nominal expert episode per missing parent**, using:

- That parent’s frozen training initial object configuration and the standard robot reset.
- The existing loading task’s nominal fixture geometry, destination, orientation and seat-height definition.
- The unchanged qualified geometric expert and controller.
- The existing simulator proxy, cameras and action-recording convention.

These are **new simulated loading demonstrations initialized from training-source configurations**. Do not describe them as loading actions recovered from Nat’s phone videos. Starting from these source configurations may differ from the historical staging-start distribution; record that explicitly.

No novel off-axis grasps, recovery controller, fixture-height sweep or outcome-selected destination. Native expert retries already in its fixed routine may run; do not add retries or replace difficult parents after inspecting outcomes.

### Required qualification

All four loading demonstrations must pass the original loading scorer, including seating, release and stability. Verify complete action/observation records and canonical conversion.

**If any cannot be qualified within the data gate, stop before training and report the specific teacher/data blocker.** Do not admit a failed trajectory, substitute a quarantined development example or keep searching for easier geometry.

Keep unchanged:

- P_validation `0006/0014/0016`.
- All historical loading/development evaluation configurations.
- Original V4 and later recovery quarantines.
- Existing human exclusions and source-family identities.

This is deliberately nominal loading acquisition—not yet a test of different acquired grasps.

### Feeding selection and matched exposure

Retain the **verified varied-goal feeding dataset** used by the initializer.

At each four-example optimizer batch:

| Batch slots | F | FL |
|---|---|---|
| **Two common slots** | Identical feeding examples | Identical feeding examples |
| **Two comparison slots** | Feeding examples from the four designated source parents | Loading examples from those same four parents |

Over 3,000 updates:

- **6,000 common feeding presentations**, balanced across the six P_train families.
- **6,000 comparison presentations**, exactly 1,500 per designated parent.
- **12,000 presentations per model**, not independent demonstrations.

Match family ordering and random window-selection quantiles in the comparison slots; their trajectories need not have identical lengths or timestamps. Use complete windows under the existing loader—no new phase oversampling, terminal padding or time compression.

This tests **replacing half the additional feeding exposure with loading supervision**. It does not isolate fixture appearance, height and task wording individually.

## 3. Freeze the interface and optimizer

Keep the initializer’s qualified native pipeline: O2, current observations, existing domain29 proprio/action encoding, 30 commands, 20 Hz, H30 execution, ten refinement steps, existing controller and original scorers.

**Audit the task instruction once before training.** Loading records must contain the actual loading destination and task description, not an inherited instruction to place in the staging square. Use the existing task-specific template; verify that it survives tokenization. No new free-form prompt search.

Use the qualified fixed/varied **full-joint continuation recipe**, pinned from its executed manifest—not current upstream defaults. Copy its optimizer groups, learning rates, decay/masking, clipping, precision and accumulation settings identically into F and FL. Print those resolved settings in the protocol before launch.

The planned budget is **3,000 optimizer updates at effective batch four**, with fresh optimizer state, one fixed training seed **131901**, and both shared VLM and action-transformer weights trained throughout. Preserve inactive domain rows according to the already-qualified implementation.

Keeping the action representation and data processing fixed matters here: X-VLA’s method includes both adaptation and aligned, balanced action supervision. This is a bounded task-data continuation experiment, not reproduction of its published training scale. [arXiv](https://arxiv.org/html/2510.10274v1)

Save the **final update-3,000 checkpoint only as the scientific endpoint**. Recovery saves are not selectable candidates. Fresh-process restoration must reproduce the saved probe outputs under matched inputs, noise and flags.

## 4. Fixed evaluation: 48 policy rollouts, source-first

Freeze evaluation IDs and seeds **before generating new training trajectories or inspecting new policy outcomes**.

| Panel | Selection | Per model | Interpretation |
|---|---|---:|---|
| **Loading source competence** | Four qualified training configurations, seed7417 | 4 | Can the model execute the loading task it was taught? |
| **Loading development** | Six distinct configurations from the existing frozen 12-trial loading panel, retaining the first recorded seed for each | 6 | Performance on the already-inspected loading configurations |
| **Feeding retention** | One existing varied new-combination configuration per source family, chosen by scenario-ID order; retain its first recorded seed | 6 | Whether feeding ability survives the continuation |
| **Total** | P, F and FL on the same 16 cases | **48 overall** | Small matched development comparison |

Verify that no loading-development configuration is identical to an admitted training configuration. **Do not move an overlapping evaluation case into training.** Resolve any overlap before the freeze; if the intended panel cannot be formed, document the blocker rather than quietly reclassifying it.

Run the four loading-source cases first for all three models, then complete the frozen development and retention panels regardless of source success. No loading-result-based checkpoint selection or extra training.

Use existing matching expert feasibility evidence where its configuration/controller hashes qualify. Otherwise allow **at most two predetermined development expert controls**, separate from policy counts. A failed control stays in the record; it does not justify replacing the case.

### Report the physical chain

For every policy episode retain:

**Approach → bilateral supported pickup → transport destination → fixture entry/seating → orientation → release/stability.**

Report these separately from strict success. In particular:

- Closing without contact is not grasping.
- Correct yaw at the feeding square is not loading.
- Placement on the fixture rim is not seating.
- Report distance to **both** the requested fixture and old feeding destination.
- Keep the existing pickup definition and its threshold; do not pool incompatible historical definitions.

Retain original strict task criteria. Region-only outcomes may be descriptive secondary results, labelled as such—not replacement endpoints.

## 5. What the result will decide

| Outcome | Interpretation |
|---|---|
| FL loads on source configurations; P/F do not | Task-specific supervision can establish loading competence under this setup |
| FL also improves loading-development cases | Evidence of transfer within this bounded task-data recipe |
| FL fits source loading but not development | Loading is learnable here; initial-state/task-distribution coverage remains limiting |
| FL reaches the fixture but fails seating or release | Destination selection improved; contact/height/terminal control remains unresolved |
| FL still goes to the feeding region, including source cases | More examples under this recipe did not establish the task distinction; inspect fitting and conditioning before extending training |
| F improves similarly to FL | Additional optimization, rather than loading data alone, may explain the gain |
| FL gains loading but loses feeding | A measured task trade-off, not an unqualified overall improvement |
| Teacher qualification fails | We do not yet have the intended supervised experiment; do not label it a policy-learning failure |

A positive result **will not prove numeric-language grounding**: fixture appearance, task instruction and demonstrated destination still change together. A negative result will not show that fixture height or imagery is intrinsically incompatible with the model.

## 6. Time cap and operational stops

**Hard cap: 180 elapsed minutes from the recorded start.**

| Boundary | Required action |
|---|---|
| **T+40 min** | Data, initializer, instruction and evaluation manifests qualified; otherwise stop before training |
| **T+110 min** | Both training runs and final reload checks complete; no training extension |
| **T+155 min** | Stop policy/expert evaluation |
| **T+180 min** | Finish report and verified Mint archive, with incomplete stages explicitly listed |

The previously measured ~0.31 seconds per full-joint update implies roughly **31 minutes for 6,000 updates**, before loading, checkpoint I/O and verification. Treat that as a planning estimate, not a guaranteed benchmark for this dataset.

Preflight **local scratch capacity for both full checkpoints, temporary writes and offload headroom**. Use atomic finalization and verified Mint copies; do not repeat the quota-truncated save pattern. Save authoritative frame/sample IDs and arrays through the established dataset path. The failed weak-reference telemetry registry is not required.

Stop an affected run for nonfinite training, malformed observations/labels, failed reload, unintended parameter changes, insufficient storage or the protected deadline. Preserve failures; no hyperparameter search or silent fallback.

**Execute LOAD01 after sealing the LIVE01-T failure receipt.** The purpose is to test whether verified fixture-placement examples unlock a missing task while retaining feeding—not to spend another cycle making metadata look complete or assuming that more identical feeding updates will solve loading.