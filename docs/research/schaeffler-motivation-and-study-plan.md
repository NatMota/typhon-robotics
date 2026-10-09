# Why focus on feeding and loading?

The research started with a practical question: can experience from feeding parts help a robot load a different workstation, with less additional teaching and fewer interventions?

Humanoid's Schaeffler proof of concept provides the industrial motivation. Its published account describes bearing rings picked from cluttered bins, transferred to a buffer table and carried between stations. It also discusses reuse across ring types and correcting grasps. Feeding and loading preserve that source-to-destination relationship while keeping the first simulator experiment manageable. [Published Schaeffler account](https://thehumanoid.ai/humanoid-and-schaeffler-successfully-completed-proof-of-concept-for-bin-picking-with-pre-alpha-robot/)

The experiment uses a coloured block and a shallow fixture. It does not claim to reproduce metal-ring geometry, factory throughput or machine operation.

## Task priorities

These qualitative assessments were used to choose a research focus. They are not measured customer risk or financial estimates.

| Task | Likely repetition | Economic opportunity | Downside of failure |
|---|---|---|---|
| Box and tote handling | High | Medium to high | Medium: dropped loads, delays and possible injury |
| Feeding | High | High | High: incorrect parts and production stoppage |
| Part loading | High | High | High: machine damage, scrap and downtime |
| Assembly | Medium to high | High | Very high: defective assemblies, rework and potentially latent faults |
| Inspection | High | High, depending on defect costs | Very high: missed defects, recalls and safety consequences |
| Packaging | High | Medium | Low to medium: damaged goods and incorrect shipments |

## Illustrative opportunity model

The following figures are preserved from the planning notes as an illustrative scenario. The 25,000 workstations, task shares, operation counts and net values are assumptions. They are not verified Schaeffler site counts, a deployment forecast, or measured returns. A cost model supporting the net-value estimates was not supplied, so these figures cannot support an investment decision.

| Task | Assumed share | Assumed workstations | Assumed annual operations | Assumed net value per robot per year |
|---|---:|---:|---:|---:|
| Box and tote handling | 30% | 7,500 | 375 million | €13,200 |
| Feeding | 20% | 5,000 | 750 million | €20,000 |
| Part loading | 20% | 5,000 | 500 million | €19,900 |
| Assembly | 10% | 2,500 | 156 million | €9,100 |
| Inspection | 10% | 2,500 | 188 million | −€15,700 |
| Packaging | 10% | 2,500 | 94 million | €13,200 |
| Total | 100% | 25,000 | 2,063 million, approximately 2.06 billion | No aggregate specified |

The model made feeding and loading worth investigating. Whether they create value in a real deployment depends on utilisation, cycle time, rejected work, maintenance, integration and the human work that remains.

## Proposed training data: Job A, part feeding

The initial target was approximately 60 demonstrations.

| Scenario | Planned demonstrations | Example |
|---|---:|---|
| Normal feeding | 40 | Pick scattered components and place them in staging positions |
| Human interference | 10 | A person moves, removes or pre-places a component |
| Other disturbances | 10 | Target occupied, component missing or an action fails |
| Total | 60 | Collection target, not an achieved count |

The completed dataset was smaller. Use the repository's dataset provenance and experimental manifests for actual collection counts and training splits.

## Proposed transfer test: Job B, machine loading

Introduce another workstation with a different destination and goal. Supply the policy with:

- The workstation specification, potentially through a QR code.
- The current machine and fixture state.
- The available action primitives.
- Readiness and safety constraints.

The proposal excludes machine-loading demonstrations from training. This is a proposed information contract, not a claim that QR input or a machine-readiness interface has been implemented.

| Planned test | Trials |
|---|---:|
| Feeding with unfamiliar layouts | 20 |
| Loading under normal conditions | 20 |
| Loading with human interference | 20 |

Run a rule-based baseline on the same scenarios, with the same observations, supplied goals, execution limits and scoring. Report task changes separately from layout changes. Failure recovery and already-completed task preservation also need separate denominators.

These are original planning targets. They must not be substituted for the frozen protocols or reported as completed evaluations.

## Intended deployment platform

NVIDIA Jetson Thor was the intended onboard deployment target; the development platform was an RTX 4090. Humanoid's published use of Thor made compute and memory efficiency relevant to the experiment. [Humanoid's NVIDIA integration](https://thehumanoid.ai/humanoid-deploys-nvidia-technologies-to-power-advanced-humanoid-robots/)

A Thor deployment needs hardware validation of model compatibility, total memory, observation-to-action latency and sustained operation. The research does not claim that these checks have passed.
