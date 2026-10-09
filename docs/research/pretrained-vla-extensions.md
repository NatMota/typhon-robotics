# Research notes: extending a pretrained VLA

The aim is to improve task transfer and manipulation reliability without retraining an entire foundation model for each new workflow. The working setup uses a Franka robot in simulation and an RTX 4090. NVIDIA Jetson Thor is a potential deployment target.

This is a shortlist for experiments. It does not imply that every technique has been implemented, works with the selected X-VLA checkpoint or improves the current results.

## 1. Adaptation and corrective learning

| Technique | What changes | Training required? | Relevance to this project |
|---|---|---|---|
| FlowDAgger | A small latent policy selects the starting noise for a frozen generative policy, using inverted expert corrections as targets | Yes, for the small latent policy | High priority for corrective learning; requires a compatible generative policy and inversion procedure |
| Residual policy learning | A learned component adjusts a base policy's proposed actions | Yes | Useful for bounded corrections; action limits and interference with the base policy need testing |
| DAgger | Expert labels are collected in states reached by the learner and added to training | Yes | Directly addresses the difference between demonstration states and states reached during execution |
| LoRA | Small trainable matrices adapt selected model layers | Yes | A way to adapt more of the model with fewer trainable parameters |
| Soft prompts | Learned embeddings condition a pretrained model | Yes | A small adaptation budget; improved prediction loss still needs a task-success check |
| Midtraining | A model learns an embodiment before later task adaptation | Yes | Relevant to the embodiment gap, but potentially more work than a small task-specific experiment |

[FlowDAgger](https://microsoft.github.io/FlowDAgger/) learns from corrective interventions while keeping its base generator frozen. Its latent policy is trained to recover corrections through that generator. [Microsoft Rho](https://microsoft.github.io/rhobotics/) incorporates this correction approach alongside robot-specific midtraining. Rho's reported results do not establish that the same method transfers unchanged to X-VLA.

## 2. Planning and task transfer

| Technique | Purpose | Training required? |
|---|---|---|
| Hierarchical task planning | Break a goal into reusable manipulation skills | Not necessarily |
| Skill libraries | Select existing grasp, place, insert and recovery controllers | Not necessarily |
| Object-centric goals | Describe what should change about objects and their relationships | Depends on the implementation |
| Goal-conditioned policies | Use the same controller for different requested outcomes | Usually |
| State verification | Check whether a subtask succeeded before continuing | Not necessarily |

[LUCID](https://lucid-robot.github.io/) motivates separating task intent from the movements of a particular body. It predicts short-horizon intent from human video and uses an embodiment-specific controller to turn that intent into actions. That is a useful research direction, not evidence that Typhon has implemented LUCID or its training pipeline.

The question to test is whether existing skills can be combined to reach a new goal with little or no additional task training. Hand-written sequencing and learned transfer should remain distinguishable.

## 3. Memory and experience

| Technique | Purpose |
|---|---|
| Short-term state memory | Track task progress, recent observations and previous actions |
| Episodic memory | Store relevant successes and failures |
| Retrieval-augmented planning | Retrieve useful experience before choosing an action |
| Failure memory | Recognise conditions associated with earlier failures |
| Online adaptation | Update a small component using newly collected experience |

Memory can sit outside the VLA without changing its weights. Its contribution needs a matched comparison, including whether stale or irrelevant memories make behaviour worse.

## 4. World models and action evaluation

An action-conditioned world model estimates how a scene may change after an action. [V-JEPA 2](https://ai.meta.com/vjepa/) includes a robot-adapted, action-conditioned approach for planning. Other candidates include model-predictive control, ranking proposed actions, uncertainty estimates and checks for disagreement between predicted and observed outcomes.

One experiment would rank several candidate actions before execution, then compare that choice with the same policy without ranking. Measure prediction error, task success and compute cost. A world model is not automatically a dependable safety checker.

## 5. Inference and deployment

Humanoid reports using [NVIDIA Jetson Thor](https://thehumanoid.ai/humanoid-deploys-nvidia-technologies-to-power-advanced-humanoid-robots/). The deployment goal is to fit the full inference system into the selected device's memory and control-time budget. No Thor performance has been established by these notes.

| Technique | Main benefit | Training or calibration needed? |
|---|---|---|
| INT8 or INT4 quantisation | Reduce memory and potentially inference time | Post-training methods may use calibration; some methods retrain |
| `torch.compile` | Optimise supported execution paths | No model training; compilation and compatibility checks remain |
| TensorRT | Optimise inference for supported NVIDIA execution paths | Usually conversion and calibration, rather than model training |
| Fewer diffusion or flow steps | Reduce iterative generation time | May be changed without retraining, but action quality needs measurement |
| Action chunking | Generate several commands per model call | Already part of many policies; longer chunks change feedback frequency |
| Asynchronous inference | Generate the next chunk while the current one executes | Often no new model training; timing and state alignment still matter |
| Real-time chunking | Coordinate overlapping chunks and reduce discontinuities | Depends on the method and model interface |
| Adaptive inference frequency | Avoid calls when fresh reasoning is unnecessary | Can use fixed rules or a learned scheduler |
| Vision-token reduction | Reduce image-processing and attention cost | Depends on the method; small-object visibility may suffer |
| Action-head or model distillation | Train a cheaper predictor to imitate a more expensive one | Yes |
| Action caching | Reuse an action sequence while its assumptions remain valid | Not necessarily; requires invalidation when the scene changes |
| Early exit or dynamic computation | Use less computation for selected inputs | Implementation-dependent; some approaches need training |

These techniques primarily target efficiency. They do not by themselves establish better task transfer. Compare accuracy and realised robot behaviour as well as latency. If inference pauses the simulator, report that instead of claiming real-time operation.

## 6. Automated corrective data

Let the learner attempt a task in simulation. When a declared intervention condition is reached, a scripted expert can take over from the same state and supply a corrective demonstration. Retain failed expert attempts as well as successful ones.

Those demonstrations could support DAgger, FlowDAgger, residual learning or conventional fine-tuning. Automated expert recovery is the data-generation strategy. FlowDAgger is one specific way to learn from the corrections.

Keep the intervention rule, expert access to simulator truth, collected data and evaluation scenarios explicit. A timeout does not prove that the learner reached an irrecoverable state. Success by the expert after takeover must remain separate from success by the learner without help.

## What would count as progress?

Choose one addition at a time. Freeze the task, observations, scoring and allowed actions before comparing it with the unchanged policy. Measure completed tasks, verified grasps, recovery, inference time and total memory. A lower training loss or faster model call is useful evidence, but neither substitutes for successful execution.
