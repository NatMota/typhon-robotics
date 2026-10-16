# INIT11-Q — reload-context discrepancy

Six fixed probe calls isolated a repeatable loader-context mismatch. The current INIT11 context returned the same incorrect output in all four A/B calls; the original INIT07 context returned the frozen reference exactly in both C calls. No robot rollout or training occurred.

| Context | Fresh processes | Calls | Exact reference matches | Maximum XYZ difference |
|---|---:|---:|---:|---:|
| Restored INIT11 cache | 2 | 4 | 0/4 | 3.90625 mm |
| Original INIT07 cache | 1 | 2 | 2/2 | 0 mm |

All six calls used identical effective input values, identical initial sampled action noise, the original seed7381 inference helper, bf16 autocast, ten refinements and the unchanged Q1 checkpoint/probe. All model parameter and buffer hashes match across contexts and remain unchanged after calls. Every context was internally repeatable. Current-context outputs differ in240/600 values, with0.03136degrees maximum decoded orientation difference and0.000872 maximum gripper probability difference; no gripper threshold crossing occurs. Raw-identical original outputs yield a roughly2e-6-degree numerical matrix/arccos roundoff, not a physical discrepancy.

## Identified restoration defect

The new INIT11 worktree linked .cache-xvla to humanoid-task-supervisor-m2/.cache-xvla. Original Q1 evaluation linked it to humanoid-m6-joint/.cache-xvla, whose checkpoint resolves to .cache-xvla-pt. Configuration, model wrapper and action-hub source differ. The wrong context instantiated Florence2Attention; the original instantiated Florence2SdpaAttention. Both shared-encoder embedding aliases were correctly tied and final checkpoint parameters were identical. Action-hub source contains differing XYZ_SCALE constants (100 versus500); this diagnostic does not isolate the causal contribution of each difference or establish whether each changed class is active. The first captured transformer refinement already differs.

The before/after source audit that initially found no drift had compared the restored cache with its own historical copy, rather than the original Q1 evaluation cache. That audit was insufficient to establish model-runtime equivalence.

## Scope and next gate

A and B retain the current worker initialization; C retains the original worker/root and documented launcher environment. The original worker prefix was run unchanged except a transparent loading-report wrapper and replacement of the first probe assertion with two fixed evidence-saving calls. No worker readiness or action loop is entered. Complete actual/expected tensors, inputs, noise, ten refinement outputs, RNG states, flags, imported modules, CUDA library hashes and full cache source are retained.

Restore the original cache only in a new INIT11-R worktree. Require the original exact probe in two fresh processes, then88 archived input/noise/action calls and the original prefix/physics gates. Preserve stopped INIT11 and this six-call comparison. No changed expected output, tolerance, model weight, scientific condition, simulator state or holdout.

## Reproduction and evidence

Executor commit f07a1c0a6c433789ea89159dc02c0f8c5edae987; analysis commit8259ae05109d9a0676106a2d3a461c7c3778f9cd. Private branch research/init11q-reload-20261016.

Commands, with both archived worker contexts restored and a fresh output directory:
~~~bash
python3 scripts/init11q_run.py
python3 scripts/init11q_analyse.py
~~~

The runner refuses an existing output; never overwrite the frozen results to repeat this diagnostic. Runtime archive: /media/nm/Fast Drive/Typhon/research-20261015/init11q-reload-qualification. Backup verified88files,21,129,872bytes,zeroSHA errors. The recorded six-call execution completed in under2minutes; the full archive was verified before the50-minute deadline.

This resolves the source of the reproducibility block at the cache-context level. It does not answer whether a rescued grasp can finish the loading task.
