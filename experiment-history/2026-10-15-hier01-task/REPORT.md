# HIER01-TASK: a supplied-task dispatcher preserves both local capabilities

A fixed dispatcher using GOAL-varied for feeding and LOAD01-FL for loading completes all four selected contexts; the single LOAD04-B checkpoint completes three. The predeclared second-seed confirmation repeats the useful difference: routing completes G1 feeding and shared-scene loading; B again misses G1 while completing loading. This is a practical local composition of existing policies, with no new training.

Ticket #11066 is a bounded child of HIER01 #11000. It does not qualify the blocked autonomous phase selector, recovery controller or residual-RL branches.

## What the dispatcher does

`select_checkpoint(task)` reads only the caller's externally supplied `feeding` or `loading` field. Feeding selects GOAL-varied (SHA d5449244…); loading selects LOAD01-FL (76334b9a…). Single-model control always selects LOAD04-B (de861da4…). Full hashes are in the frozen protocol.

Selection occurs once before the episode. Scenario ID, target coordinates, RGB, simulator truth, candidates and outcomes cannot choose or change the route. An unsupported task raises an error before execution. There is no observer, additional guard, expert rescue, phase detection, extra waiting, new stopping rule or mid-episode switch. The original task instruction still goes to the selected VLA.

These are learned motor policies trained with geometric-expert action supervision. Loading was explicitly included in FL training. This is neither a learned hierarchy nor loading transfer from feeding-only data.

## Results by context

Each cell gives strict completion and final position error. A/B/C/D are context IDs; “single B” names the checkpoint.

| Stage / supplied request | Single B | Fixed task router |
|---|---|---|
| Main A: original G0 feeding | Pass, 7.92 mm | GOAL pass, 3.12 mm |
| Main B: relocated G1 feeding | Fail, 145.95 mm | GOAL pass, 5.53 mm |
| Main C: feeding, both stations visible | Pass, 7.45 mm | GOAL pass, 4.68 mm |
| Main D: loading, identical scene to C | Pass, 10.91 mm | FL pass, 8.13 mm |
| Confirmation B: G1, second noise seed | Fail, 138.75 mm | GOAL pass, 7.11 mm |
| Confirmation D: loading, second noise seed | Pass, 4.62 mm | FL pass, 5.93 mm |

Main stage: router **4/4**, single B **3/4**. Conditional confirmation: router **2/2**, single B **1/2**. The confirmation stage was triggered by the frozen all-four-success/no-loss rule; it is not an independently selected reliability sample.

All 12 executions acquire a contact-supported lift and transport the block. Final face tilt is zero in every case; yaw errors are 0.04–1.42 degrees. Both B G1 failures miss centre/footprint while preserving orientation, release and stability. The shared-scene feeding/loading runs complete their requested destination, not the opposite station. The final comparison sheet was visually inspected.

## Qualification and unchanged contracts

The three checkpoint hashes and native O2/domain29/interleaved action convention were verified. Direct and dispatched calls on one fixed packet/noise per specialist have exact prepared inputs, outputs and actual RNG/noise arrays, and reproduce the archived FIT direct outputs exactly. These are four separate qualification generations, not physical episodes.

Each route pair has identical initial available simulator/controller/random state, raw images, processed inputs and original instruction. Every query uses matched actual noise arrays. C and D have byte-identical initial state and exactly the same physical configuration; only the caller's task/instruction and consequent selected checkpoint differ. H30, 20 Hz, ten refinement steps, 1,500-tick horizon, 40 reset ticks, 24 stability ticks and low-level controller are unchanged. Every strict score was recomputed independently.

Twelve native videos and two route-comparison videos have 367 decodable frames at 5 fps. The paired videos cover G1 feeding and shared-scene loading. `policy_io.npz`, predictions, inference timings and rich controller traces remain in the raw archive.

## Runtime and trade-off

Mean episode wall time was 17.39 seconds for routing and 17.62 seconds for B; model inference totals about 2.95 seconds across 48 queries in either condition. Fresh worker loading averages 2.42 versus 2.52 seconds, excluding preceding file hashing and orchestration. These small timing differences are not a speed claim.

The router requires two existing approximately 3.73 GB checkpoint files, versus one for B. Only one model is resident on the GPU at a time. The experiment deliberately reloads per episode for isolation; it is not a production serving benchmark.

## Boundary of the result

This is one previously inspected, personally derived training family, with existing G0/G1 and shared-station scenes. Repeated noise seeds are not independent demonstrations. The result supports a simple explicit-task interface in these contexts; it does not repair B internally, establish broad generalisation, infer free-form intentions, or grant a perception harness authority to stop or redirect motion.

The older EVAL02 136-baseline results and 68 unexecuted intervention trials remain separate. This router is not silently substituted for that planned intervention arm. Frozen data, holdouts, older configurations and results are unchanged.

## Reproduction and preservation

Branch `research/hier01-task-routing`, execution freeze `2b2c27660fba4052b63982a8918edd6a775172d9`. Use the exact environment, archive and checkpoint paths in `protocol.json`; scripts use a dated root, so choose a new root for reproduction and preserve sealed originals.

```bash
python3 scripts/hier01_task_prepare.py
.venv-sim/bin/python scripts/hier01_task.py qualify
.venv-sim/bin/python scripts/hier01_task.py main
# Only if the frozen next-branch.json selects confirmation:
.venv-sim/bin/python scripts/hier01_task.py confirmation
.venv-sim/bin/python scripts/hier01_task_analyse.py
```

Preparation encountered UCL's 100 GB home quota before any physical run. A 135,517,741-byte old Mint pilot archive was copied to Mint, checksum-verified and offloaded; models and labs were untouched. Qualification still finished within the original ten-minute limit. The quota receipt and incident record are included.

Raw output: `/var/tmp/typhon-hier01-task-20261015`. Verified backup: `/media/nm/Fast Drive/Typhon/research-20261015/hier01-task-routing`. Private Git carries source, protocols, metrics and figures; bulk arrays/videos are on Mint. The research programme stays active and the planner has the outcomes for the next bounded test.
