# HIER01-CHAIN — Preserved partial run and handover failure

**Feed→load failed at its second task.** Feeding passed; the loading policy then stayed near home with its gripper open, leaving the brick untouched in the feeding square. The reverse workflow executed a successful loading first stage, then stopped at an overly strict raw-image equivalence check before its feeding stage.

This archive contains **three physical task segments, one completed workflow and one interrupted workflow**. The unexecuted reverse second stage is not counted as a failed policy trial. Any retry requires a separately recorded amendment and fresh run.

| Workflow | Segment | Strict outcome | Supported pickup | Centre error |
|---|---|---|---|---:|
| Feed→load | Feed | Pass | Yes | 4.55 mm |
| Feed→load | Load | Fail | No | 266.21 mm |
| Load→feed | Load | Pass; subsequent verification stop | Yes | 5.35 mm |
| Load→feed | Feed | Not executed | Not evaluated | Not measured |

## Continuous-world qualification

The simulator, controller, object, joints, velocities, solver warmstart, contacts and available random state remain live across the completed handover. Before/after request-initialization state blobs are byte-identical. Stage 2's initial truth and rendered scene equal stage 1's final state. There is no environment reset, home prefix, object injection or repeated 40-tick startup.

Each request uses 1,436 policy-action steps and 24 settling steps. The first request starts at tick 40 and ends at 1,500; the second ends at 2,960. Only the lifetime step-budget bookkeeping is extended to permit two requests; controller gains, action limits, geometry and rendering remain unchanged. The new policy request uses measured current EEF position/orientation with its ordinary zero gripper/dummy first-query convention, then native previous-command history. Task switching occurs at the fixed budget, not an oracle success signal.

## What failed in feed→load

After feeding, the brick centre is (0.127365, -0.116110, 0.814784) m; the loading destination remains the fixture at (0.12, 0.15) m. The arm is near its home pose.

Across all 48 loading queries and 1,440 predicted actions, commanded gripper values range 0.000276–0.001584, below the 0.5 closing threshold. Every proposed EEF target is at least 417.20 mm from the brick's initial centre; the closest actually executed EEF approach is 451.73 mm. There are zero contact samples, zero object displacement and no supported lift. The brick still satisfies the feeding endpoint, while the requested loading endpoint fails position, footprint, height and seating.

The earliest observed failure is **no useful approach/acquisition proposal**, not a controller rejecting a valid pickup. The internal explanation remains unresolved: task/context grounding, completed-scene bias and missing source-state coverage are hypotheses. This result does not prove inability to recognise the blue numeral.

## Why load→feed was interrupted

An additional native-equivalence gate compared every raw rendered pixel with the previous standalone loading run. After a successful first stage, it found six secondary-camera channel values differing by one intensity level: two RGB pixels at query 46, rows 478/479 and columns 168/167, among 37,748,736 channel values.

Read-only audit establishes:

- All prepared actor input hashes are exact at all 48 queries.
- All 28,800 predicted action values and all 960 command-history values are exact.
- All actual noise arrays, measured EEF trajectories, initial/final/stability truth and initial prepared tensors are exact.
- Primary-camera raw images are exact; only those six secondary-camera values differ.

The raw rendering difference was eliminated by native preprocessing. Nevertheless, the frozen gate stopped execution and the simulator was closed. We preserved the original gate, error log and partial run; no state-restored continuation or silent replacement was performed.

## Evidence and next work

Three native videos, model inputs/proposals/noise, rich controller traces, the partial contact sheet and the pixel-level audit are saved. The contact sheet was inspected visually. Raw output is /var/tmp/typhon-hier01-chain-20261015.

The planner was given the completed development panel, this failure and a proposed transparent amendment: keep exact prepared-input/action/state/noise checks, log raw-render differences separately, and run only one fresh load→feed workflow. That would produce five total physical segments including the interrupted duplicate first stage. The planner response and any new execution belong to a separate result root and commit.

Branch: research/hier01-continuous-handover. Execution freeze: 7fb0f3b850a19c91122d12d2c101ebef953de7d6. Use the pinned simulator interpreter with scripts/hier01_chain.py qualify, scripts/hier01_chain.py run and scripts/hier01_chain_audit.py. The original runner intentionally reproduces the strict gate; choose fresh output paths and an explicit clock, preserving this archive. The unused four-segment analysis draft is not evidence of completion.

No training, expert rescue, observer, TTC, new holdout or original EVAL02 intervention trial was added. The research programme remains active.
