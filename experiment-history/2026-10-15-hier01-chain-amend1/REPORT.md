# HIER01-CHAIN — Isolated skills did not compose

**Both completed workflows failed on the second task.** Feed→load completed feeding but did not acquire the brick from the feeding square. Load→feed completed loading but did not acquire it from the fixture. The explicit task router works on the tested reset scenes; that does not establish a usable continuous workflow.

| Continuous workflow | First stage | Second stage | Whole workflow |
|---|---|---|---|
| Feed→load | Feeding passes; 4.55 mm centre error, supported lift | Loading fails; 266.21 mm error, no supported lift | Fail |
| Load→feed | Loading passes; 5.35 mm centre error, supported lift | Feeding fails; 274.74 mm error, no supported lift | Fail |

Both second stages leave the brick satisfying the preceding task's endpoint. Neither requested second task was already satisfied at its start. The second-stage final scores are freshly computed from that segment's own observations, not inherited from the preceding successful stage.

## What was executed and preserved

There are **five executed physical task segments** across the original block and this explicit amendment: two feed→load stages, one loading first stage interrupted by a technical verification stop, and two stages in one fresh load→feed attempt. The original unexecuted feeding stage is not counted as a policy failure. The repeated loading stage is extra execution evidence, not an additional independent workflow.

The original partial record is sealed at 47089a52b59460a8dcb61112cb3aed3c986743a7 and its complete 403-file archive is SHA-verified on Mint. This amendment was approved in Hm challeng p2 before execution and frozen at 97c2ce04b50477adbed1900aa79853eaf00030a3. It repeated only load→feed, using the same model hashes, scene, seeds, task strings and numerical settings. It completed before the original 15:25:50 UTC execution deadline.

## The verification amendment

The original extra gate stopped on six raw secondary-camera channel differences of one intensity level, after a successful loading stage. All actual prepared actor tensors, predictions, noise and physical outcomes were identical. The amendment records raw differences separately and retains exact checks on all 48 effective model input hashes, predictions, command histories, robot trajectories, requested/applied commands, rich controller/contact traces, actual noise arrays, initial full available state and terminal/stability truth.

The fresh first loading stage passed those checks. No control component consumes raw RGB outside the qualified actor preprocessing; other RGB consumers are instrumentation and artifact writers. This was not a numerical image tolerance or a relaxation of the success scorer. Any new effective-input, physical-state or runtime discrepancy would have stopped the single follow-up.

## Continuous-state proof

The same simulator and low-level controller remain alive across each task boundary. The available physical/controller/solver/contact/random-state blob is byte-identical before and after request initialization; prior terminal truth and rendered observations equal the next segment's initial state. The new request starts from measured current EEF pose and a fresh native command history.

No reset, home motion, object injection, extra physics ticks or success-triggered switch is used. The lifetime bookkeeping guard is extended to tick 2,960, while each stage still executes exactly 1,436 policy actions plus 24 settling steps. Controller gains, motion limits, geometry and camera settings are unchanged. The first stages match their standalone DEV effective inputs and execution; the second stages deliberately receive the inherited state and are not required to match a reset episode.

## Evidence and limits

The combined contact sheet was visually inspected. Four canonical segment videos, two continuous workflow videos (734 frames at 5 fps each), and the interrupted first-stage video are retained across the linked archives. Raw tensors, native proposals, traces, endpoint scores and boundary state blobs are preserved. See validation_summary.json, comparisons.csv and combined-summary.json.

The completed feed→load trace already shows inactive model proposals: open-gripper values throughout, zero contacts and no object movement, with the EEF staying more than 45 cm from the brick. This identifies an approach/acquisition failure in policy output, not an executed valid grasp rejected by the controller. The internal cause and reverse-direction mechanism are left to HIER01-HANDOFF #11070, a saved-evidence request/visibility/training-coverage diagnosis.

These are two fixed task orders in one inspected development context, not a general reliability estimate. No claim about numeral recognition, autonomous completion, hardware safety or arbitrary composition follows. No training, recovery suffix, hidden reset, observer, TTC or holdout change was introduced. Old EVAL02 counts remain unchanged.

## Reproduction and locations

Branch: research/hier01-handover-amendment. Raw: /var/tmp/typhon-hier01-chain-amend1-20261015. Original raw: /var/tmp/typhon-hier01-chain-20261015. Mint destination: /media/nm/Fast Drive/Typhon/research-20261015/hier01-handover-amendment.

Use the pinned simulator Python and exact archived sources/checkpoints in protocol.json. For a fresh replication, select new roots and explicitly timestamp its clock; preserve both sealed runs.

```bash
SIM_PY=/cs/student/msc_rai/2026/ndossant/Projects/typhon-experiments/data07r-training-repairs/.venv-sim/bin/python
"$SIM_PY" scripts/hier01_chain_amend.py qualify
"$SIM_PY" scripts/hier01_chain_amend.py run
"$SIM_PY" scripts/hier01_chain_amend_analyse.py
```

The combined analysis also requires the preserved original feed→load records and partial-validation receipt. The research programme remains active after this experiment.
