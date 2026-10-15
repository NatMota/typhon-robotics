# HIER01-DEV — Task routing on two additional starting configurations

The unchanged caller-task router completed **feeding 4/4 and loading 4/4**. Single B completed **feeding 4/4 and loading 2/4**. Routing gained two loading cases and lost none. This qualifies the next fixed two-task handover diagnostic; it does not establish broad generalisation or autonomous task selection.

The two B failures were **not pickup failures in this panel**. Both lifted and transported the brick to the fixture centre, then left its marked face tilted 90 degrees. They failed face-up, height and seating criteria despite small centre and yaw errors. FL placed it face-up and seated in both matched cases.

## Frozen comparison

Exact archived LOAD02 demo_0012/demo_0013 configurations, cross-checked against LOAD03's visible-fixture arrays; feeding and loading requested with their existing complete instructions. Both stations remain present. Two native policy/reset seeds, 912171 and 912172. Routes: B versus feeding→GOAL-varied / loading→FL. All sixteen qualified cases were executed.

These are previously inspected development families seen in feeding training, with their specified loading combinations excluded from loading supervision. They are not new independently sampled families. No state, label or trajectory from this experiment entered training.

## Paired outcomes

Errors below are centre distances to the requested destination. Strict success also requires yaw, marked-face tilt, footprint, height, release, stability and loading seating.

| Family | Request | Seed | B strict / error mm | Router strict / error mm |
|---|---|---:|---|---|
| demo_0012 | feeding | 912171 | True / 1.51 | True / 4.55 |
| demo_0012 | feeding | 912172 | True / 1.49 | True / 4.92 |
| demo_0012 | loading | 912171 | False / 1.97 | True / 5.35 |
| demo_0012 | loading | 912172 | False / 1.43 | True / 7.06 |
| demo_0013 | feeding | 912171 | True / 5.96 | True / 10.95 |
| demo_0013 | feeding | 912172 | True / 6.37 | True / 12.55 |
| demo_0013 | loading | 912171 | True / 3.13 | True / 7.24 |
| demo_0013 | loading | 912172 | True / 3.03 | True / 10.52 |

All sixteen episodes achieved the independently checked contact-supported lift (bilateral contact and at least 10 mm lift for three consecutive samples). No episode completed the opposite endpoint. Release and stability passed in all sixteen. All router final tilts were zero; the two B failures were 90 degrees.

Routing did not improve centre precision: B had smaller centre error in every pair. The two demo_0013 feeding router errors were 10.95 and 12.55 mm against a 15 mm tolerance. Its demonstrated benefit here is preserving complete task success across both destinations, including face-up loading.

## Qualification and evidence

The executed task dispatcher, actor modules, simulator/controller source tree, checkpoint/probe hashes and native O2/domain29/H30/ten-refinement/20 Hz contract match the sealed HIER01-TASK qualification. The direct/dispatched proof was reused without new qualification model calls. Each physical rollout performed the existing fresh reload probe. Unsupported tasks still fail before selection; the dispatcher accepts only the supplied task.

Archived initial raw images, effective proprioception, qpos/qvel/mocap arrays and truth were reproduced exactly. Full currently available simulator/controller/random state was byte-identical across route pairs and feeding/loading requests at each family/seed. This is not a claim that unarchived historical controller internals were compared. Prepared model inputs, including the original task instruction, and all 48 actual per-query noise arrays matched within route pairs. Both endpoint scores were recomputed from saved outcomes with the original destination-specific scorer.

All sixteen native videos and two side-by-side comparison videos decode through frame 367 (5 fps). The final contact sheet was visually inspected: B demo_0012 loading shows the unprinted face upward; FL shows the numeral face upward. Raw model proposals, effective histories, prepared inputs, rendered observations, noise and rich controller traces are preserved. No simulator truth entered the actor. No observer, TTC, movement guard, expert rescue, new stopping rule or learning was added.

## Runtime and continuation

single: mean native episode wall 17.41 s; inference total 2.95 s per episode; fresh worker load 2.47 s, excluding preceding checkpoint hashing.

router: mean native episode wall 17.30 s; inference total 2.95 s per episode; fresh worker load 2.42 s, excluding preceding checkpoint hashing.

The frozen all-eight-success/no-loss rule selects HIER01-CHAIN #11068: feed→load and load→feed on demo_0012 with the simulator retained across the task boundary. No claim about continuous handovers is made by this reset-based panel. The broader research goal remains active.

## Reproduction and preservation

Execution freeze: cb876a293e65fc50677c98a73e2ec361519ea8bf. Branch: research/hier01-development-routing. Start from this branch with the exact archives/checkpoints listed in protocol.json. For a fresh replication, use a new output root in all three scripts and record a new 10/35/15-minute clock; never overwrite sealed output. Use the pinned Python 3.11.12 / NumPy 1.26.4 / MuJoCo 2.3.7 simulator environment:

```bash
SIM_PY=/cs/student/msc_rai/2026/ndossant/Projects/typhon-experiments/data07r-training-repairs/.venv-sim/bin/python
"$SIM_PY" scripts/hier01_dev_prepare.py
"$SIM_PY" scripts/hier01_dev.py
"$SIM_PY" scripts/hier01_dev_analyse.py
```

Raw: /var/tmp/typhon-hier01-dev-20261015. Backup destination: /media/nm/Fast Drive/Typhon/research-20261015/hier01-development-routing. Private Git stores code, protocol, metrics and contact sheet; bulk arrays/videos remain in the verified Mint archive. Verification receipts carry the final commit and all file hashes. M0/M1/M1.5, historical evaluations, holdouts and quarantines remain unchanged. These are zero additions to the old EVAL02 intervention arm.
