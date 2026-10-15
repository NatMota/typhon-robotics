# INIT02 — matched loading-initiation adaptation

P is the frozen FL parent. R receives further rehearsal and completed-feeding preservation training. T receives the same common samples, with 1,500 old-loading presentations replaced by new loading-transition demonstrations.

| Test | P: parent | R: control | T: new loading |
|---|---:|---:|---:|
| Load from completed feeding | 0/8 | 0/8 | 5/8 |
| Preserve completed feeding | 8/8 | 8/8 | 0/8 |
| Earlier feeding | 2/3 | 2/3 | 2/3 |
| Earlier loading | 3/4 | 3/4 | 3/4 |

These panels are reported separately. The eight initiation trials are two seeds at four nearby starts, not eight independent scene families.

## Supported pickup

| Panel | P | R | T |
|---|---:|---:|---:|
| initiation | 0/8 | 0/8 | 5/8 |
| prior feeding | 3/3 | 2/3 | 2/3 |
| prior loading | 4/4 | 3/4 | 3/4 |

Supported pickup requires the existing contact-supported lift of at least 10 mm for three consecutive steps. It is distinct from strict endpoint success.

## Completed-feeding motion

| Model | Maximum brick motion (mm) | Contact cases | Mean maximum wrist motion (mm) | Mean wrist path (m) |
|---|---:|---:|---:|---:|
| P | 0.0000 | 0/8 | 36.5 | 1.777 |
| R | 0.0000 | 0/8 | 9.5 | 0.361 |
| T | 311.7698 | 8/8 | 602.4 | 3.528 |

Preservation requires strict feeding success, no gripper-object contact and at most 1 mm object movement. Wrist travel is descriptive; object preservation alone does not prove a completion-recognition mechanism.

## Exact prior-skill cases

| Case | P | R | T |
|---|---|---|---|
| retain-personal_demo_0002-goal0 | PASS; pickup yes; 3.9 mm; 0.4 deg | PASS; pickup yes; 8.7 mm; 0.7 deg | PASS; pickup yes; 12.6 mm; 2.3 deg |
| retain-personal_demo_0002-goal1 | PASS; pickup yes; 12.7 mm; 0.2 deg | PASS; pickup yes; 6.4 mm; 0.4 deg | PASS; pickup yes; 6.3 mm; 0.0 deg |
| retain-personal_demo_0001-goal2 | FAIL; pickup yes; 456.2 mm; 55.0 deg; position;orientation;footprint_inside_target;placement_height;released;stable | FAIL; pickup no; 453.3 mm; 54.5 deg; position;orientation;footprint_inside_target | FAIL; pickup no; 453.3 mm; 54.5 deg; position;orientation;footprint_inside_target |
| retain-load01-training-demo_0001 | FAIL; pickup yes; 435.7 mm; 54.5 deg; position;orientation;footprint_inside_target;placement_height;released;stable;seated | FAIL; pickup no; 433.8 mm; 54.5 deg; position;orientation;footprint_inside_target;placement_height;seated | FAIL; pickup no; 433.8 mm; 54.5 deg; position;orientation;footprint_inside_target;placement_height;seated |
| retain-load01-training-demo_0002 | PASS; pickup yes; 8.1 mm; 0.0 deg | PASS; pickup yes; 8.8 mm; 0.6 deg | PASS; pickup yes; 7.5 mm; 0.6 deg |
| retain-load01-training-demo_0003 | PASS; pickup yes; 13.1 mm; 0.9 deg | PASS; pickup yes; 6.4 mm; 0.7 deg | PASS; pickup yes; 9.0 mm; 0.8 deg |
| retain-load01-training-demo_0005 | PASS; pickup yes; 7.2 mm; 0.6 deg | PASS; pickup yes; 7.6 mm; 0.8 deg | PASS; pickup yes; 0.5 mm; 0.2 deg |

All paired gains and losses in both strict result and pickup are enumerated in paired-changes.json. The full 69-case table, including loading failures, is comparisons.csv.

## Predetermined advancement gate

| Criterion | Passed |
|---|---|
| T loading at least6 | False |
| T loading at least3parents | True |
| T preservation8 | False |
| positive net loading gain vs R | True |
| no retention loss vs P or R | True |

The gate failed; the optional development panel was not run. No extra updates, new examples or outcome-selected checkpoint were added.

## Training

| Model | Joint updates | Wall seconds | Weight SHA-256 |
|---|---:|---:|---|
| R | 3000 | 984.28 | c6cdbd3eeddda9eb45e8e80d33c617581bbd08f3b5e614b8c06fbcb35bff1d21 |
| T | 3000 | 983.27 | 4d7a4a2547f084b6026164f27a6b7d43b5cb4d3b83ee4875343d108908f35636 |

Both runs start from the identical original FL weights and initial RNG, with fresh optimizer state, 12,000 presentations, effective batch four and seed 151901. All shared VLM/core weights and selected prompt/head parameters train jointly; inactive domain rows remain unchanged. Final update 3,000 is the only candidate. Optimizer moments were not archived.

## Interpretation limits and reproduction

The new examples use expert robot action trajectories at personally derived scenarios. They are not recovered human motor commands or DAgger data. The four initial geometries span only 10.542 mm. See METHODS.md, protocol.json and PLANNER_PROTOCOL.md for the frozen design and success definitions.

Independent re-scoring was run twice; comparisons.csv, validation_summary.json and optional-gate.json are byte-identical, and both passes verify unchanged raw inputs. Technical interruptions and pre-action initialization failures are preserved separately under integration-issue directories. The qualification amendment requires exact actual actor inputs; raw rendering is not claimed bitwise reproducible.

Read-only reproduction from this worktree:

    .venv-sim/bin/python scripts/init02_analyse.py
    TYPHON_INIT02_ANALYSIS_OUTPUT=/var/tmp/init02-independent-check .venv-sim/bin/python scripts/init02_analyse.py
    .venv-sim/bin/python scripts/init02_media.py
    .venv-sim/bin/python scripts/init02_report.py

Runtime: /var/tmp/typhon-init02-20261015. Archive root on Mint: /media/nm/Fast Drive/Typhon/research-20261015/init02-matched-adaptation. Large weights, tensor packets, traces and videos are archived outside Git. Full execution needs the frozen inputs at their documented paths; moving to another machine requires explicit path remapping and a fresh output namespace.

Four matched videos and four contact sheets are derived only from saved frames. media-validation.json records source hashes and frame counts. No new model calls or physics rollouts are used to create them.

The fixed INIT02 experiment ends at this branch; the wider OST programme continues with the planner.
