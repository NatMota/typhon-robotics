# INIT13 — measured release clearance

A fixed, measured-open vertical retreat removed the two reproduced finger–fixture contacts while retaining all three loading successes and the already-completed feeding result. **All eight prescribed episodes and all qualification checks passed.** This qualifies the rule for further testing, not deployment or generalisation.

The model did not learn this motion. Q1 is unchanged; the new routine uses measured robot pose/aperture and declared fixture geometry to supply a short clearance primitive.

## Frozen comparison

Four previously inspected development cases were selected before this run: two known contact cases, one contact-free loading control and one already-completed feeding control. Each ran with D5 only and D5 plus clearance (C). No outcome retries, new seeds, holdouts, training or conditional prior-skill trials were added.

D5 remains the existing command-triggered 5 mm world-Z decrease for two fresh H30 chunks (60 actions). The new rule activates once, after D5 finishes and an executed close has occurred, when both the previous executed and current proposed gripper commands are open and measured aperture is at least 55 mm.

It holds measured wrist XY and orientation, setting Z so the fully open collision envelope would be 20 mm above the highest fixture wall. It returns to native targets after measured actual envelope clearance is at least 10 mm for three consecutive 20 Hz samples. A new close or 60-action timeout would return native control and flag failure. Native gripper commands and H30 inference continue unchanged, including inference during the primitive; history contains actual issued first-nine float32 pose values plus eleven zeros.

This is additional controller knowledge, explicitly not an RGB-only, object-truth-triggered or learned obstacle-avoidance method. Object truth and contact wrenches score outcomes offline.

## Results

| Case / seed | Task | D5 strict | C strict | Contact samples D5 / C | Final position mm / yaw deg (C) |
|---|---|---:|---:|---:|---:|
| 0001-loading-912171 | loading | True | True | 1 / 0 | 5.643 / 0.819 |
| 0003-loading-912172 | loading | True | True | 1 / 0 | 5.849 / 1.007 |
| 0002-loading-912171 | loading | True | True | 0 / 0 | 6.936 / 0.557 |
| 0001-feeding-912171 | feeding | True | True | 0 / 0 | 12.068 / 0.338 |


Both contact cases reproduce their historical D5 contact at action 149. C activates at 147 in all three loading cases and replaces five pose commands (0.25 seconds of simulated motion), handing back before 152. No timeout, premature close or later forbidden contact occurred. Feeding does not activate either rule and has zero block contact or displacement.

The commanded rises are 22.570, 16.448 and 15.620 mm. Measured minimum clearance over the final three primitive samples is 10.923, 12.962 and 13.109 mm respectively. Additional total measured wrist path is 6.627, 7.784 and 6.739 mm. These are measured-path differences, not the target rise or extra wall-clock time.

Position/yaw errors above are final goal errors. Both loading branches finish with the same block placement to floating-point precision. Release and seating remain successful. The original score includes position, yaw, correct face up, footprint containment, placement height, release, temporal stability and seating; forbidden contact is a separate qualification gate.

Contact evidence is sampled at 20 Hz. Zero recorded contacts is not a statement about unsampled substep maxima or physical-robot safety. No hardware trial took place.

## Qualification and evidence

- The predeclared trace audit found measured 55 mm opening before the known contact in all four historical offending traces; only the prescribed two contact cases were executed here.
- A detached static-geometry sweep found no obvious vertical obstruction. It did not step or modify live simulation and was not substituted for executed outcomes.
- The exact restored Q1 reload probe passed with maximum action difference 0.
- All eight original strict scores and within-run issued-action/history contracts were independently recomputed.
- All four initial-state/input/noise pairs and pre-clearance comparisons are exact. All four D5 controls reproduce their complete historical D5 trajectories exactly.
- D5 still changes exactly 60 commands in loading and none in completed feeding. Clearance changes only active-arm pose; gripper and inactive channels are preserved.

Execution commit: f7e84b41d4c2acc2dfa72647af22b1e12f3886b8.
Independent analyser: eb081ff6437d61c77796b2538ae3b0dae0a53318.
Q1 checkpoint SHA-256: 26d573eb90f791d9af86bbff30e138f51c5aa658ac7abefe76787ba352363a7a.

The report uses the original corrected model cache, bfloat16 inference, ten denoising steps, H30, domain 29/O2 and the existing controller. No weights, gains, limits, episode horizons or frozen earlier outputs changed.

## Visual review

Agent review of all four saved-video sheets shows the brick seated in the fixture at both loading endpoints; the completed feeding case remains untouched. The side camera shows the brief raised handback pose. The five-fps sheets cannot establish absence of a single 20 Hz contact sample; telemetry provides that evidence.

- [Contact case 0001 / 912171](visual-review/init-demo_0001-loading-912171.png)
- [Contact case 0003 / 912172](visual-review/init-demo_0003-loading-912172.png)
- [Contact-free regression 0002 / 912171](visual-review/init-demo_0002-loading-912171.png)
- [Completed feeding preservation](visual-review/init-demo_0001-feeding-912171.png)

The sheet labels refer to matched comparison times. “Handback” is actual clearance handback only in C loading; neither D5 nor the inactive feeding condition performs this primitive. Frame indices, original control ticks and video checksums are in [frame-provenance.json](visual-review/frame-provenance.json). This is agent inspection, not additional human annotation.

## Interpretation and next decision

The result supports a narrow mechanism: acquisition improved under D5, and a short measured release-clearance motion can remove its selected post-opening wall contacts without changing those placement outcomes. It does not show that Q1 understands fixture collision geometry, recognises task completion or generalises to new loading starts.

Keep D5+C frozen for a separately specified broader qualification. Do not tune the height or thresholds on further failures. INIT12's failed advancement and original RGB-first exact-parity failure remain unchanged; this experiment does not retroactively pass them.

## Reproduction and archive

Commands for this exact frozen execution were:

~~~sh
cd /var/tmp/typhon-worktrees-20261016/init13
.venv-sim/bin/python scripts/init13.py
.venv-sim/bin/python scripts/init13_analyse.py
.venv-sim/bin/python scripts/init13_media.py
~~~

Do not rerun into the frozen directories. For saved-artifact verification, import the analyser, point its input root at the archive and its output at a fresh directory. A new physical execution requires a separate output namespace, frozen cases and explicit clock; original absolute deadlines are deliberately not reusable.

Full archive: /media/nm/Fast Drive/Typhon/research-20261015/init13-measured-clearance

Verified 1105 files, 254,601,536 bytes, zero checksum errors. Source-manifest SHA-256: 799c53b30584890f404a4bb86949d1eadfd915b85203d2458cdb4d3e40d973c6.

Clock: activated03:08:16 UTC; physical completion03:20:51 UTC; all artifacts verified03:31:29 UTC, before the original04:23:16 closeout deadline. No clock reset.

The private branch contains source, protocol, report, bounded evidence and review sheets. The full Mint archive retains videos, all raw/issued predictions, noise, prepared inputs, controller/force traces, state boundaries, exact source manifests and qualification records. No dataset or model is stored on the Mac.
