**Approved. Use that reconstruction-based qualification; CTRL05 need not stop solely because the original full tick590 snapshot was not saved.**

The new comparison requires S and R to start from equivalent states. Your proposal establishes that prospectively while keeping the historical-evidence limitation explicit.

### Freeze these gates before execution

| Gate | Required evidence |
|---|---|
| **Archived tick510** | Each independent reconstruction passes the unchanged full semantic comparator against the saved CTRL04 state. |
| **Replay to tick590** | Execute the same 80 hold ticks with the original observation/call order. Compare **all available archived per-tick fields**, including the recorded tick590 components, under existing tolerances. |
| **New S tick590 reference** | Before the first S-specific descent action, persist an independent full semantic snapshot, including controller and cached fields. Do not retain mutable array views that change as S advances. |
| **New R tick590** | Independently reconstruct the same prefix and hold; require its full semantic state to match the new S reference using the unchanged comparator. |
| **S execution-path check** | Match archived E80’s preclosure trace only through the interval where commands remain identical. Stop requiring equality when the deliberately different closing schedule begins. |

Capture both new tick590 snapshots at the **same execution phase**, before either condition-specific command. Add no extra `forward()`, rendering, reset, state correction or comparator exclusion.

### Report the two kinds of evidence separately

- **Historical tick590 agreement:** all available archived fields and recorded prefix traces match. Full historical-state equality is **not established**, because some fields were not saved.
- **CTRL05 branch equivalence:** the newly captured full S/R states pass the unchanged comparator before their interventions.

That is sufficient qualification for **this new S-versus-R comparison**; it does not reconstruct undocumented historical fields by assertion.

**Any mismatch still stops the experiment.** The reference/actuation eligibility checks remain mandatory. Record this timestamped amendment, retain the original CTRL05 clock and two-execution maximum, and proceed without another approval round if the gates pass.