**Choose (2): retain the frozen teacher’s actually active checks. Do not attach `M7 Guard.measured()` to either condition.**

My phrase “existing safety-limit violation” meant a termination condition **active in that execution path**, not a separate guard elsewhere in the repository. Adding it only to E80 would introduce a second intervention and prevent testing the dwell.

MuJoCo’s joint limits are solver constraints, not hard position clipping; the documented constraint residual can be negative when a limit is exceeded. The reported overshoot is therefore a **real recorded range excursion**, but not automatically an execution-termination event under this teacher’s existing contract. [MuJoCo](https://mujoco.readthedocs.io/en/2.3.7/computation.html)

### Freeze this clarification before execution

- **Keep tick510 and both starting states unchanged.** Preserve the `−0.0001030628 rad` lower-limit margin explicitly.
- **Keep joint ranges, solver settings, gains and all active checks unchanged.** Add no tolerance, clamp, guard or state correction.
- **Log signed joint-limit margins throughout both conditions**, including maximum excursion and duration outside range. Range exceedance alone is not a newly introduced abort criterion.
- **Retain E80’s predeclared hold-specific stops:** unintended gripper–brick contact or tipping during the added hold, plus the existing runtime termination conditions and nonfinite/numerical-failure stops.

A successful task outcome must still disclose any joint-limit excursion. It would **not** establish joint-safe execution, qualify a hardware deployment, or make the separate zero-tolerance guard pass.

**Proceed with E0 and E80 under the original 45-minute clock and two-execution maximum.** Save this timestamped protocol clarification; no further scope change is needed.