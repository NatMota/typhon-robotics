# Consumer interpretation: failed LIVE01-T candidate

Use `source_contract` only as a diagnostic append-only record. Existing numerical payloads remain unchanged.

| Information | Interpretation |
|---|---|
| Latest policy target | Historical proposal with query/waypoint identity; active only when explicitly selected as current command source |
| Applied command | Workspace-adjusted normalized OSC at its recorded control tick; during settling, identify settling rather than the retained policy target |
| Actuator control / generalized force | Separate simulator quantities; neither represents a measured hardware torque sensor |
| Control and policy clocks | Policy elapsed clock has origin at control tick 40; it includes subsequent settling and is not the actor's action counter |
| RGB frame acquisition | Unqualified in this candidate: every tested observation reference is unresolved; render logs alone do not identify the copied policy images |
| Publication time | Payload assembly/writing time; never use as camera acquisition time |
| Integrated joints / cached wrist pose | Preserve distinct source and computation-phase labels; exact simultaneity is unknown |
| Held-state estimate | Passive diagnostic only; object pose, orientation and completion remain unknown; no veto or phase-advancement authority |
| Repeated/delayed data | Retain source ordering, identity and unknown ages; publication does not make old evidence new |

The consumer fixtures pass their narrow contract, but the real copy-boundary failure prevents reliable camera age/identity use. No freshness threshold or stop policy is qualified.
