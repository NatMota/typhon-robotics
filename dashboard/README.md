# M7 observation console

The private 13 October claim/media review is in [review/media_claim_manifest.json](review/media_claim_manifest.json). It qualifies eight archived records; none has an exact matching entry in the audited public index. The hosted site remains its labelled 9 October snapshot. The existing summary-object importer does not automatically ingest the newer row-list experiments; no deployment or schema conversion was performed.

The historical interface below is implementation documentation, not control-authority qualification.

A read-only browser dashboard for learned-policy evaluations and controller telemetry. It uses Python's standard library and plain HTML/CSS/JavaScript; there is no build step or package installation.

```bash
M7_EVIDENCE_DIR=/path/to/evaluation-runs \
M7_TRAINING_DIR=/path/to/training-watchers \
M7_LIVE_DIR=/path/to/live-observer \
python3 dashboard/server.py --port 8765
```

Open `http://127.0.0.1:8765/`. The server always binds to loopback. For a remote evaluation host, forward its loopback port through your existing SSH connection. The dashboard has no robot commands, training controls, public listener or third-party requests.

```bash
python3 dashboard/server.py --check
python3 dashboard/check_http.py http://127.0.0.1:8765
```

The check verifies manipulation/preservation separation, evidence path boundaries and the current inventory. The original installation defaults point to dated `/var/tmp` evaluation and watcher directories; set the three environment variables when using another dataset. `M7_LIVE_DIR` defaults to `M7_EVIDENCE_DIR/live`.

## Recorded evaluation layout

```text
evaluation-runs/
  <run-id>/
    summary.json
    checkpoint_load.json
    <scenario-id>/
      episode.json
      scenario.json
      trajectory.jsonl.gz
      initial.png
      final.png
      rollout.mp4                 # optional
```

`summary.json` is an object with `results[]`. Each result supplies `scenario_id`, `group`, `success`, `position_error_mm`, `yaw_error_deg`, `final_criteria` and `video_saved`. Other summary formats are ignored. A group of `completed` means the goal was already satisfied at the start: its successes are **goal preservation**, counted separately from manipulation. Model runs and episodes are discovered from these summaries.

`episode.json` supplies the recorded task, model/checkpoint provenance, start time, measured outcomes, criteria and errors. `scenario.json.config.evaluation` supplies the actual tolerances. Each gzip JSONL trajectory row supplies `step`, `requested_action`, `applied_action` and `truth`, including `sim_time_s`, `eef_position_m` and `gripper_joint_position_m`. The original pilots did not save arm q/dq; the UI states that limitation. The telemetry sample cursor is independent of video playback. Native video controls support playback and seeking via HTTP byte ranges.

## Live observer contract

Publish `latest.jpg` and `frame.json` atomically in the live directory. The image can contain the primary and secondary RGB views side by side. The optional producer also writes an equivalent `latest.json`; the dashboard reads `frame.json`.

```json
{
  "timestamp_unix": 1791583331.19,
  "timestamp_utc": "2026-10-09T22:02:11.190+00:00",
  "run_id": "example-run",
  "scenario_id": "example-case",
  "control_step": 52,
  "sim_time_s": 2.6,
  "phase": "observing",
  "status": "hold",
  "live": true,
  "image_file": "latest.jpg",
  "telemetry": {
    "source": "MuJoCo controller observation",
    "joint_names": ["joint1"],
    "joint_position_rad": [0.1],
    "joint_velocity_rad_s": [0.0],
    "joint_torque_nm": [0.02],
    "joint_torque_source": "MuJoCo qfrc_actuator, simulated actuator torque",
    "eef_position_m": [0.1, 0.2, 0.9],
    "gripper_joint_position_m": [0.04, -0.04],
    "gripper_open": true,
    "temperature_c": null
  },
  "command": {
    "applied_action": [0, 0, 0, 0, 0, 0, -1],
    "requested_action": [0, 0, 0, 0, 0, 0, -1],
    "units": "7 normalized OSC components in [-1,1]"
  },
  "guard": {
    "reason": "visual_uncertainty",
    "event": {
      "status": "hold",
      "complete_latched": false,
      "checker": {
        "schema_valid": false,
        "schema_error": "Invalid JSON",
        "assessment": {"status": "unknown"}
      }
    }
  }
}
```

The UI marks a sample live only when its capture timestamp is under 15 seconds old and `live` is not false. Finished or older frames remain visible with a **stale sample** badge. Producer finish messages must retain the original image capture time. Simulator joint coordinates and simulated actuator torque are labelled separately from policy commands and evaluator object truth. Missing motor temperature is stated as unavailable.

The latest observer run/case is labelled independently of the selected recorded evaluation. Observer guard output, including unknown or invalid checker output, is displayed without turning it into success. When `guard.event.checker.classifier` is present, the pane also shows its actual score, threshold, confidence type and model SHA256. Optional `checker.visual_stability` fields expose observed image-position drift and whether continuous yaw is available. A classifier score is not presented as a calibrated probability.

## Training watcher layout

The training root contains `watcher/watch-status.json` with `pods.*.jobs`, and `run3-watcher/status.json` with `jobs`. The parser accepts both flat jobs and nested run-number groups (for example `jobs.3` and `jobs.4`). Each job supplies `state`, `updates_seen` and optionally `expected_updates`; the watcher supplies `last_checked_epoch` or `heartbeat_epoch`. Explicit `expected_updates` always wins. For V1–V3, M3/M4 targets fall back to 1,000 updates and M5/M6 to 3,000. Other runs with no recorded target show pending, rather than an invented denominator. Watcher data older than 90 seconds is labelled stale. The Run 4 status comes from the Run 3 watcher's explicit hold state.

## API and source notes

`GET /api/state`, `/api/episode?run=…&case=…` and `/api/trajectory?run=…&case=…` read existing evidence. `/media/<run>/<case>/<file>` serves only named evidence files, with path boundaries checked. `/live/latest.jpg` serves the latest observer image. No API writes are implemented. Keep these data directories trusted; this small server is intended for one researcher through loopback/SSH.

The visual treatment was independently implemented from user-supplied photos of an AWS robotics event console. The exact demonstration product/source was not verified. Related projects inspected were [humanoid-robot-gui](https://github.com/cemalhekim/humanoid-robot-gui) (no explicit project license) and [neon-the-g1](https://github.com/cagataycali/neon-the-g1) (MIT). No code or assets were copied from either repository, and this console makes no AWS product affiliation claim.
