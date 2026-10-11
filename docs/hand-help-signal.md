# Hand help signal

Hearth runs `SuperSura/hand-gesture-yolo26n` locally alongside posture inference on the same webcam frames. Both send structured observations to the existing monitor; frames are not saved or uploaded. The published model has no help class, so Hearth assigns help meaning to a deliberate held gesture.

Default signal: **one closed fist toward the camera, held for two seconds**, with confidence >= 0.65. Subscribe to **Help requested** for room-01 in the nurse page. After one alert, continued fist detections do not create duplicates. Show a different recognizable gesture, such as an open palm, for one second to release the latch; a ten-second cooldown from the last request also applies. Occlusion/no detections/low scores do not rearm it. Multiple detected hands are uncertain. This is a one-person demo without identity or persistent hand tracking; large box-center jumps reset pending holds.

Configuration in `backend/hearth/monitor/config.json`: `hand_help_label` (`fist`, `palm` or `call`), `hand_gesture_confidence_min`, `hand_gesture_max_center_shift`; existing `gesture_hold_seconds`, `gesture_release_seconds`, `gesture_cooldown_seconds`, and `max_observation_gap_seconds` apply. Hand predictions are authoritative for help-gesture detection when present; pose-raised-arm logic remains available for observations without them. A visible hand can request help even if posture inference misses the body.

From the repository root in PowerShell:

```powershell
# Download verified weights (already downloaded on Sam's laptop).
.\.venv\Scripts\python.exe edge/run_fall_model.py --download-only --hand-gestures

# Start Hearth with this webcam (stop any other camera runner first):
.\scripts\start.ps1 -Camera 0
```

The webcam client runs with `--hand-gestures` and shares its annotated frames with the dashboard's **Live camera** feed. If you use another port (`-Port`), the camera client follows it. Terminal output now includes `hand: fist score=...` and the preview draws hand boxes. Test a brief fist (no alert), held fist (one alert), continued fist (no repeats), release/cooldown, and another held fist. Only claim performance after testing the real camera and combined-model latency; frame gaps over one second reset pending holds. Current published labels/weight compatibility were verified by loading the actual checkpoint and blank-frame CPU inference; actual gesture accuracy has not been verified on a camera.

Pinned revision: `26cab531b83cdc9f09808a73c0dc485e35920c6b`; SHA-256 `5695ba86a2a6e37994d1e9ad49a6e901009b02ae219a12be6c9272cc65f4abc9`. Source: https://huggingface.co/SuperSura/hand-gesture-yolo26n (repository declares Apache-2.0). The JSON/TypeScript contracts include `Observation.hand_gesture_prediction`; no numeric class IDs are hardcoded. Alert metadata includes label, model revision, score and held duration. Existing nurse routing, acceptance, resolution and room subscriptions apply.
