# Demo runbook: live camera part (Jetson edge)

Covers the live-camera part of the README's 90-second demo: the camera on a consenting
teammate with pose skeleton and face presence, then a safe lie-down on a mat that produces a
**possible-fall** alert. All commands run on the laptop from the repo root unless marked Jetson.

> Status 2026-10-10 ~06:00: camera, pose, face presence, backend link and offline detection all
> work on the real hardware. A live **possible-fall alert has not yet fired** with the real
> camera: it needs the camera position below and likely a backend threshold tweak
> (see "Known limits" in README.md). Re-check with `tools/check_rules.py` before relying on it.

## Before the demo (about 15 minutes ahead)

1. **Power and link.** Jetson on a real 5 V 3 A supply (not a laptop port); USB cable to the
   laptop. Check: `ssh jetson true` returns with no error.
2. **Camera position.** About 1.5-2 m high, ~3.5 m from the mat, tilted down 10-15 degrees, so a
   standing person fits head to feet. Aim with the live view:
   ```sh
   edge/tools/view_camera.sh
   ```
   Standing: whole skeleton, hip dot **above** the cyan line. Lying sideways **across** the
   picture on the mat: head and hip dots **below** the line. Close the window when done.
3. **Clock.** Required if the Jetson has no internet at the venue:
   ```sh
   edge/tools/sync_clock.sh          # asks for the Jetson password only if it needs to correct
   ```
4. **Health check** (optional): `ssh jetson 'bash ~/hearth-watch/edge/tools/check_jetson.sh'`
5. **Fan** (optional, long sessions in the case): 
   `ssh -t jetson "sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'"`
6. **Team backend installed** once: `.venv/bin/pip install -e 'backend[dev]'`

## Start

```sh
edge/tools/live_test.sh --real
```

It checks the camera, copies the latest edge code to the Jetson, starts the team backend
(port 8000), opens the video window and starts the Jetson. Then open in a browser:

- `http://127.0.0.1:8000/nurses`: choose `room-01` and Possible falls
- `http://127.0.0.1:8000/sources`: live Jetson telemetry

## During the demo

1. **Live camera.** The window shows the skeleton, a red face box when a face is found, and the
   info panel: FPS, keypoints, the posture numbers being sent, then the **backend's** view
   (posture, alerts). Say: the Jetson sends keypoints and numbers only, no video; face
   detection means "a face is visible", never who it is.
2. **Lie-down.** Teammate lowers slowly onto the mat, lying **across** the picture, and stays
   still for 5 s. The backend needs about 3 s at floor level: the panel turns
   `posture fallen`, then `ALERT possible fall: open`, and the alert appears on `/nurses`.
   Acknowledge it there.
3. **Recovery.** Teammate stands up: the backend records recovery.
4. **Silence is not safety** (optional). Press Ctrl+C in the terminal: about 3 s later the
   backend raises `source_offline`. Press Enter to stop the backend afterwards.

## If something goes wrong

| Problem | Fix |
|---|---|
| "camera is in use by ..." | Stop that program (Ctrl+C where it runs), or `ssh jetson pkill -INT -f run_edge.py` |
| No video window / frozen video | Close it and run `edge/tools/view_camera.sh --player-only` |
| Backend shows the Jetson OFFLINE | Check `ssh jetson true`; the backend must listen on `0.0.0.0:8000` |
| Posture stays `unknown` | Whole body not in view: move the camera back/up (step 2 above) |
| No alert while lying | Lying toward the camera, or above the cyan line: lie across, closer to the bottom edge |
| Live detection fails entirely | Replay a recorded clip **that `check_rules.py` confirmed alerts**: `.venv/bin/python edge/tools/replay.py recordings/<clip>.jsonl` and say it is a replay (it shows as `replay-01`) |

## What to claim (and not)

- Alerts are **possible** safety events for human review, not diagnoses or certified fall
  detection. Report measured numbers as small-sample tests.
- Face detection is not identification. No video is stored or sent by default.
- Never ask a judge to fall. Never imply that lying down proves a fall.
