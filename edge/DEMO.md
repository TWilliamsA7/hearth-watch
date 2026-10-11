# Demo runbook: live camera part (Jetson edge)

Covers the live-camera part of the README's 90-second demo: the camera on a consenting
teammate with pose skeleton and face presence, then a safe lie-down on a mat that produces a
**possible-fall** alert, and recovery when they stand up. All commands run on the laptop from the
repo root unless marked Jetson.

> Status 2026-10-10 ~07:40: works live end to end. Lying across the mat for about 3 s raised a
> possible-fall alert in the team backend; standing up recorded recovery. The backend thresholds
> are calibrated for the camera position below (floor head/hip lines 0.40/0.45): if the camera
> moves, re-check with `tools/record_clips.sh` + `tools/check_rules.py` before relying on it.
>
> Update 2026-10-10 evening: our own posture model ([POSTURE_MODEL.md](POSTURE_MODEL.md)) is now on
> by default and is what the backend's alerts use; it doesn't depend on those height thresholds,
> only on seeing the whole body. The head/hip thresholds only apply with `--no-posture-mlp`.

## Before the demo (about 15 minutes ahead)

1. **Power and link.** Jetson on a real 5 V 3 A supply (not a laptop port); USB cable to the
   laptop. Check: `ssh jetson true` returns with no error.
2. **Camera position** (what the thresholds were calibrated for). About 1.5-2 m high, ~3.5 m
   from the mat, tilted down 10-15 degrees, so a standing person fits head to feet and a person
   lying on the mat fits side to side. Aim with the live view:
   ```sh
   edge/tools/view_camera.sh
   ```
   Standing: whole skeleton, posture numbers in the panel. Lying sideways **across** the picture
   on the mat: an amber (rotated) skeleton over the whole body. Close the window when done.
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

It checks the camera (and stops a leftover backend from an earlier run), copies the latest edge
code to the Jetson, starts the team backend on port 8000, opens the video window and starts the
Jetson. Then open in a browser:

- `http://127.0.0.1:8000/nurses`: choose `room-01` and Possible falls
- `http://127.0.0.1:8000/sources`: live Jetson telemetry

Our posture model runs by default: the backend gets its upright / sitting / floor call on
every observation and the panel shows it for everyone in view (see
[POSTURE_MODEL.md](POSTURE_MODEL.md)). Keep the whole body in view (no head-only close-ups).

## During the demo

1. **Live camera.** The window shows the skeleton, a red face box when a face is found, and the
   info panel: FPS, keypoints, the posture numbers being sent, then the **backend's** view
   (posture, alerts). Say: the Jetson sends keypoints and numbers only, no video; face detection
   means "a face is visible", never who it is.
2. **Lie-down.** Teammate lowers slowly onto the mat, lying **across** the picture, and stays
   still. After about 3 s at floor level the panel shows `posture fallen` and a red
   `ALERT possible fall: open N s`, and the alert appears on `/nurses`.
3. **Recovery.** Teammate stands up. The backend records recovery: the panel line turns amber,
   `... up again after N s - review on /nurses`, and the nurse card says "Upright posture
   observed afterward". The alert stays open on purpose (the person may still be hurt) until a
   nurse acknowledges/resolves it on `/nurses`; then the panel line disappears.
4. **Silence is not safety** (optional). Press Ctrl+C in the terminal: about 3 s later the
   backend raises `source_offline`. Press Enter to stop the backend afterwards.

## Fallback: replay a recorded fall (verified)

If the live camera can't be used, replay recorded keypoints into the backend. It is labelled
`source_type: replay`, source `replay-01`, so **say it is a replay**. With the backend running
(the `live_test.sh --real` session, or `.venv/bin/python -m uvicorn hearth.api.app:app
--app-dir backend --host 0.0.0.0 --port 8000`), in another terminal:

```sh
.venv/bin/python edge/tools/replay.py --join recordings/demo/stand.jsonl \
    recordings/demo/lie_down.jsonl recordings/demo/stand.jsonl
```

About 30 s: standing, then the possible-fall alert at ~13 s, then recovery at ~20 s, all in one
run. Verified 2026-10-10 07:57 against the real backend at normal speed (keep `--speed 1`:
faster playback shortens the time on the floor). The clips live only on this laptop
(`recordings/` is not in git).

## If something goes wrong

| Problem | Fix |
|---|---|
| "camera is in use by ..." | Stop that program (Ctrl+C where it runs), or `ssh jetson pkill -INT -f run_edge.py` |
| "port 8000 ... in use by pid N" | A backend you started yourself is running: use it, or stop it (`kill N`) |
| No video window / frozen video | Close it and run `edge/tools/view_camera.sh --player-only` |
| Backend shows the Jetson OFFLINE | Check `ssh jetson true`; the backend must listen on `0.0.0.0:8000` |
| Posture stays `unknown` | Whole body not in view: move the camera back/up (step 2 above) |
| No alert while lying | Lie **across** the picture with the whole body in view; feet toward the camera can't be detected |
| Alert line keeps showing after getting up | Expected: resolve it on `/nurses` |
| Live detection fails entirely | Use the replay fallback above |

## What to claim (and not)

- Alerts are **possible** safety events for human review, not diagnoses or certified fall
  detection. Report measured numbers as small-sample tests.
- Face detection is not identification. No video is stored or sent by default.
- Never ask a judge to fall. Never imply that lying down proves a fall.
