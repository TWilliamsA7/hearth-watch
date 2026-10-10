# Evaluating the monitor: simulations, recorded clips and live sessions

Every path below uses the same three parts:

- **Monitor:** the real `MonitorService` with the shipped `backend/hearth/monitor/config.json`. In offline runs its clock is taken from the records.
- **Scorer:** `hearth.eval.scoring.evaluate`.
- **Truth format:** `GroundTruth` JSONL.

Commands run from the repo root with `.venv\Scripts\python.exe`.

## 1. Simulated scenarios

```powershell
python -m hearth.eval batch --seeds 5                 # scripted scenarios
python -m hearth.eval study --count 200 --jobs 8      # random generated scenarios, broken down by factor
python -m hearth.eval study --count 200 --jobs 8 --edge-like   # observations shaped like the Jetson's
```

- Output goes to `recordings/eval/` (`summary.md` and `study/study.md`).
- `--set key=value` overrides a monitor threshold, and `--monitor-config file.json` swaps the whole config.
- Simulated posture features use the edge's formulas. `backend/tests/test_feature_parity.py` runs the real `edge/hearth_edge/features.py` next to the sim and requires matching numbers.
- `--edge-like` also matches the Jetson's message shape:
  - keypoints carry no confidence;
  - the face fields are set only on a face check every 0.5 s;
  - brief dropouts are skipped rather than sent as "no person".

## 2. Jetson clips

1. **Record on the Jetson** with `edge/tools/record_clips.sh` or `run_edge.py --record LABEL --dry-run`. Copy the clips to the laptop under `recordings/real/jetson/`. That directory is gitignored, and no video is ever stored.
2. **Make truth templates:**
   ```powershell
   python -m hearth.eval label-template recordings/real/jetson
   ```
   This writes one `<clip>.truth.jsonl` per clip: a single whole-clip segment labelled from the clip-name prefix. `lie_down` becomes `fall` with an alert expected. `stand`, `sit`, `bend` and `walk` expect no alert. Existing files are kept unless you pass `--force`.
3. **Time the segments by hand:**
   ```powershell
   python -m hearth.eval timeline recordings/real/jetson/lie_down-20261010-063944.jsonl
   ```
   The output prints every 0.5 s: the detector's posture state, the features, which floor rule is failing, and any events. Split the template into segments, for example `standing 0–4.2`, then `fall 4.2–end` with `expects_alert: true`.
   - `t_start` and `t_end` are seconds from the clip's first record.
   - `truth_label` is one of `walking, standing, sitting, bending, lying_intentional, fall, occluded, camera_outage, absent`.
   - `scenario_id` is the clip name and `seed` is 0.
4. **Score:**
   ```powershell
   python -m hearth.eval real recordings/real/jetson
   python -m hearth.eval real recordings/real/jetson --sweep floor_hip_y_min=0.40,0.45,0.50 --sweep floor_hold_seconds=2,3
   ```
   The output goes to `recordings/eval/real/` and contains:
   - per-clip reports and the events of each clip;
   - `summary.md`, with detection, misses, false alerts and delay per clip;
   - one table per sweep;
   - a **features-by-activity** table.

   Run the same command on `recordings/sim` to get the simulator's feature table for the same activities. Comparing the two shows where the simulator differs from the camera.

Clips without truth are still replayed, and their alerts are listed under "Not labelled yet".

## 3. Fall-model videos

```powershell
python edge/run_fall_model.py --video clip.mp4 --dry-run --record recordings/real/video/clip.jsonl
python -m hearth.eval label-template recordings/real/video/clip.jsonl   # then edit the times
python -m hearth.eval real recordings/real/video
```

- `--record` writes observations and roughly 1 Hz heartbeats, stamped on the video's media time.
- With `--dry-run --video` it runs as fast as inference allows.
- When a clip has model predictions, they decide its posture (`pose_fallback` only uses pose features for frames that have no prediction).

## 4. Live sessions

- **Preferred:** record while running live, with `run_edge.py --record LABEL` next to the normal backend stream. You can then replay the session offline as in section 2, which also lets you re-score it with different thresholds.
- **Score what the live backend actually did:**
  ```powershell
  curl http://127.0.0.1:8000/events/export?source=jetson-01 > recordings/real/live/session.events.jsonl
  python -m hearth.eval score --start 2026-10-10T14:00:00Z --truth recordings/real/live/session.truth.jsonl --events recordings/real/live/session.events.jsonl --source jetson-01
  ```
  `--start` is the wall-clock time that the truth file's `t = 0` refers to.
- **Simulated scenarios against the live app:** `/sim` and `/eval` are mounted on the main app.
  - Start a scenario with `POST /sim/load` and `POST /sim/play`. It feeds the same `MonitorService` as the camera, so its alerts appear on the nurse desk.
  - The after-action report is at `GET /eval/runs/{run_id}`.

## What these numbers are

- **Staged clips** from one camera position test *this setup*. They are not clinical accuracy.
- **Simulated results** test the rules against synthetic bodies.

Report both, and label each one for what it is.
