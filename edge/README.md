# Hearth edge (Jetson Nano)

The edge node watches one room with one USB camera and sends **observations** to the team
backend: pose keypoints, posture numbers and face presence, about 10 per second, plus a
heartbeat each second. It never decides "fall": the backend's monitor does, for live, simulated
and replayed data alike. No raw video leaves the Jetson; the optional video preview only goes to
the laptop's window over the USB link and is never recorded.

For running the demo, see **[DEMO.md](DEMO.md)**. Measured speed/memory: **[BENCHMARKS.md](BENCHMARKS.md)**.
Our own posture model (upright / sitting / floor, trained in `pc/`, **on by default** and the
model the backend's alerts use): **[POSTURE_MODEL.md](POSTURE_MODEL.md)**.

## Hardware and software

- Jetson Nano 2GB, JetPack 4.6.1 (L4T R32.7.1), stock Python 3.6, headless, 5 GB swap.
  Set up with `tools/setup_jetson.sh`, checked with `tools/check_jetson.sh`.
- Logitech C270 USB webcam at 640x480 MJPEG, 15 FPS requested.
- Laptop and Jetson talk over the USB cable: Jetson `192.168.55.1` (user `g`), laptop
  `192.168.55.100`. Works with no internet.
- Pose: jetson-inference poseNet `resnet18-body` (TensorRT). Face: OpenCV DNN `res10`
  (`tools/get_face_model.sh`), presence only. Optional `--face-id`: SFace (TensorRT) face
  templates for opted-in names on the laptop, see below. Identity is never decided here.
- Edge code is standard-library Python 3.6 (plus the JetPack-provided `jetson_inference`,
  `jetson_utils`, `cv2`, `numpy`). Laptop tools and tests use the repo's `.venv` (Python 3.11+).

## How it works (one frame)

```
USB camera (Logitech C270, MJPEG 640x480, ~13 FPS)
  -> poseNet resnet18-body (TensorRT): every person's keypoints              pose.py
  -> rotated poseNet passes for people who look lying / incomplete           pose_rotate.py
  -> tracker: the person the edge follows (a passer-by can't take over)      features.py
  -> everyone in view gets an anonymous id: p1, p2, ...                      people.py
  -> posture numbers for each person: torso angle, head/hip height,
     box shape, hip velocity (joints missing <= 0.3 s bridged)               features.py
  -> our posture model on everyone: upright / sitting / floor                posture_mlp.py
  -> face check on the followed person (res10, background, every 0.5 s)      face.py
  -> one Observation per person (shared contract), ~10/s, batched over USB   contract.py, client.py
     + heartbeat every second (FPS, model time, memory)
  -> optional: video preview to the laptop (skeleton + info panel)           overlay.py
```

The backend (Person 2) decides what is a possible fall. With our posture model's prediction on
every observation it uses the model's label (3 s of "fallen" -> possible_fall; details in
[POSTURE_MODEL.md](POSTURE_MODEL.md)); with `--no-posture-mlp` it uses its pose rules on the
posture numbers instead.

## Run it

```sh
# On the Jetson (from ~/hearth-watch/edge; the laptop tools copy edge/ there for you):
python3 run_edge.py                      # send to the backend at http://192.168.55.100:8000
python3 run_edge.py --dry-run            # print observations as JSON, send nothing
python3 run_edge.py --stream rtp://192.168.55.100:1234   # + video preview for the laptop
python3 run_edge.py --no-posture-mlp     # without our posture model (on by default, POSTURE_MODEL.md)
python3 run_edge.py --single-person      # only the followed person, no person_id (the old behaviour)
python3 run_edge.py --help               # all options (face, rates, recording, ...)
```

Most of the time, use the one-command laptop tools below instead.

## What it sends (shared contract, schema v1)

- `POST /ingest/observations`: batches of `Observation`, ~10/s **per person in view**. Each
  carries that person's `person_id` (`p1`, `p2`, ...: anonymous tracks, not identities; an id
  survives up to 1 s out of view, then the person gets one `person_detected: false` observation
  and is forgotten). With nobody in view, one observation without a `person_id` says so.
  `--single-person` sends only the followed person, without `person_id`. Keypoints are normalized to the
  frame and sent **without** confidence (poseNet measures none; null = unknown). Posture
  features: `torso_angle_deg` (0 upright, 90 lying), `head_y`, `hip_y` (0 = top of frame),
  `bbox_aspect` (only when part of the torso is visible), `vertical_velocity` (+ = moving down).
  Anything not measurable is omitted, never guessed.
- `posture_prediction` from our posture model on **every** observation, for that observation's
  person (`model_id` `hearth/posture-mlp`, trusted by the team backend's config). With `--no-posture-mlp
  --fall-model`: the Hugging Face model's prediction instead.
- `face_present` is true/false on frames where a face check ran (about every 0.5 s, on a
  background thread) and null on all others ("did not run", not "no face").
- `POST /ingest/heartbeat` every second with FPS, model time and memory, **only while camera
  frames arrive**: a dead camera goes silent so the backend shows it offline.
- If the backend is unreachable, the newest ~10 s are buffered and sent on reconnect; a message
  the backend rejects is dropped and logged (that means a contract mismatch).

## Code (`hearth_edge/`)

| File | Does |
|---|---|
| `config.py` | All settings (backend URL, camera, rates, face, preview) |
| `pose.py` | Camera + poseNet; skeleton drawn only after the face check copied the frame |
| `features.py` | Posture numbers from keypoints (pixel-space angles, smoothed hip velocity) |
| `contract.py` | Builds Observation/Heartbeat dicts matching `shared/` exactly |
| `client.py` | Batched sending, heartbeats, retry/backoff, bounded buffer |
| `face.py` | res10 face presence with head close-up, on a background thread |
| `recorder.py` | JSONL recordings (keypoints/numbers only, never video) |
| `overlay.py`, `backend_view.py` | Preview info panel (long lines wrap to the frame width); the backend's own posture/alert view |
| `pose_rotate.py`, `control.py` | Rotated poseNet passes for lying people (`RotationPolicy` for the followed person, `OthersRotation` for everyone else); stdin commands for sessions |
| `posture_mlp.py`, `posture_mlp.npz` | Our posture model (on by default): numpy network + trained weights, see [POSTURE_MODEL.md](POSTURE_MODEL.md) |
| `face_id.py` | Optional face templates (`--face-id`): SFace only for a face in someone's head box, aligned with poseNet's eyes and nose; templates (never frames) to the laptop |
| `fall_model.py`, `fall_decode.py`, `fall_rules.py` | Optional Hugging Face fall model on TensorRT (`--fall-model`), see below |

## Laptop tools (`tools/`)

| Command | Does |
|---|---|
| `tools/view_camera.sh` | Continuous camera view with skeleton, face boxes and info panel |
| `tools/live_test.sh --real` | Team backend + Jetson + video window in one command |
| `tools/live_test.sh` | Same with the stand-in `stub_backend.py` (no team backend needed) |
| `tools/sync_clock.sh` | Copy the laptop's clock to the Jetson (needed when it is offline) |
| `tools/record_clips.sh stand sit lie_down` | Guided clip recording with a live camera view |
| `tools/check_rules.py <clip>.jsonl` | Runs the backend's real FallDetector on a clip and explains each failed condition |
| `tools/replay.py <clip>.jsonl` | Re-sends clips as `source_type: replay`, `source: replay-01`, new `run_id` per pass; `--join` plays several as one run (demo fallback) |
| `tools/face_test.sh`, `tools/pose_test.sh` | Guided probes (face detector; poseNet rotation) |
| `tools/bench.py` (Jetson) | Speed/memory table for BENCHMARKS.md |
| `tools/get_face_id_model.sh` (Jetson) | Download SFace from Hugging Face (the laptop's pinned file and hash), build its TensorRT engine |
| `tools/face_id_agreement.py` (laptop) | Check that three-keypoint alignment still matches a laptop enrollment on real photos |
| `tools/get_fall_model.sh` (Jetson) | Download the ONNX fall model from Hugging Face, check its hash, build the TensorRT engine |

## Tests

```sh
.venv/bin/python -m pytest edge/tests     # on the laptop; validates against the real contract
```

## Known limits

- Works live end to end (2026-10-10): lying across the mat raises a possible-fall alert after
  ~3 s and standing up records recovery (see DEMO.md, BENCHMARKS.md).
- Our posture model (default) only needs the whole body in view; a head-only close-up is its
  one known false "floor" (POSTURE_MODEL.md, pc/RESULTS.md).
- Without it (`--no-posture-mlp`), the backend's pose rules use height thresholds calibrated for
  one camera position (about 1.5-2 m high, ~3.5 m from the mat, tilted down): head/hips below
  0.40/0.45 = floor, hips above 0.65 = upright. If the camera moves, re-check with
  `tools/record_clips.sh` and `tools/check_rules.py`.
- People lying down: plain poseNet often finds only the head and shoulders, so when the upright
  pose looks lying or incomplete a rotated poseNet pass is added (`pose_rotate.py`; lying
  across went from 0% to 100% head+shoulders+hips). `--no-rotation` turns it off. Lying with the
  feet **toward** the camera still hides the hips and foreshortens the body (torso ~19 degrees):
  a single-camera limit rotation cannot fix.
- Everyone in view is reported separately, so the backend watches each person (their own floor
  timer and alert). Ids follow the nearest box from frame to frame: two people crossing closely,
  or someone leaving and coming back, can swap or get a new id. With `--single-person` the edge
  follows one person, and if someone else is on the floor the backend is told about them instead
  (POSTURE_MODEL.md).
- Face detection at 3 m and with a covered mouth is not yet measured.

## Optional: face templates for opted-in names (`--face-id`)

```bash
bash ~/hearth-watch/edge/tools/get_face_id_model.sh      # once: SFace -> models/face-id/sface_fp16.engine
python3 run_edge.py --face-id --face-id-token-file ~/.hearth-face-token
python3 run_edge.py --face-id --dry-run                  # logs when a template would be sent
```

- **When SFace runs.** The face check (res10, every 0.5 s) takes each person's head box in turn. Only when it finds a face there, and poseNet has that person's eyes and nose at least 20 px apart (`FACE_ID_MIN_EYE_PX`), does SFace run (GPU, FP16), aligned with those three points.
- **What is sent.** The 128-value template with the person's anonymous `person_id`, posted on its own thread (newest per person) to the laptop's `/face-demo/edge-embeddings` with `HEARTH_EDGE_FACE_TOKEN`. **No frames or face crops leave the Jetson.** The laptop matches the template against opted-in enrollments and names that person's alerts. The edge never learns the name.
- **What doesn't change.** Monitoring never waits on it. A missing engine, a missing token or `--single-person` logs a reason and carries on unnamed. While on, `face_present` still goes on the followed person's observation, but it reports whichever person's head was checked.
- **Measured on the Nano (2026-10-11):** TensorRT 8.2 builds SFace FP16 in about 6 min (a 20 MB engine). It runs at 17.5 ms per face of GPU compute with trtexec (57/s). This is only paid when a face is in someone's head box, at most once per 0.5 s face check. **Not yet measured:** FPS and memory alongside poseNet and the posture model.

## Optional: fall model on the Jetson (YOLO11n + TensorRT, `--fall-model`)

Person 2's standing / sitting / fallen model (`melihuzunoglu/human-fall-detection` @ `97261b0`)
running on the Nano itself, without PyTorch: exported to ONNX on a PC (`pc/`, published as
`Lunar255/hearth-fall-yolo11n-onnx` @ `12dc1ac`) and built into a TensorRT FP16 engine on the
Jetson. Off by default.

```sh
# On the Jetson, once (~10 min; downloads ~10 MB, checks sha256, builds models/fall/best_416_fp16.engine):
~/hearth-watch/edge/tools/get_fall_model.sh
python3 run_edge.py --fall-model --dry-run
# Laptop: edge/tools/view_camera.sh --fall-model (panel shows its label), live_test.sh --real -- --fall-model
```

- A background thread checks a copy of the frame about every 0.3 s (~25 ms on the GPU at 416),
  so the camera loop never waits. Results go through a Python 3.6 port of Person 2's rules
  (`fall_rules.py`, tested against their real function), so the Jetson's `posture_prediction`
  means exactly what the laptop runner's does.
- The latest result (at most 1 s old) goes on **every** observation. The backend restarts its
  floor timer whenever observations switch between model and pose evidence, so attaching it to
  only some frames would never let the 3 s hold finish.
- If poseNet loses the person but the model sees them, the observation says
  `person_detected: true` with the prediction and no keypoints, so a lying person poseNet
  can't find still counts.
- Now panel-only by default: our posture model is what the backend trusts (POSTURE_MODEL.md).
  To send the HF model instead: `--no-posture-mlp --fall-model`, with a backend config whose
  `fall_model_id` is the HF model's.
- With a prediction present, the backend trusts the model over the pose rules (see below), so
  alerts depend on the model's call. If the engine is missing, the edge logs it and keeps sending
  pose evidence only.

## Optional: laptop standing / fallen model (YOLO11, Person 2)

`run_fall_model.py` runs [melihuzunoglu/human-fall-detection](https://huggingface.co/melihuzunoglu/human-fall-detection), a YOLO11 object detector with **Standing**, **Sitting**, and **Fallen** classes. Inference runs locally; only structured JSON goes to the existing Hearth API. It does not send or save frames. The checkpoint is pinned by revision and SHA-256 in `backend/hearth/perception/model.json`.

### Laptop setup (Python 3.11+)

Run from the repository root after setting up the backend:

```powershell
# CPU builds avoid installing CUDA packages on the laptop.
.\.venv\Scripts\python.exe -m pip install 'torch>=2.2,<3' 'torchvision>=0.17,<1' --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e './backend[vision]'
.\.venv\Scripts\python.exe edge/run_fall_model.py --download-only
.\.venv\Scripts\python.exe edge/run_fall_model.py --check
```

The weights are about 5.5 MB; PyTorch and other dependencies are larger. Downloads happen during setup. Normal inference uses the verified local weights and disables Ultralytics synchronization/automatic installation. `--check` runs the actual checkpoint on a blank frame without opening a camera; this tests runtime compatibility, not fall accuracy.

### Camera

Restart the backend after updating the shared schema, then run:

```powershell
.\.venv\Scripts\python.exe edge/run_fall_model.py --camera 0 --backend http://127.0.0.1:8000 --room room-01
```

The camera opens only when `--camera` is explicitly selected. Stop with Ctrl+C. Use `--preview` to opt into a local annotated OpenCV window (Q exits); no video is recorded. Use `--device cpu` (default), or a locally supported device. `--max-fps` defaults to 5; actual throughput depends on the device. `--max-frames 30` limits a test run. Match the backend port if you are using the preview on 8001.

Subscribe to **Possible falls** in `/nurses`. The source-health section and `/room` show the current model label and score. After a default 3 seconds of sufficiently confident Fallen observations, the backend creates one possible-fall alert. One second of sufficiently confident Standing observations records recovery. Sitting, no detections, multiple detections, low confidence, and stale sources cannot confirm recovery. The model confidence score is not a calibrated probability of a real fall.

### Existing image or video

```powershell
# One image produces one observation; it cannot satisfy a temporal alert hold.
.\.venv\Scripts\python.exe edge/run_fall_model.py --image path/to/image.jpg --dry-run
# Local recordings use video-frame time and are explicitly labelled replay.
.\.venv\Scripts\python.exe edge/run_fall_model.py --video path/to/recording.mp4 --backend http://127.0.0.1:8000
```

Use `--video-fps` only to correct missing/incorrect recording FPS metadata. Replay samples are selected at approximately `--max-fps`; the media timestamps, not processing speed, drive the hold. Every run gets a fresh run ID. `--dry-run` prints observations without sending heartbeats or observations to the backend.

### Contract and limits

- `Observation.posture_prediction` carries the model ID/revision, label, raw score, normalized bounding box, detection count, and uncertainty reason. It does **not** fabricate head/hip locations, pose keypoints, or face presence.
- Numeric class IDs are mapped using the checkpoint's own class names. An unexpected label set stops model loading. Multiple boxes produce an unknown result rather than selecting one person and possibly declaring another person recovered. This remains a one-person-at-a-time detector, without identity tracking.
- The backend retains its original pose rules for observations without a model prediction. If a prediction is present, that evidence path is authoritative; weak model output does not silently fall back to conflicting pose features. Switching models/evidence paths resets pending holds.
- Monitor thresholds: `model_confidence_min` (default 0.65), `floor_hold_seconds` (3), `model_standing_hold_seconds` (1), and `max_observation_gap_seconds` (1) in `backend/hearth/monitor/config.json`. Tune on staged traces. If measured frame intervals exceed the gap limit, the hold will correctly reset; adjust only after measuring the source.
- The detector has no pose/face output, so it cannot by itself drive the raised-arm help gesture or facial detection. Those still need the separate pose/face pipeline. Existing help-button functionality is independent.
- Raw images remain on whichever device runs this process. Laptop camera use is a local development/demo option, not a claim that Jetson inference is working.
- This Python 3.11+ Ultralytics runner is **not compatible with stock JetPack Python 3.6**. Jetson Nano deployment requires verifying a compatible PyTorch/Ultralytics environment or a tested exported runtime; no Nano performance is claimed.

Model source and attribution: Melih Uzunoğlu, `melihuzunoglu/human-fall-detection`, revision `97261b03632a9715a27481b2cfdac6a78e26d2d9`. The model repository declares AGPL-3.0. [Ultralytics prediction documentation](https://docs.ultralytics.com/modes/predict/) describes the result interface used by the adapter. This integration does not establish clinical accuracy.

## Laptop model pipeline

See [MODEL_RUNNER.md](MODEL_RUNNER.md) for the Hugging Face posture/hand models and OpenCV contrast retry. These run on a laptop with camera-frame access. The backend accepts model predictions and calibrated Jetson pose observations through the same ingestion pipeline.
