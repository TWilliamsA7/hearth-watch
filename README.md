# Hearth

Resident safety monitoring, help requests, and care-team response simulation, built for Knight Hacks IX (October 9–11, 2026, 36 hours, team of 4).

Hearth shows how local edge computer vision could help caregivers in assisted-living and memory-care settings notice possible safety incidents involving older adults. A Jetson Nano processes one USB camera and sends structured posture and face-presence observations to a laptop. The laptop runs the monitoring pipeline, a deterministic room simulator, and a dashboard. Live and simulated observations go through the same pipeline, and simulated scenarios are scored against their ground truth.

**Hearth is not a game.** It is a care-monitoring simulation and evaluation environment. The primary hackathon target is the Full Sail challenge (a serious simulation using face detection and/or pathfinding). Face detection is the Full Sail feature; pose-based possible-fall detection is the safety application.

## Status

The Person 2 monitoring core is implemented: shared schemas and generated TypeScript types, FastAPI ingestion and WebSocket, possible-fall rules, alert lifecycle, source health, an in-memory event log with JSONL export, a fake-observation producer, and automated tests. It now includes held-arm help requests, a manual help button, nurse subscriptions, exclusive nurse acceptance, and built-in nurse/room pages. Bed/chair zone downgrading is available but disabled until zones are configured. Edge inference and the simulator/evaluator remain separate integrations; Person 4 can reuse these APIs in the planned React dashboard.

Detection thresholds are initial demo parameters, not tuned or validated on real camera traces. Synthetic tests exercise software behavior; they are not measured clinical accuracy.

## Install the models and test on a laptop (PowerShell)

The laptop runner uses two local models on the **same webcam frames**:

| Model | Purpose | Local weights |
| --- | --- | --- |
| [`melihuzunoglu/human-fall-detection`](https://huggingface.co/melihuzunoglu/human-fall-detection) | Standing, Sitting, Lying (checkpoint label `Fallen`) | `models/human-fall-detection/best.pt` |
| [`SuperSura/hand-gesture-yolo26n`](https://huggingface.co/SuperSura/hand-gesture-yolo26n) | A held **fist** requests help | `models/hand-gesture/gesture.pt` |

Run all commands below from the repository root with Python 3.11+. Models and dependencies need internet during installation; inference runs locally afterward. The downloader pins each model revision and verifies its SHA-256. Weight files are git-ignored; teammates download them rather than committing them. Camera frames are not saved or uploaded.

### 1. Install dependencies and download weights

If `.venv` already exists, reuse it; do not create a second environment. Activation is optional because these commands use the environment's Python explicitly.

```powershell
# Create this only on a fresh checkout without .venv:
python -m venv .venv

# CPU PyTorch for the laptop:
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ./shared/python -e './backend[dev,vision]'

# Download both models (safe to rerun; existing files are verified):
.\.venv\Scripts\python.exe edge/run_fall_model.py --download-only --hand-gestures

# Load the actual checkpoints and run blank-frame inference; no camera opens:
.\.venv\Scripts\python.exe edge/run_fall_model.py --check --hand-gestures
```

The check should print posture labels `fallen`, `sitting`, `standing` and hand labels including `fist`. No detections on a blank frame is expected. This verifies runtime compatibility, not camera accuracy. Installation of PyTorch is larger than the model weights.

### 2. Start or restart the backend — terminal 1

Stop the previous backend with Ctrl+C first. Restarting is required after shared-schema or configuration changes.

```powershell
.\.venv\Scripts\python.exe -m uvicorn hearth.api.app:app --host 127.0.0.1 --port 8000
```

For a Jetson connection, use `--host 0.0.0.0` instead. If the port is occupied, stop the existing server or choose another port; change the camera command and dashboard URL to match.

### 3. Start the camera — terminal 2

Close other apps/runners using this webcam. Run from the same repository folder:

```powershell
.\.venv\Scripts\python.exe edge/run_fall_model.py --camera 0 --preview --hand-gestures --backend http://127.0.0.1:8000 --room room-01
```

The preview draws body/hand boxes and shows **STANDING / SITTING / LYING / UNKNOWN**. Terminal output includes raw posture and hand labels, confidence, input processing and inference time. `fallen` in terminal/JSON corresponds to Lying in the UI. Press Q in the preview or Ctrl+C in the terminal to stop.

Open [the nurse dashboard](http://127.0.0.1:8000/nurses), enter a display name, select **room-01**, **Possible falls** and **Help requests**, keep **On duty** checked, and start the session.

### 4. Test the expected behavior

Use one consenting person at a time. Position the camera to include the full body and floor; safely lie on a mat rather than actually falling.

| Test | Expected result |
| --- | --- |
| Stand, then sit | Source health shows Standing, then Sitting when confidence is at least 0.65 |
| Safely lie down and remain visible | Lying; after 3 seconds of consecutive confident lying observations, one **Possible fall** alert |
| Stand again for at least 1 second | Recovery recorded; the alert remains open for human acknowledgement/resolution |
| Show a fist briefly, for less than 2 seconds | No help alert |
| Hold one fist toward the camera for at least 2 seconds | One **Help requested** alert; terminal should show `hand: fist` with confidence >= 0.65 |
| Keep holding the fist | No repeated help alerts |
| Show an open palm for at least 1 second, wait until 10 seconds have elapsed since the last request, then hold a fist again | A new help alert |
| Leave the frame or obscure the hand | Unknown; missing detections do not imply safety or release the help latch |
| Stop the camera runner | Source becomes stale/offline; subscribed nurses can receive a source-offline alert |

For either temporal signal, accepted observations must remain no more than 1 second apart. Low confidence, missing detections and gaps reset a pending hold. Multiple detected people/hands are uncertain. The camera model does not identify who requested help.

OpenCV CLAHE contrast retry is enabled for uncertain posture output; confident original detections are preserved. The model still determines the class—wide boxes alone do not imply lying. To compare the original model without contrast retry, stop the runner and repeat the same test with:

```powershell
.\.venv\Scripts\python.exe edge/run_fall_model.py --camera 0 --preview --hand-gestures --posture-enhancement none --backend http://127.0.0.1:8000 --room room-01
```

Contrast retry adds inference work on uncertain frames; improved accuracy must be measured on your camera. Keep `--posture-retry-threshold` aligned with backend `model_confidence_min` if changing it.

### 5. Automated tests and edge recording checks

```powershell
# Run separately: the backend and edge suites have overlapping test filenames.
.\.venv\Scripts\python.exe -m pytest backend/tests -q
.\.venv\Scripts\python.exe -m pytest edge/tests -q

# Check one actual Jetson observation recording (replace the example filename):
.\.venv\Scripts\python.exe edge/tools/check_rules.py recordings/lie_down-example.jsonl --config backend/hearth/monitor/config.json

# Check every JSONL recording in PowerShell:
Get-ChildItem recordings -Filter *.jsonl | ForEach-Object {
    .\.venv\Scripts\python.exe edge/tools/check_rules.py $_.FullName --config backend/hearth/monitor/config.json
}
```

The checker prints the real detector verdict, alert/recovery times and conditions that fail. Repeat with standing, sitting and lie-down clips. Tests with synthetic observations verify software behavior; actual recordings are needed to validate the camera calibration. Windows skips the Jetson's POSIX stdin-control test.

The live config supports both producers: supplied model predictions are authoritative; pose-only edge observations use calibrated pose rules (`posture_mode: "pose_fallback"`). Floor head/hips are 0.40/0.45 for the reported mat-camera setup; upright hips use an independent bound of 0.65. Use `posture_mode: "model"` to require model predictions. Separate camera sources are not automatically fused. See [the edge integration notes](docs/gaston-edge-integration.md), [model runner details](edge/MODEL_RUNNER.md), and [fist-help rules](docs/hand-help-signal.md).

### Troubleshooting

- **422 / `extra_forbidden` for model fields:** an older backend/schema is running. Reinstall the editable shared/backend packages using step 1, stop the old server and restart with this checkout's `.venv` Python. Do not remove the model fields.
- **Posture unknown:** inspect the preview and terminal scores. Below 0.65, absent detections, multiple people or stale observations remain unknown. Check [ingestion status](http://127.0.0.1:8000/ingest/status): increasing `stale` counts indicate delayed frames; `future` counts suggest clock mismatch. Synchronize laptop/Jetson clocks.
- **No fist alert:** check `hand: fist`, score >= 0.65, continuous detection for 2 seconds, and nurse room/help coverage. After a request, observe a non-fist gesture for release and respect the cooldown. `hand_help_label` is set in backend config, so restart after changing it.
- **Camera fails to open:** close other camera apps or try `--camera 1`.
- **Slow inference breaks the hold:** inspect `inference_ms`; compare with `--posture-enhancement none`. Raising `--max-fps` does not make the model run faster.
- **Temporary-directory permission error during installation/tests:** create `.venv/tmp`, then set `$env:TEMP` and `$env:TMP` to its absolute path in that terminal and retry.

AngelCare 8B is a separate, unverified GPU experiment; it is not part of these CPU-laptop commands or nurse alerts. See [its setup notes](docs/angelcare-live.md) if an appropriate GPU machine becomes available.

## Nurse desk and help requests

Start or restart the server with the command below, then open:

- `http://127.0.0.1:8000/nurses` — nurse coverage preferences, live alert queue, named acceptance, resolution, source health and timeline.
- `http://127.0.0.1:8000/room` — room help button and visible request/acknowledgement confirmation.

These pages are served by FastAPI from `backend/hearth/api/static/`, with no Node install, CDN or frontend build. They use the same APIs available to the React frontend.

**Minimum demo:**

1. Open `/nurses`, enter a display name, choose `room-01` and **Help requests**, keep **On duty** checked, and start the session. Click **Enable alert sound** if desired.
2. Open `/room` in a separate tab and press **Request help**. A labelled **BUTTON REQUEST** alert appears at the care desk.
3. Accept the request in the nurse tab. The room panel shows the nurse's acknowledgement. Resolve it to allow a fresh request.
4. Open a second nurse tab with a different name and the same coverage. Both see matching alerts and who accepted them; only the first accepting nurse owns the alert and can resolve it.
5. Change the second nurse's coverage to **Camera / source offline** only and save. Help requests will no longer be routed there. Off-duty sessions receive no new matching requests; their already accepted work remains visible.
6. To test a gesture without hardware, run the synthetic trace below. A brief wave or an interrupted gesture should not produce a help request.

```powershell
.\.venv\Scripts\python.exe backend/scripts/fake_observations.py --scenario gesture
.\.venv\Scripts\python.exe backend/scripts/fake_observations.py --scenario wave
.\.venv\Scripts\python.exe backend/scripts/fake_observations.py --scenario gesture_occlusion
# Optional second logical room for subscription routing:
.\.venv\Scripts\python.exe backend/scripts/fake_observations.py --scenario gesture --room room-02
```

The second logical room has no camera data until a source sends it. The default config permits `room-01` and `room-02`; change `additional_room_ids` to configure coverage. Room WebSocket streams and snapshots are isolated by room.

**Gesture contract:** Person 1 sends the existing normalized pose keypoints. A wrist must be above both the nose and same-side shoulder by `gesture_raise_margin`, with all three confidence values at least `gesture_keypoint_confidence_min`. The same arm must satisfy that condition continuously for `gesture_hold_seconds` (default 2 seconds). Missing frames, low-confidence keypoints, or occlusion clear pending evidence. One held gesture creates one `help_requested` event/alert. Re-arming requires observing the triggering arm lowered for `gesture_release_seconds` and respecting `gesture_cooldown_seconds`. Occlusion never counts as release. These rules do not identify the person or prove their intent; validate them against reaching, waving and stretching on the actual camera.

The room panel provides visual confirmation for camera gestures and button requests. It does not control a Jetson speaker. Nurse alert sound requires enabling audio in that browser tab. These are live **open-page notifications**, not SMS, mobile push, or delivery while a browser is closed. Connection interruptions are displayed; reconnect restores current matching alerts without sounding every historical request again.

Profiles, subscriptions, ownership, and events are in memory and clear when the backend restarts. Nurse profiles are local demo sessions, **not authenticated staff accounts**; nurse IDs are identifiers, not security credentials. Exclusive acceptance prevents accidental competing claims through the workflow; authentication and durable delivery would be separate deployment work.

## Run the monitoring core

From the repository root, with Python 3.11+ (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ./shared/python -e './backend[dev]'
.\.venv\Scripts\python.exe -m uvicorn hearth.api.app:app --host 127.0.0.1 --port 8000
```

For Jetson access over the local network, use `--host 0.0.0.0` and the laptop's LAN address. Run **one worker**: state is in memory. Interactive API documentation is at `http://127.0.0.1:8000/docs`. Once dependencies are installed, the backend and scripts work without internet.

In a second terminal, run the stub (no Jetson or frontend needed):

```powershell
.\.venv\Scripts\python.exe backend/scripts/fake_observations.py --scenario fall
.\.venv\Scripts\python.exe backend/scripts/fake_observations.py --scenario normal
.\.venv\Scripts\python.exe backend/scripts/fake_observations.py --scenario occlusion
.\.venv\Scripts\python.exe -m pytest backend/tests -q
```

`fall` produces one possible-fall alert and then a recovery observation; the alert remains open for human review. `normal` and `occlusion` should produce no **possible-fall** alerts. Every stopped source can subsequently produce a **source-offline** alert, routed to nurses subscribed to that situation. Each invocation uses a new `run_id`. Use `--fast` for accelerated transport with the same observation-time intervals. The stub is explicitly labelled `simulation` (or `--source-type replay`); it is not Person 3's scenario/evaluation system.

## Person 2 integration contract

| Route | Behavior |
| --- | --- |
| `POST /ingest/observations` | `ObservationBatch`, 1–500 observations; returns accepted, duplicates, out_of_order, stale and future counts |
| `GET /ingest/status` | Cumulative ingestion counts since startup and live edge timestamp limits |
| `POST /ingest/heartbeat` | `Heartbeat`; returns current `SourceStatus` |
| `WS /ws/room/room-01` | Typed observation, event, source-status and alert messages; reconnect gets current state and retained events |
| `GET /rooms/room-01/snapshot` | Latest observations per source/run, retained events, alerts and source statuses |
| `GET /alerts` | Alerts; optional `source` and `run_id` filters |
| `POST /alerts/{alert_id}/ack` | Open → acknowledged; repeated requests are idempotent |
| `POST /alerts/{alert_id}/resolve` | Acknowledged → resolved; resolving an open alert returns 409 |
| `GET /events` | Retained events; optional `source` and `run_id` filters |
| `GET /events/export` | Same filtered events as downloadable JSONL |
| `GET /sources` | Connection health and observation staleness |
| `GET /health` | Backend process health, not a declaration that the room is safe |
| `GET /rooms` | Configured room IDs and gesture hold duration |
| `POST /help-requests` | `{request_id, room_id}`; manual help request with retry/double-click deduplication |
| `POST /nurse-sessions` | Create `{display_name, room_ids, alert_kinds, on_duty}` preferences |
| `PUT /nurse-sessions/{nurse_id}` | Update coverage/duty; connected nurse streams receive a replacement snapshot |
| `GET /nurse-sessions/{nurse_id}/snapshot` | Current matching alerts, events and source statuses |
| `WS /ws/nurse/{nurse_id}` | Initial `nurse_snapshot`, then filtered alert/event/source-status updates; no observation stream |
| `POST /alerts/{alert_id}/accept` | `{nurse_id}`; atomically assign to an eligible on-duty nurse, or 409 if already owned |

For nurse-owned alerts, `POST /alerts/{alert_id}/resolve` requires the owning `{nurse_id}`. The original `/ack` flow remains available for unassigned alerts. Each alert has a `kind`: `possible_fall`, `help_requested`, or `source_offline`. Human acceptance is not proof that assistance has physically arrived. Source reconnection records recovery on an offline alert but does not resolve it automatically.

- **Person 1:** Existing observation fields remain valid. Send heartbeat context (`source`, `source_type`, `room_id`, and any scenario/run IDs) matching observations. Features must follow the normalized conventions below; missing features remain `null`. Frames with no person still send `person_detected: false`.
- **Person 3:** Send observations through the same ingest API, or use `app.state.monitor.ingest(observations)` on the backend event loop. Mount scenario-control routes onto the same FastAPI app. Keep ground truth in `hearth.sim`/`hearth.eval`; retrieve monitor events with `run_id` for scoring. Assign a new run ID for every reset/replay, even with the same scenario and seed. A run's timestamps must advance strictly; duplicate/equal/older samples do not advance detection.
- **Person 4:** Import generated server types from `shared/types/hearth.ts`. Use the `type` discriminator on `ServerMessage`. Upsert events by `event_id`, alerts by `alert_id`, and source/latest-observation records by `(room_id, source_type, source, scenario_id, run_id)`. Reconnect repeats retained history, so deduplicate it. The optional HTTP snapshot provides the same current state. Display `observation_stale` separately from `online`, and always preserve source labels. Recovery is shown with `recovered_at`; it does not automatically acknowledge or resolve an alert.

WebSocket clients that fall behind are closed with code 1013 and should reconnect to refresh state. CORS permits localhost Vite on port 5173 by default; other origins can be configured.

### Detection, timing and configuration

All tunable rules are in `backend/hearth/monitor/config.json`. Set `HEARTH_CONFIG` to a different JSON path to override it, then restart the server.

- Floor evidence requires a near-horizontal torso and low head/hip positions. Bounding-box aspect is checked when available. Upright recovery requires an upright torso and a head above the hip away from floor level.
- Descending is tracked from downward velocity. Sustained floor-level evidence can still alert if the descent was not observed; the event reason is `sustained_floor_level`, not a confirmed fall. Intentional lying down can therefore also alert.
- A default 3-second floor hold uses **observation timestamps**, so transport batching and replay speed do not change detection time. The default maximum observation gap is 1 second. A gap, missing pose evidence, or person occlusion resets a pending hold; missing information never emits recovery.
- Source health uses **backend monotonic receipt time**, with UTC receipt timestamps in health events. After 3 seconds without observations, posture becomes unknown/stale. Heartbeats can keep a source connected but cannot make old observations fresh. After 3 seconds without either observations or heartbeats, emit `source_offline`; fresh traffic emits `source_online` once.
- During simulator pause, keep heartbeats flowing if connected status is desired. Paused observations become stale and pending floor evidence is reset after the timeout. Resume with advancing scenario time; do not count wall-clock pause time as observed posture.
- Alert opening/recovery times use the observation clock. Human acknowledgement/resolution and source-health events use backend receipt time (`metadata.clock`). Evaluators should score `possible_fall` events by run, not health or acknowledgement timestamps.
- Zones use **normalized observation-frame coordinates**, not room meters. A confident hip location inside a configured bed/chair polygon downgrades an alert to `info` and records the reason; it never removes the event. Without hip location there is no downgrade. The simulator must project its room positions into the same frame before sharing zones.

Example `zones` configuration (replace with calibrated geometry):

```json
[{"name": "bed", "kind": "bed", "polygon": [[0.1, 0.7], [0.6, 0.7], [0.6, 1.0], [0.1, 1.0]]}]
```

The event log retains the latest 10,000 events by default; JSONL export includes only those retained events. Export before the capacity is exceeded or the backend restarts. Alerts and source/run state stay in memory until restart; this is a single-session hackathon backend, not durable storage. Old simulation runs remain visible by ID and can be filtered by the dashboard/evaluator.

## Scope

The initial demo uses one camera and one person at a time. The backend now supports configured logical rooms for nurse subscription routing, without a building map or cross-camera identity tracking.

In scope:

- One camera, one room, one person at a time.
- A deterministic one-room simulator with seeded, scripted scenarios: walking, sitting, bending, intentional lying down, possible fall, occlusion, camera outage, prolonged inactivity.
- A possible-fall state machine built from interpretable rules (torso angle, head/hip height, vertical motion, bounding-box aspect, time near floor). No custom model training.
- Face **detection** (is a face visible?) used to mark how well the person is observed.
- Evaluation: detection delay, true/false positives, missed events, test count.
- Held-arm/manual help requests, live nurse coverage subscriptions, named acceptance and resolution.

Out of scope — do not add these:

- Hallways, building maps, pathfinding, resident navigation.
- Multiple cameras or cross-camera tracking.
- Face **recognition**, identity assignment, biometric databases.
- Gait analysis, fall-risk scores, clinical predictions, custom fall-model training.
- Cloud services, runtime internet dependencies, physical props, robots.

## Architecture

```text
ONE REAL ROOM                         ONE SIMULATED ROOM
USB camera                            Scripted resident activity
    |                                 + ground truth (kept separate)
Jetson Nano 2GB                              |
  frame capture                       observation generator
  pose detection (+ face detection)          |
  per-frame posture features                 |
    |                                        |
    +---- Observations (edge) ---------------+---- Observations (simulation)
                                             |
                                    FastAPI on laptop
                                      ingestion
                                      monitor: possible-fall state machine
                                      source health / staleness
                                      scenario controller
                                      evaluation (reads ground truth)
                                             |
                                        WebSocket
                                             |
                                    React dashboard
                                      room view, skeleton, alerts,
                                      timeline, scenario controls,
                                      evaluation / after-action report
```

## Critical invariants

These are the rules the project's credibility depends on. Do not break them.

1. **Ground truth never reaches the monitor.** `GroundTruth` records exist only in the simulator and evaluator. Code in `backend/hearth/monitor/` must never import `hearth.sim`, `hearth.eval`, or `GroundTruth`. Add a test that enforces this.
2. **One pipeline for every source.** Jetson, simulator and replay observations all go through the same monitor code. The Jetson sends observations, not alerts. The possible-fall decision is made once, in the backend.
3. **Every record says where it came from.** `source_type` is `edge`, `simulation`, `replay`, or `manual` (a help-button request). The dashboard labels these visibly; manual requests never imply camera observations.
4. **Silence is not safety.** If a source stops sending, the room is shown as offline / unknown, never as fine.
5. **No raw video leaves the Jetson.** Only structured observations and keypoints are sent. Nothing records or uploads video by default.

## Repository layout

```text
backend/            FastAPI app on the laptop (Python 3.11+)
  hearth/api/         ingestion endpoints, WebSocket feed
  hearth/monitor/     possible-fall state machine, visibility/face events
  hearth/sim/         one-room simulator, scenarios, ground truth
  hearth/eval/        compares monitor events against ground truth
  tests/
edge/               Jetson Nano code
frontend/           React + TypeScript + Vite dashboard (SVG/Canvas, no 3D engine)
shared/             the contract every component speaks
  python/             Pydantic models: the source of truth
  schema/             generated JSON Schema (do not edit by hand)
  fixtures/           example messages, placeholder values only
  scripts/            schema export
  types/              generated TypeScript server types
scenarios/          scripted simulator scenarios
recordings/         local keypoint traces (git-ignored; never raw video)
```

## Shared contract

Everything that crosses a component boundary is defined in `shared/python/hearth_schema/models.py` (schema version `"1"`).

| Record | Meaning | Produced by |
| --- | --- | --- |
| `Observation` | What a sensor could see: keypoints, posture features, face presence | Jetson, simulator, replay |
| `Event` | What the monitor concluded | Backend monitor |
| `Alert` | Human review lifecycle for a possible-fall incident | Backend |
| `GroundTruth` | What actually happened in a simulated scenario segment | Simulator only |
| `Heartbeat` | Source liveness plus FPS, inference latency, memory | Every source, ~1 per second |
| `SourceStatus` | Backend's view of whether a source is online | Backend |
| `ServerMessage` | WebSocket envelope: `observation`, `event`, `source_status`, `alert`, or `nurse_snapshot` | Backend |
| `RoomSnapshot` | Current observations, events, alerts and source statuses | Backend |
| `NursePreferences` / `NurseProfile` | Coverage choices, duty status and demo session ID | Nurse desk / backend |
| `HelpRequest` | Idempotent manual room assistance request | Room panel |

Conventions:

- Timestamps are ISO 8601 with a timezone (producers should use UTC); naive input timestamps are rejected.
- `run_id` is optional for backwards compatibility; simulator/replay integrations should always supply a new ID for each run. Include it on heartbeats too.
- Keypoints use the 18 COCO names from poseNet `resnet18-body` and are normalized to the frame: `(0, 0)` top-left, `(1, 1)` bottom-right.
- Posture features: `torso_angle_deg` (0 = upright, 90 = horizontal), `head_y` and `hip_y` (normalized, 0 = top of frame), `bbox_aspect` (width / height), `vertical_velocity` (change in `hip_y` per second, positive = moving down). All optional; omit what can't be measured.
- `face_present: null` means face detection did not run on that frame, which is different from `false`.
- Event types: `possible_fall`, `help_requested`, `recovered`, `person_visible`, `person_not_visible`, `face_present`, `face_absent`, `source_offline`, `source_online`, `alert_acknowledged`, `alert_resolved`.
- Ground-truth labels: `walking`, `standing`, `sitting`, `bending`, `lying_intentional`, `fall`, `occluded`, `camera_outage`, `absent`. Times are seconds from scenario start.
- Models reject unknown fields (`extra="forbid"`).

To change the contract: edit `models.py`, regenerate JSON Schema and TypeScript, and commit all three. Generated TypeScript describes server output: defaulted fields are always serialized. Input producers may omit fields with model defaults.

```sh
pip install -e shared/python
python shared/scripts/export_schema.py
python shared/scripts/export_types.py
```

## Design and remaining integration

- **Transport:** the Jetson POSTs batches to `POST /ingest/observations` (body: `ObservationBatch`) at roughly 5–10 Hz, and a `Heartbeat` to `POST /ingest/heartbeat` every second. The backend marks a source offline after 3 seconds of silence. The dashboard subscribes to `WS /ws/room/{room_id}` and receives `ServerMessage`s.
- **Room IDs:** `room-01` plus configured `additional_room_ids` (`room-02` by default for routing demos). Backend on port 8000, optional Vite dev server on 5173.
- **Possible-fall state machine:** roughly `upright → descending → floor_level → possible_fall` after a configurable hold time, with `recovered` on return to upright. All thresholds in one config file. Tune on real recorded keypoint traces.
- **Alerts:** open → acknowledged → resolved, with an acknowledge action in the dashboard.
- **Simulator:** deterministic clock and seed; play, pause, reset, replay, time scaling. The same seed must reproduce the same run.
- **Zones (stretch):** bed/chair regions may downgrade an alert but must not hide uncertainty.
- **Frontend types:** generated from `shared/schema/hearth.schema.json` into `shared/types/hearth.ts`; do not edit them by hand.

## Hardware and environment constraints

- **Edge:** NVIDIA Jetson Nano **2GB**, one USB webcam (Arducam or Logitech). Memory is the main risk. Use `jetson-inference` (poseNet; facenet for faces) if it runs; verify every model and FPS figure on the device before relying on it.
- **Edge Python:** assume the stock JetPack Python, likely 3.6. Edge code must use the standard library only (no Pydantic, no modern syntax such as f-string `=` or `|` types) and build plain dicts that match the contract.
- **Laptop:** CPU-only Framework 13, Python 3.11+, Node 22.
- **Network:** direct Ethernet or LAN. The demo must work with no internet.

## Fallbacks

- If pose inference on the Jetson is blocked, keep building with recorded or synthetic observations, labelled `replay` or `simulation`, and say plainly that live capability is missing.
- If face detection doesn't work, do not claim the Full Sail face-detection requirement.
- After hour 26 of the hackathon: no new features, only fixes, benchmarks, false-positive testing and rehearsal.

## Demo (90 seconds)

1. Simulator running a normal-activity scenario.
2. Live camera on a consenting teammate, showing face presence and pose skeleton.
3. Teammate safely lowers onto a mat and a **possible-fall** alert appears.
4. A normal sit/bend that does not alert — only if it has actually been validated.
5. Replay of a simulated incident with its measured alert delay against ground truth.

Never ask a judge to fall. Never imply lying down proves a fall.

## Claims and privacy

- Alerts are **possible** safety events for human review, not diagnoses or certified fall detection.
- Do not claim clinical validation, measured accuracy beyond what was actually tested, or HIPAA compliance. Report live test results as small-sample.
- Face detection is not identification. No enrollment or biometric identification.
- Skeleton data can still be sensitive; do not describe it as anonymous.
- Test only with consenting teammates and synthetic resident profiles. Label all simulated data as simulated.

## Team ownership

| Role | Owns | Code |
| --- | --- | --- |
| 1 — Edge/CV | Camera, pose, face detection, posture features, edge client, benchmarks, recorded traces | `edge/` |
| 2 — Monitoring core | Contract, ingestion API, possible-fall state machine, source health, alerts, WebSocket | `shared/`, `backend/hearth/api/`, `backend/hearth/monitor/` |
| 3 — Simulator + evaluation | Scenarios, ground truth, observation generator, scenario controller, metrics, after-action data | `backend/hearth/sim/`, `backend/hearth/eval/`, `scenarios/` |
| 4 — Dashboard + demo | Room view, skeleton, alerts, timeline, scenario controls, evaluation view, demo script | `frontend/` |

Changes to `shared/` affect everyone; coordinate them through the monitoring-core owner.

### Posture determination

Model-only mode (`posture_mode: "model"`) is available: the Hugging Face checkpoint determines Standing, Sitting, and Lying (checkpoint/wire label `fallen`). Pose-only observations remain unknown for posture and cannot generate or recover a possible-fall incident. Pose keypoints remain usable for the separate gesture/visibility pipeline. Use `edge/run_fall_model.py` on a device with camera access; Gaston's poseNet runner alone does not produce model classifications. The default `pose_fallback` mode supports calibrated Jetson pose observations. A sustained lying classification still requires the configured hold before a possible-fall alert.

### Live edge/model integration

The live config now uses `pose_fallback`: supplied Hugging Face posture predictions remain authoritative, while rotated-poseNet observations without predictions use calibrated pose thresholds. Floor head/hips are 0.40/0.45 for the reported mid-frame mat camera; upright uses its independent `upright_hip_y_max` of 0.65 so standing at hip_y 0.60 can recover incidents. Separate sources are not implicitly fused. See `docs/gaston-edge-integration.md` and `edge/tools/check_rules.py` for clip validation.

Pose-only upright detection can include sitting: it is not a three-class posture classifier. Use the laptop model runner for the Standing/Sitting/Lying demo. `recordings/config_framing.json` explicitly uses `pose_fallback` and the independent upright bound, so the recorded mat clips work with that profile too.

### Test the integrated simulator and evaluator

Start the backend using the laptop setup above, then open <http://127.0.0.1:8000/docs>:

1. Execute `GET /sim/scenarios`; it should list the scenario library.
2. Execute `POST /sim/load` with `{"scenario_id":"possible-fall-01","seed":42}`.
3. Execute `POST /sim/play`. Watch `/room` or `/nurses` for a visibly **SIMULATED** source and a possible-fall alert. Select room-01 and fall alerts in the nurse page.
4. Once `GET /sim/status` says `ended`, copy its `run_id`. `GET /eval/runs/{run_id}` returns the actual-versus-observed report. `GET /alerts?run_id=...` isolates that run's alerts.
5. `POST /sim/reset`, then play again; a new run ID prevents old observations or incidents from contaminating the replay. `/sim/pause` and `/sim/speed` (body `{"speed":4}`) control playback.

For repeatable headless evaluation, run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m hearth.eval batch --seeds 1 --scenarios possible-fall-01 camera-outage-01
Get-Content recordings/eval/summary.md

# Run the backend and edge suites separately (they share test filenames):
.\.venv\Scripts\python.exe -m pytest backend/tests -q
.\.venv\Scripts\python.exe -m pytest edge/tests -q
```

The default factory uses the real monitor with a virtual receipt clock; no test stub or simulator truth is used to decide alerts. `HEARTH_CONFIG` selects the same profile for the API and evaluator. Synthetic results measure behavior on those scenarios, not real-world accuracy. `GET /eval/summary` becomes available after the batch writes its default output. Files and live alert state are session-scoped as described above.

The real monitor can alert on intentional lying too: sustained floor posture is a **possible** fall, not proof of an accident. The evaluator reports those false positives rather than hiding them. Calibrated bed/chair polygons downgrade alerts but do not remove them; scenario room-meter geometry is not automatically converted into camera-frame suppression polygons.

The React room-view development page on port 5174 currently plays **recorded** runs; the live pages are `/room` and `/nurses` on the backend. The simulator and evaluation APIs are connected, while a combined React scenario/evaluation control page remains dashboard work. Both frontend development ports 5173 and 5174 are allowed by the default backend CORS config. See [component interfaces](docs/interfaces.md) for the clocks, ingestion seam and routes.
