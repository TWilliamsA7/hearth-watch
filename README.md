# HearthWatch

**A room-level care companion that helps residents ask for help and caregivers respond.**

Built for **Knight Hacks IX · October 9–11, 2026**.

## Inspiration

Nursing-home staff balance scheduled care with needs that arise throughout the day. A resident may need assistance between rounds, miss an appointment, or be unable to reach a call button. We built HearthWatch to explore how local computer vision and a shared caregiver dashboard could make those moments easier to notice and respond to.

Our focus is the response: a clear request, a caregiver taking responsibility, and confirmation that the request has been handled.

## What it does

- **Flags possible falls.** Posture observations must meet confidence and duration rules before an alert opens.
- **Offers two ways to request help.** Residents can press SOS on the patient page or hold a fist toward the camera.
- **Connects requests to caregivers.** Nurses choose their rooms, alert categories, and duty status. The first nurse to accept owns the request, and other nurses can see who is responding.
- **Tracks appointments.** Caregivers can schedule and cancel appointments. An unconfirmed arrival triggers a reminder after the default 15-minute grace period.
- **Supports optional patient recognition.** An opted-in person can link appointments for recognized check-in and separately choose to include their name on camera-generated incidents.
- **Makes testing repeatable.** A seeded 3D room simulator produces observations for the same monitoring backend used by live sources. Evaluation compares the resulting alerts with simulator ground truth.

The main caregiver dashboard runs at **http://127.0.0.1:5173/**. It includes caregiver and patient views, simulation controls, a live camera view, alerts, source health, and appointments in one interface.

## How we built it

### Dashboard

React, TypeScript, and Vite provide the caregiver and patient interface. WebSocket updates deliver observations and alerts, while HTTP endpoints handle scheduling, check-in, and caregiver actions. The shared room renderer uses Three.js for simulation visualization.

The care workflow is **request → caregiver acceptance → resolution**. Room and alert subscriptions determine which requests a nurse receives. Acceptance is exclusive so two caregivers do not accidentally claim the same request.

### Edge perception

The Jetson Nano pipeline processes a Logitech camera using `jetson-inference` poseNet. Rotated pose passes improve joint detection for people lying across the image. Face detection supplies face-presence observations. Structured observations and heartbeats go to the backend.

A separate laptop runner uses Hugging Face posture and hand-gesture models. It can also match opted-in face profiles and share an annotated live preview with the dashboard. These are separate camera producers; the system does not automatically combine their identities or detections.

### Monitoring

Python, FastAPI, and Pydantic define and validate the observation contract. The monitor applies posture and gesture rules, tracks missing or stale sources, creates alerts, and manages caregiver ownership. Appointments use SQLite; optional enrollment templates and names are encrypted with a locally supplied key.

A lying prediction is evidence for a **possible fall**, rather than proof of one. Recovery is recorded separately from the caregiver's acknowledgement and resolution. Missing observations remain uncertainty.

### Simulation and evaluation

The simulator generates seeded resident activity, perception observations, and separate ground-truth timelines. Live and simulated observations pass through the same monitoring rules. The evaluator measures detection delay, detected incidents, false alerts, and missed events.

The simulator can also place a Jetson-observed person in its 3D room using camera calibration. Monitoring decisions still come from the backend. Simulated participants responding to an incident are part of the simulation, not evidence that a real caregiver has arrived.

```text
Jetson pose observations ─┐
Laptop camera models ─────┼─→ FastAPI monitor ─→ Caregiver dashboard
Simulation observations ─┘   posture + gesture     accept → resolve
                             health + appointments

Simulator ground truth ─────→ Evaluator ←─ Monitor events
```

## Hugging Face models

| Model | How we use it |
| --- | --- |
| [melihuzunoglu/human-fall-detection](https://huggingface.co/melihuzunoglu/human-fall-detection) | Detects standing, sitting, and lying postures. The checkpoint's lying label is `fallen`; monitoring adds confidence and hold-time rules. |
| [SuperSura/hand-gesture-yolo26n](https://huggingface.co/SuperSura/hand-gesture-yolo26n) | Classifies hand gestures. A sustained, confident `fist` is the configured help signal. |
| [OpenCV YuNet](https://huggingface.co/opencv/face_detection_yunet) | Locates a face for enrollment and recognition. |
| [OpenCV SFace](https://huggingface.co/opencv/face_recognition_sface) | Generates embeddings that are compared with explicitly enrolled profiles. |

We use pretrained weights, pinned revisions, and checksum verification. Internet access is needed for installation and model downloads. Downloaded models run locally. We did not train these models.

The Jetson's face-presence detector and the laptop's optional YuNet/SFace recognition are distinct integrations. The complete recognition workflow described below runs on the laptop; Jetson deployment of that workflow still requires integration and benchmarking.

## Run the full project on a laptop

Use Python 3.11 or newer, a recent Node.js version compatible with the checked-out frontend, PowerShell, and a webcam. Run commands from the repository root. Use **Terminal 1 for the backend and dashboard** and **Terminal 2 for the camera**.

### 1. Install once

Skip virtual-environment creation if `.venv` already exists.

```powershell
cd C:\Users\Sam\Documents\GitHub\hearth-watch
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e "./shared/python" -e "./backend[dev,vision,face-demo]"
$env:PYTHONPATH="$PWD\backend;$PWD\shared\python"

Push-Location frontend
npm.cmd install
Pop-Location

.\.venv\Scripts\python.exe edge/run_fall_model.py --download-only --hand-gestures
.\.venv\Scripts\python.exe -m hearth.perception.face_demo --download --check
```

This installs the API, dashboard dependencies, and camera models before the presentation.

### 2. Prepare the recognition key

For a first enrollment, generate a key once:

```powershell
if (-not (Test-Path .env.face-key)) {
    .\.venv\Scripts\python.exe -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" |
        Set-Content -Encoding ascii .env.face-key
}
```

If you already have enrollments, use their original key instead. A replacement key cannot decrypt existing profiles. Keep `.env.face-key` between runs; it is ignored by Git.

### 3. Start Terminal 1

```powershell
$env:HEARTH_FACE_DEMO="1"
$env:HEARTH_FACE_KEY=(Get-Content -Raw .env.face-key).Trim()
.\scripts\dashboard.ps1
```

Open **http://127.0.0.1:5173/** and leave this terminal running. The launcher starts the API on **8002** and Vite on **5173**. The frontend forwards API and WebSocket requests to the backend.

These environment variables must be set before the backend starts. The launcher does not automatically load the key file. If it reuses an existing backend, that process retains its original settings. The internal `HEARTH_FACE_DEMO` name remains for compatibility.

For a session without recognition, run just `.\scripts\dashboard.ps1`. Buttons, simulation, and monitoring do not require enrollment.

### 4. Start a caregiver session

Open **http://127.0.0.1:5173/nurses**. Enter a display name, select `room-01`, select **Help requests**, **Possible falls**, and **Appointment reminders**, keep **On duty** checked, and start the session. Enable alert sound if desired.

Open **http://127.0.0.1:5173/room** in another tab for patient controls. Schedule an appointment from the caregiver view if you want to demonstrate arrival or overdue reminders. Simulation does not populate the appointment schedule.

### 5. Enroll before starting the live runner

Open **http://127.0.0.1:5173/face-check-in**.

1. Enter your preferred name and give recognition consent.
2. Check **“Also show my name on possible-fall and camera help-signal alerts”** if you want named incidents. This is separate from appointment recognition consent.
3. Start the browser camera, capture three separate face views, and complete enrollment.
4. Select or link an appointment if testing recognized arrival. Check-in opens 15 minutes before its scheduled time.
5. Use **Start recognition** to verify the match.
6. Click **Stop camera** before starting the Python runner.

Only one application should own the webcam at a time. Enrollment saves the profile; the live runner must also enable recognition to match it during monitoring.

### 6. Start Terminal 2

```powershell
cd C:\Users\Sam\Documents\GitHub\hearth-watch
$env:PYTHONPATH="$PWD\backend;$PWD\shared\python"

.\.venv\Scripts\python.exe edge/run_fall_model.py `
    --camera 0 `
    --device cpu `
    --laptop `
    --backend http://127.0.0.1:5173 `
    --room room-01 `
    --patient-recognition `
    --hand-gestures `
    --dashboard-preview
```

Choose **Live camera** in the caregiver dashboard.

| Option | Purpose |
| --- | --- |
| `--camera 0` | Selects the webcam; change the index if necessary. |
| `--device cpu --laptop` | Uses the CPU and laptop inference preset. |
| `--backend ...:5173` | Sends requests through the dashboard's backend proxy. |
| `--room room-01` | Matches the room selected in caregiver coverage. |
| `--patient-recognition` | Matches enrolled profiles on the monitoring frames. Omit for unnamed monitoring. |
| `--hand-gestures` | Enables the fist classifier. |
| `--dashboard-preview` | Sends the annotated preview to the local dashboard. |

Keep your face and body visible. For a named incident, first wait for `patient: Your Name (recognized)` in the terminal. Matching requires three consistent frames and reliable body evidence. Initial model warm-up can take several seconds.

### 7. Stop after testing

Press **Ctrl+C in Terminal 2** to release the webcam, then **Ctrl+C in Terminal 1**. A backend started separately must be stopped separately. Restart nurse sessions after a backend restart or page reload. Appointments and encrypted enrollments persist on disk; enrollments expire after 24 hours.

## Test and evaluate

Run synthetic traces separately while the integrated dashboard is running:

```powershell
.\.venv\Scripts\python.exe backend/scripts/fake_observations.py --url http://127.0.0.1:5173 --scenario fall
.\.venv\Scripts\python.exe backend/scripts/fake_observations.py --url http://127.0.0.1:5173 --scenario normal
.\.venv\Scripts\python.exe backend/scripts/fake_observations.py --url http://127.0.0.1:5173 --scenario gesture
.\.venv\Scripts\python.exe backend/scripts/fake_observations.py --url http://127.0.0.1:5173 --scenario wave
.\.venv\Scripts\python.exe backend/scripts/fake_observations.py --url http://127.0.0.1:5173 --scenario gesture_occlusion
```

`fall` should create a possible-fall alert and recovery evidence. `gesture` tests the pose-based raised-arm path; it does not run the fist model. `wave` and `gesture_occlusion` should not create a help request. A stopped source may later produce a source-offline alert.

For reproducible simulation and evaluation:

```powershell
$env:PYTHONPATH="$PWD\backend;$PWD\shared\python"
.\.venv\Scripts\python.exe -m hearth.sim list
.\.venv\Scripts\python.exe -m hearth.sim run --all
.\.venv\Scripts\python.exe -m hearth.sim live --people 6 --seed 3 --duration 300 --fall-at 120 --evaluate
```

Headless commands write results locally; they do not automatically stream into the open dashboard.

For automated checks in the integrated checkout:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests edge/tests -q
Push-Location frontend
npm.cmd test
npm.cmd run build
Pop-Location
```

These are validation commands, not a claim that every test has been run for this documentation draft.

## Jetson and simulation development

The Jetson setup uses poseNet with rotated passes and face-presence detection. Camera position affects floor thresholds and recovery, so recheck recorded traces after moving the camera. The edge documentation reports an across-mat staged lying test that alerted after approximately three seconds; this is a specific test, not a general accuracy result.

- [Edge setup](edge/README.md)
- [Hardware presentation workflow](edge/DEMO.md)
- [Benchmarks](edge/BENCHMARKS.md)
- [Camera-to-room calibration](docs/live-in-sim.md)
- [Edge integration](docs/gaston-edge-integration.md)
- [Recorded-trace evaluation](docs/real-data-evaluation.md)

For the standalone Sim Studio in the reviewed `main` snapshot, start the backend on port 8000:

```powershell
$env:PYTHONPATH="$PWD\backend;$PWD\shared\python"
.\.venv\Scripts\python.exe -m uvicorn hearth.api.app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend/room-view
npm.cmd install
$env:HEARTH_BACKEND="http://127.0.0.1:8000"
npm.cmd run dev
```

Open **http://127.0.0.1:5174/studio.html**. This is the simulation development page; the combined caregiver dashboard uses 5173. For Jetson access, follow the edge network instructions and bind the backend to an appropriate network interface. The laptop-only recognition and preview routes do not automatically become remote Jetson services.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `vite` is not recognized | Run `npm.cmd install` in the frontend package you are starting. |
| Appointment card is empty | Schedule an appointment for today's local date. The simulator does not create appointments. |
| No caregiver alert | Verify an active nurse session, room, alert categories, On duty, and connection status. |
| Camera cannot start | Stop the enrollment camera, other camera tabs, and other webcam apps before starting Python. |
| Identity stays unconfirmed | Check incident-naming consent, `--patient-recognition`, face visibility, body confidence, and terminal output. Gaps over one second can reset identity confirmation. |
| Fist does not alert | Check for one confident fist, a completed hold, and sufficient consecutive detections. Lower the hand between requests and allow the cooldown. |
| Camera is laggy | Use `--laptop`; inspect `inference_ms`, `frame_gap_s`, and `pipeline_ms`. Recognition and both models share the laptop's processing budget. |
| Recognition says Disabled | Set the environment variables in the launch terminal and restart the backend. |
| Recognition says unavailable | Verify dependencies, models, and the original encryption key. |
| API returns 404 | Confirm the checkout contains the integration and restart an old backend process. |

Check recognition status and model loading from the repository root:

```powershell
Invoke-RestMethod http://127.0.0.1:8002/face-demo/status
$env:PYTHONPATH="$PWD\backend;$PWD\shared\python"
.\.venv\Scripts\python.exe -m hearth.perception.face_demo --check
```

If the launcher keeps reusing an old backend, stop the launcher first, then stop the process listening on the configured HearthWatch API port and repeat the startup steps:

```powershell
Get-NetTCPConnection -LocalPort 8002 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { Stop-Process -Id $_ }
```

If the original enrollment key is lost, existing records cannot be decrypted. To start fresh while preserving the old database, stop the backend and rename the database before restarting with the current saved key:

```powershell
$backupName="face-demo.backup-$(Get-Date -Format 'yyyyMMdd-HHmmss').sqlite"
Rename-Item -LiteralPath .\data\face-demo.sqlite -NewName $backupName
```

Then enroll again. This starts a new enrollment database; it does not recover the old profile.

## Configuration and data handling

Monitoring thresholds live in `backend/hearth/monitor/config.json`. Set `HEARTH_CONFIG` to another JSON file and restart to use a different configuration. The usual possible-fall hold is three seconds. The extended hand-model rule requires a two-second hold, at least three confident observations, and allows gaps up to 2.5 seconds; the reviewed main snapshot does not yet contain those extra hand-continuity settings.

Heartbeats describe connection health. Fresh observations describe what was recently seen. Neither a connected camera nor missing evidence proves a resident is safe. Observed recovery does not automatically resolve an alert.

Recognition is optional. Enrollment frames are processed locally without being saved; templates and enrollment names are encrypted at rest and expire after 24 hours. **Withdraw & delete** stops future recognition. Names already attached to historical alerts are separate incident records and are not erased by withdrawing enrollment. Encryption does not make named records anonymous.

The preview is opt-in and keeps the latest annotated frame in memory. Nurse sessions and most monitoring state are in memory; appointments persist in SQLite. Nurse display names are workflow identifiers, not authenticated staff credentials. Notifications require open pages. The recognizer does not implement liveness or photo-spoof checks.

HearthWatch is a hackathon prototype. Camera geometry, lighting, occlusion, and model uncertainty affect results. It has not been clinically validated. Our testing uses consenting teammates, synthetic residents, and labelled replay data.

## Repository structure

```text
backend/hearth/
  api/          HTTP routes, WebSocket feeds and enrollment endpoints
  monitor/      Posture and gesture rules, health, alerts and appointments
  perception/   Laptop model inference and optional recognition
  sim/          Room engine, scenarios and ground truth
  eval/         Detection scoring and reports
edge/           Jetson producer, camera tools and recorded-trace checks
frontend/       Integrated React caregiver and patient dashboard
  room-view/    Shared 3D renderer and Sim Studio development tools
shared/         Pydantic models, generated schemas and TypeScript contracts
scripts/        Integrated dashboard launcher
scenarios/      Scripted simulation inputs
```

The integrated dashboard, launcher, appointments, and recognition files require the newer integration described at the top of this document. Ground truth belongs to the simulator and evaluator; monitoring must not import it. Shared contract changes require regenerating the JSON schema and TypeScript types.

## What we learned

Camera geometry mattered as much as model choice. Rotated pose passes helped with bodies lying across the image, while separate floor and upright thresholds made recovery possible without weakening the floor rule. Temporal checks helped turn individual predictions into reviewable incidents, but also exposed how laptop latency can interrupt otherwise consistent evidence.

Connecting detection to caregiver actions was equally important. Source labels, uncertainty, ownership, and acknowledgement gave the alerts useful context. The simulator made those behaviors easier to reproduce and compare while camera integration continued.

## What's next

- Measure latency, missed events, and false alerts across more staged camera positions and activities.
- Integrate and benchmark the complete recognition and gesture workflow on the Jetson.
- Add authenticated caregiver accounts and durable notification delivery.
- Improve deployment-specific retention controls and evaluate recognition reliability and spoof resistance.

## Built with

Python · FastAPI · Pydantic · SQLite · React · TypeScript · Vite · Three.js · OpenCV · PyTorch · Ultralytics · Hugging Face · NVIDIA Jetson Nano · jetson-inference
