# Optional patient recognition: laptop demo

This local hackathon demo uses consenting teammates playing fictional residents. Enrollment is in the dashboard's patient view, under **Face recognition check-in** (`http://127.0.0.1:8000/room#recognition`).

## Hugging Face models

- [opencv/face_detection_yunet](https://huggingface.co/opencv/face_detection_yunet): YuNet finds a face and five landmarks for alignment.
- [opencv/face_recognition_sface](https://huggingface.co/opencv/face_recognition_sface): SFace produces a 128-value face embedding. The monitor compares it with opted-in enrollment templates using cosine similarity.

The downloader fetches ONNX weights directly from these Hugging Face repositories, pins revisions, and verifies SHA-256 checksums. OpenCV runs inference locally; no camera frames are sent to Hugging Face. Enrollment supplies reference examples; it does not train a new model. Sponsor challenge eligibility depends on the event rules.

## PowerShell setup

From the repository root, with the existing project environment installed:

```powershell
.\.venv\Scripts\python.exe -m pip install -e "./backend[vision,face-demo]"
.\.venv\Scripts\python.exe -m hearth.perception.face_demo --download --check
# Once: keep this file private and stable (git-ignored; scripts/face_id_demo.sh makes the same file).
$key = (& .\.venv\Scripts\python.exe -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())").Trim()
Set-Content -Encoding ascii .env.face-demo "HEARTH_FACE_KEY=$key"

.\scripts\start.ps1
```

`start.ps1` reads `.env.face-demo` and turns recognition on (`HEARTH_FACE_DEMO=1`). Create the key once and keep the file. A different key cannot decrypt existing enrollments; preserve the original key privately if retaining enrollments across sessions. Do not commit the key. Withdraw all enrollments before changing keys.

The feature is disabled unless explicitly enabled. Its API accepts loopback requests only.

## Demo steps

1. Open `http://127.0.0.1:8000/room#recognition` and pick the room in the header.
2. To test arrival, schedule an appointment under **My appointments → Schedule an appointment** in the same view (use the patient's name and a time a few minutes ahead).
3. Select an appointment only if you want recognition check-in. Confirm face-enrollment consent and separately select **Also show my name on possible-fall and camera help-signal alerts** if testing named safety alerts, then click **Start camera**.
4. Click **Capture 3 views and enroll**, following the prompts. Keep one face in view with good lighting.
5. Click **Check in with recognition**. Three consecutive matching frames confirm the patient. A linked, open appointment in the same room is marked arrived.
6. Verify the appointment through `GET /appointments` in `/docs`. Its method should be `face_recognition`; the appointment list shows **Recognized arrival**.
7. To demonstrate a late arrival alert, wait until the appointment grace period has passed before recognizing. Arrival records recovery; nurse acknowledgment and resolution remain separate actions.
8. Use **Withdraw & delete** to remove the enrollment and invalidate recognition sessions. Stop the camera when finished.

Rescheduling rotates the appointment token, so an old enrollment cannot check in the rescheduled appointment. Select the rescheduled appointment and click **Link selected appointment** beside the enrolled patient. Enrollment can be reused for multiple explicitly linked appointments. Recognition accepts arrival starting 15 minutes before the scheduled time, so seeing someone today does not check them into tomorrow's appointment.

## Named fall alerts from the laptop camera

If you enrolled using the previous check-in-only version, withdraw that enrollment and enroll again. The separate named-alert choice explicitly covers displaying a name on incidents; old check-in consent is not treated as permission for named alerts.

1. Finish enrollment, then click **Stop camera** in the browser. The camera must be free for the fall-model process.
2. Keep the backend running. In a second PowerShell window from the repository root:

```powershell
$env:PYTHONPATH="$PWD\backend;$PWD\shared\python"
.\.venv\Scripts\python.exe edge/run_fall_model.py --download-only
.\.venv\Scripts\python.exe edge/run_fall_model.py --camera 0 --backend http://127.0.0.1:8000 --room room-01 --preview --patient-recognition
```

Or let the start script run it: stop Hearth, then `.\scripts\start.ps1 -Camera 0` (with `.env.face-demo` present it adds `--patient-recognition`).

Use `--camera 1` if the Logitech USB camera is device 1. The same script can test an integrated laptop webcam or a Logitech webcam attached to the laptop. Add `--hand-gestures` after downloading that model if testing help gestures too.

3. In the caregiver view (`/`), start a nurse session and select room-01 and possible-fall alerts.
4. Stand alone in view, facing the camera. Wait for your demo name in the preview/terminal. Three distinct matching frames are required. Any linked open demo appointments for this room are checked in.
5. Stage the existing lie-down demonstration on a mat; do not perform a real fall. Once the fall detector's hold passes, the nurse alert should read **Possible fall · Alex**, using your enrolled demo name. The detector still reports a possible fall, not a confirmed injury.
6. If the face turns away, its confirmed name can persist for up to **8 seconds** while a single body remains continuously detected. Identity clears on a gap, an abrupt box change, multiple people, an ambiguous/unknown face, expiration, or enrollment withdrawal. The alert then shows **Identity unconfirmed**.
7. Stop the process with Q in the preview or Ctrl+C, and withdraw enrollment when done.

This is a conservative single-person demo association, not a validated multi-person tracker. Changes in posture, lighting, or box size can deliberately cause the name to be dropped. A person swapping into a similar body box is an unresolved limitation; do not present this as reliable crowded-room identification. Fall alerts continue without a name when recognition is unavailable.

## Named alerts from the Jetson camera (several people)

The Jetson computes each person's face template itself and sends only that, never frames. `scripts/face_id_demo.sh` (Git Bash or Linux, on the laptop) does all of the steps below and re-runs safely; `--help` lists its options. See [the integration contract](recognition-integration.md#jetson-face-templates-implemented-after-merging-multi-person-tracking).

1. On the laptop, choose a token and start the backend with it, as well as the key from above:

```powershell
$token = (& .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))").Trim()
Add-Content -Encoding ascii .env.face-demo "HEARTH_EDGE_FACE_TOKEN=$token"
.\scripts\start.ps1 -Lan
```

2. Enroll at `http://127.0.0.1:8000/room#recognition` **on the laptop** (enrollment stays loopback-only) and select **Also show my name on possible-fall and camera help-signal alerts**.
3. On the Jetson, once: `bash ~/hearth-watch/edge/tools/get_face_id_model.sh`. Then put the same token in a file readable only by you, and run:

```bash
python3 run_edge.py --face-id --face-id-token-file ~/.hearth-face-token
```

4. Face the camera from close enough that your eyes are at least 20 px apart in the frame. The status line counts face templates. A nurse's possible-fall alert for that person then reads **Possible fall · <alias>**, while other people's alerts stay unnamed.

## Logitech camera on Jetson (earlier notes)

The camera is just the video source. The laptop implementation sends the exact frame used by the posture model to a **local** recognition endpoint; the backend runs YuNet/SFace and attaches the verified identity to that observation's incident. It does not identify people by room alone or reuse the last name seen by an unrelated camera.

The intended Jetson layout is Logitech USB camera -> one capture process -> posture + face inference -> monitoring -> nurse dashboard. The current Python client is portable in structure, but requires the backend's Python 3.11+ environment, OpenCV YuNet/SFace APIs, and the fall-model dependencies. Older JetPack Python/OpenCV installations may need a suitable container or runtime. No Jetson hardware/performance validation has been performed for this addition.

Monitoring now accepts an optional same-frame poseNet body envelope; see [the integration contract](recognition-integration.md). The existing poseNet `edge/run_edge.py` has **not** been changed to recognize faces. Its integration still needs same-frame body boxes, explicit track continuity, and the face-model runtime adapted to its JetPack version. Do not run a second capture process against the same USB camera. Recognition endpoints intentionally remain loopback-only; moving the camera to a remote Jetson while leaving this backend on the laptop requires a separate authenticated, encrypted transport design, not simply exposing these routes over LAN.

## Data and limits

Face templates, aliases, consent timestamps, and appointment links are encrypted at rest in `data/face-demo.sqlite`. Camera frames are processed in memory and not saved by this feature. Enrollments expire after 24 hours and are purged on the next profile access or recognition attempt. Existing appointment records follow the project's existing storage behavior and are not encrypted by this feature. Use fictional appointment names. Named incident labels travel through the existing event/WebSocket/dashboard path, which is not encrypted storage by this feature. Withdrawing removes enrollment and stops future matches; it does not erase already issued incident records.

Face templates are pseudonymous biometric data, not anonymous data. Recognition uses a conservative demo threshold and an ambiguity margin, but these have not been validated for care use. There is no liveness or photo-spoof protection. Demo participants must test matching under the actual demo lighting and camera angle.

## Automated checks

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_patient_camera.py backend/tests/test_face_demo.py backend/tests/test_appointments.py -q
```
