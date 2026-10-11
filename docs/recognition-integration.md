# Recognition integration decision

> **Superseded design note.** The dashboard now includes enrollment and recognition check-in (`/room#recognition`), served by the same server; `/face-check-in` redirects there. See the root README and [face-demo.md](face-demo.md) for current behavior.

## Scope and repository findings

This review uses the local `monitoring`, `main`, `gaston/edge`, `feat/dashboard`, and `feat/evolved-simulation` refs available in this checkout. It does not fetch or change another branch. Recognition remains a staged, opt-in demo with fictional patient profiles.

- **Edge:** `edge/run_edge.py` owns the Logitech C270 capture, poseNet, rotated pose passes, and the outbound observations. `PersonTrack` follows a body by position but does not emit a persistent track identifier. The live edge currently omits `run_id`. Its res10 face detector reports presence only, and background face results can belong to an earlier frame. None of these signals establishes identity.
- **Hardware:** `edge/README.md` documents a Nano 2GB, JetPack 4.6.1, Python 3.6, and a laptop monitoring server over USB networking. The new Python 3.11+/OpenCV recognition runtime cannot be assumed to run on that stock installation.
- **Monitoring:** the existing fall/gesture rules consume sensor observations. Optional identity should annotate incidents before publication, without affecting thresholds, source health, acknowledgments, or resolution.
- **Simulation/evaluation:** anonymous `person_id` and the live avatar ID (`live-<source>` on main) are tracking/display identifiers, not patient identity. Ground truth must never enroll a profile or confirm recognition. Keep the shared Observation contract unchanged.
- **Dashboard:** `feat/dashboard` has React `ResidentPanel` and `AppointmentPanel`; the latter is still a placeholder for a connected scheduling service. The live React alert view does not yet display monitoring's recognized-patient metadata. The existing `/nurses` page on monitoring does. These are distinct frontends.

## Recommended placement

Keep one camera owner on the Jetson. Keep enrollment, encrypted templates, matching policy, consent, appointment links, and incident naming in monitoring. Use the existing Hugging Face YuNet/SFace models for recognition; do not replace the working poseNet fall path with the laptop YOLO model merely to obtain a body box.

For the hackathon, the known working recognition runtime is on the laptop. A Jetson-to-laptop integration therefore needs an authenticated encrypted connection carrying **only opted-in demo frames and their matching observation/body evidence**, with no disk recording. An SSH tunnel/relay is one deployment option, but it has not been implemented or validated here. Do not expose `/face-demo` over plain LAN HTTP or remove its loopback checks. Moving matching onto the Nano later requires a compatible model runtime and measured FPS/memory before it can be claimed to work.

A bridge that transmits frames changes edge's current "no raw frames leave the Jetson" statement, even when encrypted. Update that statement and demo consent before enabling such a bridge. If frames must stay entirely on the Jetson, port and benchmark local recognition there first instead. This change does not silently enable remote frame transport.

## Monitoring-side support implemented here

`POST /face-demo/observe` accepts the existing laptop payload unchanged. A poseNet adapter can optionally include `body`:

```json
{
  "session_id": "session returned by POST /face-demo/sessions",
  "frame": "base64 JPEG from the same captured frame",
  "observation": {"obs_id": "frame-42", "...": "complete shared Observation"},
  "body": {
    "detector": "posenet",
    "obs_id": "frame-42",
    "track_id": "anonymous-track-1",
    "detection_count": 1,
    "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9}
  }
}
```

The abbreviated observation above is illustrative; send the complete normal Observation. The poseNet request must retain its real `features` and `keypoints`, with `posture_prediction` absent/null. Do not fabricate confidence or a posture model label. Bboxes are normalized to the **original, unrotated full-frame image**. Reproject rotated-pass boxes before using them. Do not substitute a head crop's coordinate system.

- A nonempty live `run_id` is required; renew it when capture restarts. Use a session for one source/run/room.
- `body.obs_id` must equal `observation.obs_id`. Never attach a delayed face result to a newer observation.
- Use a new `track_id` when the tracked person is lost/reacquired. Track changes clear identity, including when the new body occupies a similar box.
- Supply the actual body count. Do not claim one person when an upright pass was skipped or the scene count is unknown; omit evidence/leave identity unconfirmed in that case.
- Face matching requires the face center inside the associated body box. Multiple people, a discontinuity, a missing enrollment, or ambiguous recognition clears identity.
- The original laptop YOLO path still derives its body evidence from `posture_prediction` when `body` is omitted.
- Recognition expires after 8 seconds without a fresh face match and survives face occlusion only while the same body remains continuously detected.
- The route returns ordinary ingestion counts plus `identity`. It ingests the observation itself: do not send the same successful observation again through `/ingest/observations`.
- The Python adapter also accepts plain observation/body dictionaries, avoiding a Pydantic dependency in a future edge bridge. This is transport preparation, not proof of Jetson runtime compatibility.

Face inference and enrollment extraction run on a worker thread. A lock prevents concurrent OpenCV/SQLite/session access; concurrent camera recognition requests receive 503 rather than accumulating stale work. Enrollment/withdrawal operations wait for current inference to finish. Normal monitoring routes and health polling do not use this lock. The current laptop producer still waits for each request/response, so this worker change does not guarantee its camera FPS under slow inference; measure that separately. The future edge bridge must use bounded background work and preserve ordinary observation delivery as described below. The camera client falls back to unnamed ordinary ingestion when recognition is unavailable. Keep this fallback when implementing the edge bridge.

`GET /face-demo/status` exposes `body_detectors`, `transport`, and `identity_ttl_seconds` for local capability checks. Detailed request schemas are available in FastAPI `/docs` and `/openapi.json`.

## Edge integration still required

1. Add a capture-run UUID and an anonymous tracking generation; keep both independent of patient names.
2. Copy pixels before overlays/camera buffer reuse, and pair them with the exact observation and original-frame body geometry.
3. Run the recognition transport outside the capture/pose loop with bounded work and a latest-frame policy. If selected observations use `/observe`, route each observation through exactly one ingestion path. Do not combine the existing batch sender with duplicate `/observe` submissions; preserve ordering and the monitor's maximum observation gap.
4. On congestion, failure, or unknown body count, retain normal unnamed observation/heartbeat delivery. Face matching must not stop fall detection.
5. Add authenticated encrypted transport or a compatible on-device runtime, then test camera reconnects, tracking changes, lying down, and performance on the actual Jetson.

No changes to `edge/` were made in this monitoring-only integration review. The section below supersedes steps 1-5 for multi-person Jetson sources.

## Jetson face templates (implemented after merging multi-person tracking)

The frame-bridge design above was replaced by computing the face template **on the Jetson** and sending only that. No frames or face crops leave the device, so edge's "no raw frames" statement still holds. The template is still pseudonymous biometric data.

- **Compute only when useful.** The existing res10 check takes each tracked person's head box in turn (`edge/hearth_edge/face_id.py`, `FaceTurns`). SFace runs only when res10 finds a face whose centre is inside that head box, and poseNet has that person's eyes and nose at least `FACE_ID_MIN_EYE_PX` (20 px) apart. SFace is the laptop's own pinned model and checksum, built for TensorRT FP16 (`edge/tools/get_face_id_model.sh`). It is aligned with those three keypoints and no landmark model. `edge/tools/face_id_agreement.py` measured 0.78-0.86 against laptop five-point enrollment on a real photo with 4 px keypoint jitter (threshold 0.55). A 12 px-eye face fell apart under jitter, which is why the minimum exists.
- **Transport.** `POST /face-demo/edge-embeddings` receives `{detector:"res10+sface", source, room_id, run_id?, person_id, obs_id, timestamp, face_box, embedding[128]}` (`backend/hearth/monitor/edge_identity.py`, `EdgeFaceTemplate`). The Jetson is not on loopback, so this one route takes `Authorization: Bearer $HEARTH_EDGE_FACE_TOKEN` instead (constant-time comparison; refused with any `Origin` header, closed when the token is unset or empty). All other face-demo routes stay loopback-only, and the token does not open them. Plain HTTP over the USB link: the token and templates are not encrypted in transit.
- **Identity per track.** A name belongs to `(room, source, run_id, person_id)`. It needs three consecutive matches within 3 s, and an unknown or ambiguous face clears it. It lasts 8 s after the last match. Any of these ends it: the person's "not detected" end observation, the source going offline, withdrawal or enrollment change, or expiry. One enrollment names at most one live track; two tracks matching the same face both lose it. Templates must belong to a person whose latest observation is within one observation gap.
- **Labelling.** `MonitorService.incident_labeler` adds `resident_label`, `patient_profile_id`, `identity_method: "edge_face_track"` and `identity_demo` to that person's `possible_fall` incident only. A hand-gesture help request stays anonymous, because the hand is not tied to a body here. Ordinary ingest still cannot assert a name: the label comes from server-side state.
- **run_id.** The live edge sends no `run_id`, since a per-run id would change the source key and strand the previous run's offline alert. Person ids restart at `p1` after a restart, so the source going offline (a restart always outlasts the 3 s source timeout) clears every name on that source.
- **Edge never learns names.** The route answers only `status`, `matches` and `checked_in`.

Measured on the Nano: the SFace FP16 engine builds with TensorRT 8.2 (trtexec PASSED, 20 MB) and runs at 17.5 ms per face of GPU compute. Not yet measured: FPS and memory with poseNet, the posture MLP and SFace together, and poseNet eye/nose accuracy at the C270's real distances. Id swaps when two people cross remain a tracking limitation; the uniqueness rule and TTL limit, but do not prevent, a swapped name.

## Dashboard integration contract

The existing alert/event transport already carries the fields under `metadata`; no new shared schema or WebSocket message type is needed:

- `resident_label`: enrolled demo display name.
- `patient_profile_id`: enrollment identifier, not the anonymous `Observation.person_id`.
- `identity_method`: `face_match` or `recent_face_body_track` (laptop camera), `edge_face_track` (Jetson templates).
- `identity_demo`: true for this staged workflow.

React should display **Possible fall — <resident_label>** only when `identity_demo` is true and the label is present. Otherwise use **Identity unconfirmed**. Treat identity on an alert as evidence at that incident, not the current occupant of a room. Render names as text, never HTML. Do not infer identity from `source`, simulation names, or `person_id`.

Link to the backend's `/face-check-in` page for enrollment for now. A cross-origin React dev page cannot call the local enrollment endpoints directly under the current same-origin policy; changing that needs an explicit authenticated enrollment design. Connect `AppointmentPanel` to `/appointments` and use explicit profile-to-appointment links. Matching alone should not acknowledge or resolve a nurse alert.

## Merge and validation

Only the monitoring working tree is edited. Do not wholesale copy main's simulator/dashboard files into this branch. Preserve the ordinary ingest and publish paths when merging; resolve `api/app.py` wiring deliberately and retain the newer simulation mount on main. Recognition does not import simulator/evaluator code.

Tests cover poseNet observations without YOLO, exact-frame binding, track changes, multiple people, the check-in-only consent regression, local-only transport, and monitoring responsiveness while recognition is slow. These are software tests with synthetic face embeddings, not face-accuracy measurements. Actual YuNet/SFace weights were previously smoke-tested on the laptop; full Jetson and React integration remain unverified.
