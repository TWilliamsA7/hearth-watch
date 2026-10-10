# Hearth

Single-room resident safety monitoring and simulation, built for Knight Hacks IX (October 9–11, 2026, 36 hours, team of 4).

Hearth shows how local edge computer vision could help caregivers in assisted-living and memory-care settings notice possible safety incidents involving older adults. A Jetson Nano processes one USB camera and sends structured posture and face-presence observations to a laptop. The laptop runs the monitoring pipeline, a deterministic room simulator, and a dashboard. Live and simulated observations go through the same pipeline, and simulated scenarios are scored against their ground truth.

**Hearth is not a game.** It is a care-monitoring simulation and evaluation environment. The primary hackathon target is the Full Sail challenge (a serious simulation using face detection and/or pathfinding). Face detection is the Full Sail feature; pose-based possible-fall detection is the safety application.

## Status

Repository scaffold and shared contract (`shared/`) only. The backend, edge, simulator and dashboard folders are empty placeholders. Everything under "Planned design" below is agreed but not yet implemented.

## Scope

**One room only.** This is a deliberate decision, not a limitation to fix.

In scope:

- One camera, one room, one person at a time.
- A deterministic one-room simulator with seeded, scripted scenarios: walking, sitting, bending, intentional lying down, possible fall, occlusion, camera outage, prolonged inactivity.
- A possible-fall state machine built from interpretable rules (torso angle, head/hip height, vertical motion, bounding-box aspect, time near floor). No custom model training.
- Face **detection** (is a face visible?) used to mark how well the person is observed.
- Evaluation: detection delay, true/false positives, missed events, test count.

Out of scope — do not add these:

- Multiple rooms, hallways, building maps, pathfinding, resident navigation.
- Multiple cameras or cross-camera tracking.
- Face **recognition**, identity assignment, biometric databases.
- Gait analysis, fall-risk scores, clinical predictions, custom fall-model training.
- Cloud services, internet dependencies, staff dispatch, physical props, robots.

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
3. **Every record says where it came from.** `source_type` is `edge`, `simulation` or `replay`, and the dashboard always labels live versus simulated data visibly.
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
scenarios/          scripted simulator scenarios
recordings/         local keypoint traces (git-ignored; never raw video)
```

## Shared contract

Everything that crosses a component boundary is defined in `shared/python/hearth_schema/models.py` (schema version `"1"`).

| Record | Meaning | Produced by |
| --- | --- | --- |
| `Observation` | What a sensor could see: keypoints, posture features, face presence | Jetson, simulator, replay |
| `Event` | What the monitor concluded | Backend monitor |
| `GroundTruth` | What actually happened in a simulated scenario segment | Simulator only |
| `Heartbeat` | Source liveness plus FPS, inference latency, memory | Every source, ~1 per second |
| `SourceStatus` | Backend's view of whether a source is online | Backend |
| `ServerMessage` | WebSocket envelope to the dashboard: `observation`, `event` or `source_status` | Backend |

Conventions:

- Timestamps are ISO 8601 UTC.
- Keypoints use the 18 COCO names from poseNet `resnet18-body` and are normalized to the frame: `(0, 0)` top-left, `(1, 1)` bottom-right.
- Posture features: `torso_angle_deg` (0 = upright, 90 = horizontal), `head_y` and `hip_y` (normalized, 0 = top of frame), `bbox_aspect` (width / height), `vertical_velocity` (change in `hip_y` per second, positive = moving down). All optional; omit what can't be measured.
- `face_present: null` means face detection did not run on that frame, which is different from `false`.
- Event types: `possible_fall`, `recovered`, `person_visible`, `person_not_visible`, `face_present`, `face_absent`, `source_offline`, `source_online`, `alert_acknowledged`.
- Ground-truth labels: `walking`, `standing`, `sitting`, `bending`, `lying_intentional`, `fall`, `occluded`, `camera_outage`, `absent`. Times are seconds from scenario start.
- Models reject unknown fields (`extra="forbid"`).

To change the contract: edit `models.py`, then regenerate the JSON Schema files and commit both.

```sh
pip install -e shared/python
python shared/scripts/export_schema.py
```

## Planned design (not yet implemented)

- **Transport:** the Jetson POSTs batches to `POST /ingest/observations` (body: `ObservationBatch`) at roughly 5–10 Hz, and a `Heartbeat` to `POST /ingest/heartbeat` every second. The backend marks a source offline after 3 seconds of silence. The dashboard subscribes to `WS /ws/room/{room_id}` and receives `ServerMessage`s.
- **Room ID:** `room-01`. Backend on port 8000, Vite dev server on 5173.
- **Possible-fall state machine:** roughly `upright → descending → floor_level → possible_fall` after a configurable hold time, with `recovered` on return to upright. All thresholds in one config file. Tune on real recorded keypoint traces.
- **Alerts:** open → acknowledged → resolved, with an acknowledge action in the dashboard.
- **Simulator:** deterministic clock and seed; play, pause, reset, replay, time scaling. The same seed must reproduce the same run.
- **Zones (stretch):** bed/chair regions may downgrade an alert but must not hide uncertainty.
- **Frontend types:** generate TypeScript types from `shared/schema/hearth.schema.json` rather than writing them by hand.

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
