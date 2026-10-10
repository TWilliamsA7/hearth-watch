# Hearth — Single-Room Resident Safety Monitoring and Simulation

## Project context

Hearth is a 36-hour Knight Hacks IX (October 9–11, 2026) hackathon project for a team of 2–3. The primary target is the Full Sail challenge: build a serious game or training simulation using pathfinding, face detection, or both to help someone achieve a goal, practice a skill, or respond to a scenario. Hearth is **not a game**. It is a **single-room care-monitoring simulation and evaluation environment** intended to demonstrate how local edge computer vision can help caregivers notice possible safety incidents involving older adults, particularly in assisted-living and memory-care settings.

**Scope decision: One room only.** Do not implement a building floor plan, simulated hallways, multiple rooms, resident navigation, cross-camera tracking, facility-wide resident schedules, or pathfinding merely to satisfy the challenge. Face detection is the intended explicit Full Sail feature; pose-based possible-fall detection is the primary safety application. The training/simulation component tests monitoring and response behavior under controlled, repeatable room scenarios. This interpretation of sponsor fit is not guaranteed; keep the training/evaluation purpose visible in the demo.

## Core product and demonstration

The laptop displays a digital twin of one monitored room: a room outline with configured zones (floor, bed/chair if represented), a person/skeleton visualization, activity state, live incident alerts, event timeline, and a scenario/evaluation panel. The Jetson Nano 2GB processes one USB camera locally and sends structured observations/events to the laptop. The same monitoring/alert pipeline also consumes simulated room observations. The user can run reproducible scenarios (ordinary walking, sitting, bending, intentional lying down, possible fall, occlusion/camera outage) and inspect detection latency, missed events, and false alarms. The interface clearly labels live versus simulated data.

A 90-second demo should show: (1) one-room simulator running a normal-activity scenario; (2) real camera detection of a consenting teammate, including face presence and pose skeleton; (3) teammate safely lowering onto a mat to trigger a **possible-fall** alert; (4) a normal sit/bend scenario that does not trigger the same alert, if validated; (5) replay of a simulated incident and its measured alert delay/ground truth. Do not ask a judge to actually fall. Do not imply that lying down proves a fall or that the system is clinically validated.

## Hardware and constraints

- NVIDIA Jetson Nano **2GB** as the edge CV node.
- One Arducam USB or Logitech USB webcam; use **one camera for MVP**.
- CPU-only Framework 13 laptop for FastAPI, simulation, and React UI.
- Direct local network/Ethernet where practical; demo should work without internet.
- No printed models, servos, moving robots, multiple rooms, or physical construction required.
- Jetson memory and JetPack/Python compatibility are high risks. Validate camera and pose inference early; do not assume any model name or FPS target works without testing on the device.

## Functional requirements

### P0 — required end-to-end slice

1. **One-room simulator:** A deterministic room state machine with a simple 2D representation and seeded scripted scenarios. Maintain separate **ground truth** (what happened) and **observations** (what the monitor could know). Include normal movement, sitting/bending, and possible-fall/lying-down scenarios. Allow play/pause, reset, replay, and optionally time scaling.
2. **Jetson perception:** Capture from one camera; run a compatible person/pose model (test `jetson-inference`/`poseNet` if installed and supported). Extract keypoints or basic posture observations. Measure actual FPS, inference latency, and memory use.
3. **Possible-fall state machine:** Use temporal posture features (e.g., torso angle, relative head/hip height, vertical motion, bounding-box aspect ratio, time near floor) to emit `possible_fall` after a configurable threshold. Add `recovered`/resolved transitions. Distinguish sustained floor-level posture from a medically confirmed fall. Start with interpretable rules, not custom training.
4. **Unified event interface:** Live Jetson and simulator events share one schema, including timestamp, source type, room ID, event type, confidence when meaningful, and metadata. The backend and UI must never silently treat simulation ground truth as a sensor observation.
5. **Single-room dashboard:** Show room activity, skeleton/keypoints or stylized person state, connection status, alert rail, alert acknowledgement, and event timeline. Clearly identify simulated vs. live observations.
6. **Evaluation:** Compare simulated scenario ground truth against generated alerts. Report detection delay, true/false positives, missed events, and the test count. For live staged tests, report small-sample results without claims of clinical accuracy.

### P1 — Full Sail alignment and useful extensions

7. **Face detection:** Run a compatible lightweight face detector on the Jetson, ideally at reduced frequency if needed. Emit `face_present`/`face_absent` or a face detection observation. Use face presence meaningfully to mark whether a person is visible/observed in the room, and show how partial observation affects interpretation. Face **detection is not identity recognition**. Do not make enrollment or biometric identification necessary for MVP.
8. **Room zones:** Configurable chair/bed/floor regions to reduce obvious false alarms; zone suppression must not hide genuine uncertainty.
9. **Scenario variations:** Occlusion, delayed observations, camera outage, prolonged inactivity, and intentional lying down; demonstrate failure modes and monitoring limits.
10. **Training/evaluation mode:** Present scenario assumptions, run a monitoring scenario, and produce an after-action report showing what the system observed versus what actually happened. This is a serious simulation for understanding and evaluating resident-monitoring workflows, not a game with points or invented clinical recommendations.

### Explicitly out of scope

- Building-scale simulation, multiple rooms/hallways, pathfinding, multiple cameras, cross-camera identity tracking.
- Face recognition, resident biometric databases, automatic identity assignment.
- Gait speed/variability, long-term personalized fall-risk scores, clinical predictions, custom fall-model training.
- HF-generated handoffs, Neon/cloud dependencies, 14-day histories, staff dispatch mechanics, and elaborate physical props.

## Architecture

```text
ONE REAL ROOM                         ONE SIMULATED ROOM
USB camera                            Scripted resident activity
    |                                 + incident ground truth
Jetson Nano 2GB                              |
  frame capture                       observation generator
  person / pose detection                    |
  optional face detection                    |
  posture state machine                      |
    |                                        |
    +---- structured live events ------------+---- simulated observations
                                             |
                                    FastAPI on laptop
                                      event ingestion
                                      alert state machine
                                      scenario controller
                                      evaluation / event log
                                             |
                                        WebSocket/HTTP
                                             |
                                    React/Vite dashboard
                                    single-room visualization
                                    alerts, replay, evaluation
```

**Critical invariant:** Simulator ground truth is stored separately from observations and alerts. The monitoring algorithm cannot use ground truth to make its decisions. A live observation must actually affect the same event/alert interface used by the simulation.

## Suggested stack

- **Edge:** Python compatible with the installed JetPack, `jetson-inference`/TensorRT where available, V4L2/GStreamer/OpenCV as supported. Avoid assuming modern Python packages run on the Jetson's older environment.
- **Backend:** Python 3.11+, FastAPI, WebSockets or simple HTTP ingestion; SQLite or in-memory persistence initially.
- **Simulation:** Python deterministic state machine with timestamped observation generation, scenario seeds, and ground-truth annotations.
- **Frontend:** React + TypeScript + Vite; SVG/Canvas room and skeleton display; no 3D engine required.
- **Network:** LAN; no external services needed for core operation.

### Example event contract

```json
{
  "event_id": "evt-001",
  "timestamp": "2026-10-10T02:15:00Z",
  "source": "jetson-01",
  "source_type": "edge",
  "room_id": "room-01",
  "event_type": "possible_fall",
  "confidence": null,
  "metadata": {
    "posture": "floor_level",
    "duration_seconds": 3.2
  }
}
```

Simulation events use `source_type: "simulation"` and a `scenario_id`. Keep a separate ground-truth record for evaluation. Use the same event contract in stubs before Jetson integration.

## Team ownership

- **Developer A — Edge/CV:** Camera setup, pose inference, face detection (P1), posture state machine, event transport, FPS/memory benchmarks.
- **Developer B — Backend/simulation:** Scenario state machine, ground truth vs observations, alert rules, API, event persistence, evaluation.
- **Developer C — UI/integration:** One-room visualization, skeleton display, incident alerts, scenario controls, replay, evaluation results, demo flow.
- **If only two developers:** Combine backend and UI, and prioritize the single complete scenario over extra features.

## 36-hour schedule and gates

| Time | Main work | Exit gate |
|---|---|---|
| 0–4 h | Verify Jetson camera/pose feasibility; agree on event schema; scaffold FastAPI, simulator, UI | A synthetic event appears in the room dashboard; camera captures frames |
| 4–10 h | Build scripted room scenarios and ground-truth logging; run one-camera pose inference; draw skeleton/person state | Both simulated observations and real pose data can reach the backend |
| 10–18 h | Implement temporal possible-fall logic, alert states, acknowledgement, and basic evaluation | Controlled safe lying-down demo produces a possible-fall alert; simulated scenario reports latency |
| 18–26 h | Integrate face detection if feasible; add chair/bed suppression, occlusion/outage scenarios, replay | Complete demo flow works; negative tests and disconnect behavior are visible |
| 26–32 h | Freeze features, benchmark, test false positives, rehearse, record backup demo, write submission | Three successful end-to-end rehearsals |
| 32–36 h | Buffer and submission | No new features |

**Fallback rule:** If edge pose inference remains blocked, continue simulation/UI development with recorded or synthetic observations, but explicitly label the missing live capability. If face detection is not working, do not claim to have met Full Sail's face-detection requirement; prioritize getting a minimal real face detector working before adding optional analytics.

## Verification checklist

- Camera capture and Jetson inference run stably on the actual Nano 2GB; log observed FPS, memory, and latency.
- One-room simulated normal-activity and possible-fall scenarios are repeatable using the same seed.
- Real and simulated observations use the same event contract and display clearly different source labels.
- Possible-fall alert fires on a controlled safe floor-level posture; ordinary sitting/bending are tested for false positives.
- Face detector actually detects faces in a live feed (if claimed for Full Sail); face detection is not presented as identification.
- Simulated evaluation compares alerts with ground truth without leaking truth into the alert algorithm.
- Disconnecting the Jetson shows an offline/stale observation state rather than silently implying the room is safe.
- Dashboard, replay, and evaluation work without external internet access.
- Rehearse the 90-second demo three times from a cold start.

## Privacy, ethics, and claims

- Keep raw camera frames on the Jetson; do not persist or upload video by default.
- Send minimal structured events and keypoints. Skeleton data can still be sensitive; don't claim it is inherently anonymous.
- Use consenting teammates and synthetic resident profiles only.
- Do not implement biometric identification unless separately justified and consented to.
- Label synthetic histories and simulated incidents. Never present them as real patient evidence.
- Frame alerts as possible safety events for human review, not clinical diagnoses or certified fall detection.
- Do not claim HIPAA compliance or clinical validation.

## Full Sail pitch positioning

**Hearth is a single-room, privacy-conscious resident safety monitoring and training/evaluation simulation.** It uses real edge computer vision (including face detection) to produce observations and potential safety alerts. Its deterministic room simulator reproduces ordinary activity, possible falls, and perception failures so users can inspect what the monitoring system saw, what actually happened, and how reliably it responded. The objective is to explore how such systems could assist care professionals in recognizing resident safety incidents, not to replace professional judgment.

The key demo claim must be supported by actual implementation: **a real Jetson observation enters the same monitoring pipeline as simulated room observations, and the simulation can evaluate the consequences of imperfect perception.**

## Instructions to implementation agent

Implement this as an end-to-end working project, starting with the shared event contract and a single-room simulation slice. Work in parallel on Jetson feasibility and the laptop application. Do not expand scope to multiple rooms, a facility map, extra hardware, or a game. Do not invent clinical efficacy or measured results. Make incremental integrations early and prefer reliable, measurable behavior over extra features. When a hardware/model dependency is unverified, test it first and adapt to what actually runs on the Jetson Nano 2GB.
