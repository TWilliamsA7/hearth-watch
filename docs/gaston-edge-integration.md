# Gaston edge integration

Reviewed `origin/gaston/edge` at `9d9ccba` on 2026-10-10. Monitoring changes live on `monitoring`; Gaston's branch was not merged or edited.

## What Gaston implemented

- Jetson poseNet (`resnet18-body`), largest visible pose, normalized COCO keypoints.
- Pixel-space torso angle; normalized head/hip height; torso-only box aspect; smoothed vertical hip velocity with reset on missing hips/gaps.
- Standard-library HTTP client: observation batches every 0.15 seconds, up to 10 observations/second; heartbeats about once/second with FPS, latency and memory.
- Bounded queue (100 observations), retry/backoff, duplicate delivery possible after network failure. Camera inactivity pauses heartbeats after roughly two seconds.
- Optional facenet checks every 0.5 seconds; intervening observations use `face_present: null`. This is detection, not recognition.
- JSONL observation recording and replay tools; explicit optional video streaming, disabled by default.

## Monitoring support

Existing endpoints accept these payloads unchanged: `POST /ingest/observations` (an `observations` array) and `POST /ingest/heartbeat`. Missing `run_id`, `scenario_id`, and `posture_prediction` remain valid. The default posture mode now requires Hugging Face predictions for standing/sitting/lying and possible-fall decisions. Gaston's pose-only observations remain accepted for visibility and gestures but have unknown posture. The model runner needs camera-frame access; no image classifier runs on received keypoints. Use explicit `posture_mode: "pose_fallback"` only for legacy pose testing.

In the optional legacy pose-fallback mode, source status exposes pose-derived upright/floor-level posture without a model confidence score. The default model mode displays Standing, Sitting, or Lying from model predictions (`fallen` on the wire). A floor-level display does not itself mean an alert: the configured hold still applies. Missing/incomplete observations and stale sources display unknown. Face null does not generate a face-absent event. Observation staleness is tracked independently of heartbeats.

Keypoint confidence may now be omitted/null. PoseNet does not expose a measured per-keypoint score. The existing edge placeholder `1.0` remains accepted for compatibility, but is not a measured probability. **Edge follow-up:** set `KEYPOINT_CONFIDENCE_PLACEHOLDER = None` and replace `test_contract_still_requires_keypoint_confidence` with a test asserting omitted confidence validates. Until that update, gesture and zone confidence gates cannot distinguish the placeholder from a real score. With unknown confidence, monitoring conservatively disables confidence-gated gestures and zone downgrades; posture-feature falls still work and the help button remains available.

## Connect the two machines

On the laptop, from the monitoring checkout:

```powershell
.\.venv\Scripts\python.exe -m uvicorn hearth.api.app:app --host 0.0.0.0 --port 8000
```

On the Jetson, from Gaston's checkout:

```sh
python3 edge/run_edge.py --backend http://192.168.55.100:8000 --source jetson-01 --room room-01
```

Use the laptop's actual reachable IP if it differs; allow port 8000 on the private network. Stop the edge stub backend if it occupies the same port. Open `/nurses` on the laptop and select room-01 and Possible falls. Check `/sources` for live telemetry. Pose thresholds still need tuning against staged recordings with the actual camera framing.

Gaston replay rewrites timestamps to playback wall time and uses source `replay-01`. Run at `--speed 1` for hold-time comparisons: faster playback shortens the observed floor interval. It currently supplies no run ID, so repeated runs share a source history. For evaluation isolation, use a new `--source` per run (or add matching run IDs to observations and heartbeats upstream).

Queued live observations can arrive late after reconnection. Monitoring drops edge observations older than `edge_observation_max_age_seconds` (default 3 seconds), or more than `edge_future_tolerance_seconds` (default 5 seconds) ahead of backend UTC. Dropped frames cannot refresh source health, advance a hold or recover an incident. Historical buffered incidents outside this window are not evaluated; preserve JSONL recordings for replay. Simulation/replay timestamps are exempt. Synchronize device clocks before the demo.

`POST /ingest/observations` returns `accepted`, `duplicates`, `out_of_order`, `stale`, and `future` counts. `GET /ingest/status` reports cumulative counts since backend startup plus configured timestamp limits. The edge client currently counts any successful HTTP batch as sent, so use these backend counters to distinguish transmitted frames from accepted frames. Heartbeats still represent transport liveness; they cannot clear observation staleness.

## Validation

`backend/tests/fixtures/gaston_edge.json` was generated with Gaston's actual contract and feature builders from synthetic pixel poses (not a hardware recording). Backend regression tests cover HTTP ingestion, WebSocket posture, one sustained-floor alert, recovery, sampled face events, deduplication, replay/live separation, unknown confidence, staleness and offline alerts. Hardware inference, network performance and thresholds still require a Jetson test.
