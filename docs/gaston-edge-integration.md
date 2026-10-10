# Gaston edge integration

Integrated edge files from `origin/gaston/edge` at `ca7cf19` on 2026-10-10 into the monitoring checkout. Laptop model runners are retained. Changes remain uncommitted; the remote branch is unchanged.

## What Gaston implemented

- Jetson poseNet (`resnet18-body`), largest visible pose, normalized COCO keypoints.
- Pixel-space torso angle; normalized head/hip height; torso-only box aspect; smoothed vertical hip velocity with reset on missing hips/gaps.
- Standard-library HTTP client: observation batches every 0.15 seconds, up to 10 observations/second; heartbeats about once/second with FPS, latency and memory.
- Bounded queue (100 observations), retry/backoff, duplicate delivery possible after network failure. Camera inactivity pauses heartbeats after roughly two seconds.
- Updated OpenCV DNN face detection and rotated poseNet passes; optional short keypoint holds improve whole-body observation when lying across the frame. Missing face checks remain null.
- Replay now supplies run IDs to observations and heartbeats; backend-view overlays show monitor decisions.
- JSONL observation recording and replay tools; explicit optional video streaming, disabled by default.

## Monitoring support

Existing endpoints accept these payloads unchanged: `POST /ingest/observations` (an `observations` array) and `POST /ingest/heartbeat`. Missing `run_id`, `scenario_id`, and `posture_prediction` remain valid. The live config now uses `pose_fallback`: Hugging Face predictions are authoritative when present; observations without them use calibrated pose rules. Weak/unknown supplied model predictions never silently fall back to pose features. The model runner still requires camera-frame access, and separate sources are not automatically fused. Set `posture_mode: "model"` to require model-only evidence.

For pose-only observations, source status exposes pose-derived upright/floor-level posture without a model confidence score. Model observations display Standing, Sitting, or Lying from model predictions (`fallen` on the wire). A floor-level display does not itself mean an alert: the configured hold still applies. Missing/incomplete observations and stale sources display unknown. Face null does not generate a face-absent event. Observation staleness is tracked independently of heartbeats.

The updated edge sends unknown keypoint confidence as null/omitted rather than a fabricated score. Confidence-gated raised-arm gestures and zone downgrades remain conservative; the separate hand model produces scored fist-help predictions.

Camera calibration in the live config: floor head >= 0.40, floor hips >= 0.45, torso 60–120 degrees and box aspect >= 1.0, held for 3 seconds. Upright has its independent `upright_hip_y_max: 0.65`, torso <= 30 degrees and head above hips. These are specific to the reported mid-frame mat camera setup. The checker now uses the same upright bound. Test the actual clips before applying this calibration to another camera.

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

Gaston replay rewrites timestamps to playback wall time, uses source `replay-01`, and now supplies run IDs. Run at `--speed 1` for hold-time comparisons: faster playback shortens the observed floor interval.

Run clip checks with `.\.venv\Scripts\python.exe edge/tools/check_rules.py recordings/your_clip.jsonl --config backend/hearth/monitor/config.json`. Local actual clips were not available during integration; regression tests use synthetic observations matching the reported feature ranges, not a claim of rerunning the team's recordings.

Queued live observations can arrive late after reconnection. Monitoring drops edge observations older than `edge_observation_max_age_seconds` (default 3 seconds), or more than `edge_future_tolerance_seconds` (default 5 seconds) ahead of backend UTC. Dropped frames cannot refresh source health, advance a hold or recover an incident. Historical buffered incidents outside this window are not evaluated; preserve JSONL recordings for replay. Simulation/replay timestamps are exempt. Synchronize device clocks before the demo.

`POST /ingest/observations` returns `accepted`, `duplicates`, `out_of_order`, `stale`, and `future` counts. `GET /ingest/status` reports cumulative counts since backend startup plus configured timestamp limits. The edge client currently counts any successful HTTP batch as sent, so use these backend counters to distinguish transmitted frames from accepted frames. Heartbeats still represent transport liveness; they cannot clear observation staleness.

## Validation

`backend/tests/fixtures/gaston_edge.json` was generated with Gaston's actual contract and feature builders from synthetic pixel poses (not a hardware recording). Backend regression tests cover HTTP ingestion, WebSocket posture, one sustained-floor alert, recovery, sampled face events, deduplication, replay/live separation, unknown confidence, staleness and offline alerts. Hardware inference, network performance and thresholds still require a Jetson test.
