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

Existing endpoints accept these payloads unchanged: `POST /ingest/observations` (an `observations` array) and `POST /ingest/heartbeat`. Missing `run_id`, `scenario_id`, and `posture_prediction` remain valid. Pose features use the original temporal rules; the Hugging Face model is a separate optional producer, not required on the Nano.

Source status now exposes pose-derived upright/floor-level posture to the nurse/room UI without claiming a model confidence score. A floor-level display does not itself mean an alert: the configured hold still applies. Missing/incomplete observations and stale sources display unknown. Face null does not generate a face-absent event. Observation staleness is tracked independently of heartbeats.

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

Queued live observations can arrive late after reconnection. Current monitor timing uses observation timestamps for holds and receipt time for source health; a buffered incident may therefore appear after reconnection. Synchronize device clocks and do not interpret delivery time as incident time.

## Validation

`backend/tests/fixtures/gaston_edge.json` was generated with Gaston's actual contract and feature builders from synthetic pixel poses (not a hardware recording). Backend regression tests cover HTTP ingestion, WebSocket posture, one sustained-floor alert, recovery, sampled face events, deduplication, replay/live separation, unknown confidence, staleness and offline alerts. Hardware inference, network performance and thresholds still require a Jetson test.
