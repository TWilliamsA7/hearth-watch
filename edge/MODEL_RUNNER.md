# Local standing / sitting / lying inference

> **Note (2026-10-10):** the team backend's `config.json` now trusts our own posture model by
> default (`fall_model_id` `hearth/posture-mlp`, `model_confidence_min` 0.5; see
> [POSTURE_MODEL.md](POSTURE_MODEL.md)). To use this runner's Hugging Face predictions with the
> backend, start the backend with a config whose `fall_model_id` is
> `melihuzunoglu/human-fall-detection` and `model_confidence_min` 0.65, via `HEARTH_CONFIG`.

`run_fall_model.py` runs [melihuzunoglu/human-fall-detection](https://huggingface.co/melihuzunoglu/human-fall-detection), a YOLO11 object detector with **Standing**, **Sitting**, and **Fallen** classes. Inference runs locally; only structured JSON goes to the existing Hearth API. It does not send or save frames. The checkpoint is pinned by revision and SHA-256 in `backend/hearth/perception/model.json`.

## Laptop setup (Python 3.11+)

Run from the repository root after setting up the backend:

```powershell
# CPU builds avoid installing CUDA packages on the laptop.
.\.venv\Scripts\python.exe -m pip install 'torch>=2.2,<3' 'torchvision>=0.17,<1' --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e './backend[vision]'
.\.venv\Scripts\python.exe edge/run_fall_model.py --download-only
.\.venv\Scripts\python.exe edge/run_fall_model.py --check
```

The weights are about 5.5 MB; PyTorch and other dependencies are larger. Downloads happen during setup. Normal inference uses the verified local weights and disables Ultralytics synchronization/automatic installation. `--check` runs the actual checkpoint on a blank frame without opening a camera; this tests runtime compatibility, not fall accuracy.

## Camera

For automatic help requests from a held fist, add `--hand-gestures` to both the download and camera commands. This runs `SuperSura/hand-gesture-yolo26n` on the same frames. See `docs/hand-help-signal.md` for the signal, hold/release rules and nurse subscriptions. Restart the backend after updating the shared schema.

Restart the backend after updating the shared schema, then run:

```powershell
.\.venv\Scripts\python.exe edge/run_fall_model.py --camera 0 --backend http://127.0.0.1:8000 --room room-01
```

The camera opens only when `--camera` is explicitly selected. Stop with Ctrl+C. Use `--preview` to opt into a local annotated OpenCV window (Q exits); no video is recorded. Use `--device cpu` (default), or a locally supported device. `--max-fps` defaults to 5; actual throughput depends on the device. `--max-frames 30` limits a test run. Match the backend port if you are using the preview on 8001.

Subscribe to **Possible falls** in `/nurses`. The source-health section and `/room` show the current model label and score. After a default 3 seconds of sufficiently confident Fallen observations, the backend creates one possible-fall alert. One second of sufficiently confident Standing observations records recovery. Sitting, no detections, multiple detections, low confidence, and stale sources cannot confirm recovery. The model confidence score is not a calibrated probability of a real fall.

## Existing image or video

```powershell
# One image produces one observation; it cannot satisfy a temporal alert hold.
.\.venv\Scripts\python.exe edge/run_fall_model.py --image path/to/image.jpg --dry-run
# Local recordings use video-frame time and are explicitly labelled replay.
.\.venv\Scripts\python.exe edge/run_fall_model.py --video path/to/recording.mp4 --backend http://127.0.0.1:8000
```

Use `--video-fps` only to correct missing/incorrect recording FPS metadata. Replay samples are selected at approximately `--max-fps`; the media timestamps, not processing speed, drive the hold. Every run gets a fresh run ID. `--dry-run` prints observations without sending heartbeats or observations to the backend.

## Contract and limits

- `Observation.posture_prediction` carries the model ID/revision, label, raw score, normalized bounding box, detection count, and uncertainty reason. It does **not** fabricate head/hip locations, pose keypoints, or face presence.
- Numeric class IDs are mapped using the checkpoint's own class names. An unexpected label set stops model loading. Multiple boxes produce an unknown result rather than selecting one person and possibly declaring another person recovered. This remains a one-person-at-a-time detector, without identity tracking.
- The optional `posture_mode: "model"` requires a Hugging Face prediction for posture and possible-fall decisions. Missing predictions remain unknown even when pose features exist. Explicit `posture_mode: "pose_fallback"` is available for legacy pose-only replay and simulator tests. If a prediction is present, that evidence path is authoritative; weak model output does not silently fall back to conflicting pose features. Switching models/evidence paths resets pending holds.
- Monitor thresholds: `model_confidence_min` (default 0.65), `floor_hold_seconds` (3), `model_standing_hold_seconds` (1), and `max_observation_gap_seconds` (1) in `backend/hearth/monitor/config.json`. Tune on staged traces. If measured frame intervals exceed the gap limit, the hold will correctly reset; adjust only after measuring the source.
- The detector has no pose/face output, so it cannot by itself drive the raised-arm help gesture or facial detection. Those still need the separate pose/face pipeline. Existing help-button functionality is independent.
- Raw images remain on whichever device runs this process. Laptop camera use is a local development/demo option, not a claim that Jetson inference is working.
- This Python 3.11+ Ultralytics runner is **not compatible with stock JetPack Python 3.6**. Jetson Nano deployment requires verifying a compatible PyTorch/Ultralytics environment or a tested exported runtime; no Nano performance is claimed.

Model source and attribution: Melih Uzunoğlu, `melihuzunoglu/human-fall-detection`, revision `97261b03632a9715a27481b2cfdac6a78e26d2d9`. The model repository declares AGPL-3.0. [Ultralytics prediction documentation](https://docs.ultralytics.com/modes/predict/) describes the result interface used by the adapter. This integration does not establish clinical accuracy.

The UI displays the checkpoint label `fallen` as **Lying**; the wire label remains `fallen` for compatibility. Lying is a posture, not proof of an accidental fall. Gaston's poseNet runner alone does not produce these model predictions. Run the model on the device with camera-frame access; the monitor cannot infer image classes from pose keypoints. PoseNet can still supply gesture and visibility observations.

### Contrast retry and lying preview

The runner now retries uncertain posture output with OpenCV CLAHE luminance enhancement (default `--posture-enhancement clahe`). Confident original standing/sitting predictions do not trigger the second pass. The retry is accepted only when the same Hugging Face model produces one known class at or above `--posture-retry-threshold` (default 0.65); scores are not inflated and wide boxes do not create lying labels. Multiple original person boxes remain unknown. The preview shows LYING/SITTING/STANDING/UNKNOWN and the selected input path, and observations/alerts retain that path for comparison. Use `--posture-enhancement none` for an A/B test. This adds inference cost on uncertain frames and is not evidence of improved accuracy until tested on your camera. Aim the camera so the whole body and floor are visible.

The integrated live backend config uses `pose_fallback`, allowing calibrated Jetson pose observations without a model prediction. Supplied model predictions remain authoritative. This combines supported evidence paths; it does not associate separate devices or residents automatically.
