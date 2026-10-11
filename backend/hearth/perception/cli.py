"""Camera inference with optional local annotated dashboard preview."""
import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from .fall_model import ModelSettings, download_weights, load_model, observation_from_result
from .hand_model import settings as hand_settings, LABELS as HAND_LABELS, prediction_from_result as hand_prediction
from .posture_vision import infer_posture, draw_status


def post(base, endpoint, body):
    request = Request(base.rstrip("/") + endpoint, data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=2) as response:
        return json.load(response)


class Recorder:
    """Writes observations and ~1 Hz heartbeats as JSONL envelopes, stamped on record time.

    The file is what python -m hearth.eval real (and the room view's recorded feed) read:
    {"type": "observation" | "heartbeat", "data": {...}} per line."""

    def __init__(self, path):
        from hearth_schema import Heartbeat
        self._heartbeat_model = Heartbeat
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._last_beat = None

    def write(self, observation, context, metrics):
        if self._last_beat is None or (observation.timestamp - self._last_beat).total_seconds() >= 1:
            beat = self._heartbeat_model(timestamp=observation.timestamp, fps=metrics.get("fps"),
                                         inference_ms=metrics.get("inference_ms"), **context)
            self._line("heartbeat", beat)
            self._last_beat = observation.timestamp
        self._line("observation", observation)

    def _line(self, kind, model):
        self._file.write(json.dumps({"type": kind, "data": model.model_dump(mode="json")}) + "\n")

    def close(self):
        self._file.close()


def main(argv=None):
    config = ModelSettings.load()
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--camera", type=int, help="Local camera index; opens only when explicitly selected")
    inputs.add_argument("--video", type=Path, help="Local prerecorded video; labelled replay")
    inputs.add_argument("--image", type=Path, help="One local image; labelled replay; cannot satisfy a temporal hold")
    inputs.add_argument("--check", action="store_true", help="Load real weights and infer on a blank frame; no camera/backend")
    inputs.add_argument("--download-only", action="store_true")
    parser.add_argument("--weights", type=Path, default=Path("models/human-fall-detection/best.pt"))
    parser.add_argument("--runtime-dir", type=Path, default=Path(".hearth-runtime"))
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    parser.add_argument("--room", default="room-01")
    parser.add_argument("--source-id", default="fall-model-01")
    parser.add_argument("--device", default=config.device)
    parser.add_argument("--max-fps", type=float, default=config.max_fps)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--video-fps", type=float, help="Override missing/incorrect video FPS metadata")
    parser.add_argument("--dashboard-preview", action="store_true", help="Share annotated frames with the local dashboard; no frame recording")
    parser.add_argument("--preview", action="store_true", help="Show local annotated video; Q closes it")
    parser.add_argument("--dry-run", action="store_true", help="Print observations without contacting backend")
    parser.add_argument("--record", type=Path, help="Also write observations and heartbeats to this JSONL file "
                        "(score it with python -m hearth.eval real). With --dry-run --video, runs as fast as possible")
    parser.add_argument("--hand-gestures", action="store_true", help="Also classify hand gestures on the same frames")
    parser.add_argument("--hand-weights", type=Path, default=Path("models/hand-gesture/gesture.pt"))
    parser.add_argument("--posture-enhancement", choices=["none", "clahe"], default="clahe",
                        help="Retry uncertain posture detections with OpenCV local contrast")
    parser.add_argument("--posture-retry-threshold", type=float, default=.65,
                        help="Score below which to retry; match backend model_confidence_min")
    parser.add_argument("--help-gesture", choices=["fist", "palm", "call"], default="fist",
                        help="Preview guidance only; backend hand_help_label controls alerts")
    parser.add_argument("--patient-recognition", action="store_true",
                        help="Opt-in staged patient demo: match faces locally and name fall alerts")
    parser.add_argument('--laptop', action='store_true', help='CPU preset: posture 384px, hand 640px, no contrast retry')
    args = parser.parse_args(argv)
    if args.laptop:
        config = replace(config, image_size=384)
        args.posture_enhancement = 'none'
    if args.dashboard_preview:
        from urllib.parse import urlparse
        if args.camera is None or args.dry_run or urlparse(args.backend).hostname not in {"localhost", "127.0.0.1", "::1"}:
            parser.error("--dashboard-preview requires a live camera and local backend")
    if args.patient_recognition and (args.camera is None or args.dry_run):
        parser.error("--patient-recognition requires a live --camera and local backend")
    patient_client = None
    if args.patient_recognition:
        from .patient_client import PatientClient
        patient_client = PatientClient(args.backend, args.room)
    if not math.isfinite(args.max_fps) or args.max_fps <= 0:
        parser.error("--max-fps must be positive and finite")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if not math.isfinite(args.posture_retry_threshold) or not 0 < args.posture_retry_threshold <= 1:
        parser.error("--posture-retry-threshold must be in (0,1]")
    if args.download_only:
        path = download_weights(args.weights, config)
        if args.hand_gestures:
            print(f"Verified hand weights: {download_weights(args.hand_weights, hand_settings()).resolve()}")
        print(f"Verified weights: {path.resolve()}\nRevision: {config.revision}")
        return
    model = load_model(args.weights, config, args.runtime_dir)
    hand_config = hand_settings() if args.hand_gestures else None
    if hand_config and args.laptop:
        hand_config = replace(hand_config, image_size=640)
    hand_model = (load_model(args.hand_weights, hand_config, args.runtime_dir, expected_labels=HAND_LABELS)
                  if hand_config else None)
    import cv2
    import numpy as np
    if args.check:
        started = time.perf_counter()
        results = model.predict(np.zeros((config.image_size, config.image_size, 3), dtype=np.uint8),
                                device=args.device, imgsz=config.image_size,
                                conf=config.detection_confidence_min, agnostic_nms=True, save=False, verbose=False)
        print(json.dumps({"model": config.repo_id, "revision": config.revision, "labels": model.names,
                          "blank_frame_detections": len(results[0].boxes),
                          "cold_inference_ms": round((time.perf_counter()-started)*1000, 1)}))
        if hand_model:
            hand_result = hand_model.predict(np.zeros((800,800,3),dtype=np.uint8),device=args.device,
                imgsz=hand_config.image_size,conf=hand_config.detection_confidence_min,save=False,verbose=False)[0]
            print(json.dumps({"hand_model":hand_config.repo_id,"labels":hand_model.names,
                              "blank_prediction":hand_prediction(hand_result,hand_config).model_dump(mode="json")}))
        return
    cap = None
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    elif args.video:
        if not args.video.is_file():
            parser.error("Video file does not exist")
        cap = cv2.VideoCapture(str(args.video))
    if cap is not None and not cap.isOpened():
        cap.release()
        raise RuntimeError("Cannot open camera/video source")
    camera_reader = None
    preview_sender = None
    camera_sequence = 0
    if args.camera is not None:
        from .live_io import LatestCamera
        camera_reader = LatestCamera(cap)
    if args.dashboard_preview:
        from .live_io import LatestPreview
        from urllib.parse import quote
        from urllib.request import build_opener, ProxyHandler
        preview_url = args.backend.rstrip('/') + f"/camera-preview/{quote(args.room, safe='')}/{quote(args.source_id, safe='')}"
        preview_opener = build_opener(ProxyHandler({}))
        def send_preview(annotated):
            scale = min(1., 640/annotated.shape[1])
            resized = cv2.resize(annotated, (round(annotated.shape[1]*scale), round(annotated.shape[0]*scale)))
            ok, jpeg = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ok:
                request = Request(preview_url, data=jpeg.tobytes(), headers={'Content-Type':'image/jpeg'}, method='POST')
                with preview_opener.open(request, timeout=1) as reply:
                    reply.read()
        preview_sender = LatestPreview(send_preview)
    video_fps, stride = None, 1
    if args.video:
        video_fps = args.video_fps if args.video_fps is not None else cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(video_fps) or video_fps <= 0:
            cap.release()
            parser.error("Video FPS is unknown; set --video-fps to the recording frame rate")
        stride = max(1, round(video_fps / args.max_fps))
    start_time = datetime.now(timezone.utc)
    if args.video:
        length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if math.isfinite(length) and length > 0:
            start_time -= timedelta(seconds=length/video_fps)
    context = dict(source=args.source_id, source_type="edge" if args.camera is not None else "replay",
                   room_id=args.room, run_id=str(uuid4()))
    metrics = {"fps": None, "inference_ms": None}
    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            try:
                post(args.backend, "/ingest/heartbeat", dict(context, timestamp=datetime.now(timezone.utc).isoformat(), **metrics))
            except (OSError, URLError, ValueError):
                pass  # The observation sender reports transport errors.
            stop.wait(1)

    thread = None
    if not args.dry_run:
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
    recorder = Recorder(args.record) if args.record else None
    processed, frame_index = 0, -1
    last_warning = float("-inf")
    last_preview = float("-inf")
    previous_frame_time = None
    started = time.monotonic()
    try:
        while True:
            tick = time.monotonic()
            if args.image:
                frame = cv2.imread(str(args.image))
                if frame is None:
                    raise RuntimeError("Cannot read image")
            else:
                if camera_reader:
                    frame, capture_time, camera_sequence = camera_reader.read(camera_sequence)
                    ok = True
                else:
                    ok, frame = cap.read()
                if not ok:
                    if args.camera is not None:
                        raise RuntimeError("Camera stopped delivering frames")
                    break
                frame_index += 1
                if args.video and frame_index % stride:
                    continue
            timestamp = (start_time + timedelta(seconds=frame_index/video_fps)
                         if args.video else capture_time if camera_reader else datetime.now(timezone.utc))
            frame_gap = (timestamp - previous_frame_time).total_seconds() if previous_frame_time else 0
            previous_frame_time = timestamp
            inference_start = time.perf_counter()
            observation, result = infer_posture(model, frame, settings=config, device=args.device,
                context=context, timestamp=timestamp, enhancement=args.posture_enhancement,
                retry_threshold=args.posture_retry_threshold)
            hand_result = (hand_model.predict(frame, device=args.device, imgsz=hand_config.image_size,
                conf=hand_config.detection_confidence_min, agnostic_nms=True, save=False, verbose=False)[0]
                           if hand_model else None)
            metrics["inference_ms"] = (time.perf_counter()-inference_start)*1000
            if hand_result is not None:
                observation.hand_gesture_prediction = hand_prediction(hand_result, hand_config)
            patient_caption = "Identity unconfirmed"
            processed += 1
            metrics["fps"] = processed / max(time.monotonic()-started, .001)
            if recorder is not None:
                recorder.write(observation, context, metrics)
            if args.dry_run:
                print(observation.model_dump_json(), flush=True)
            else:
                try:
                    response = None
                    if patient_client:
                        try:
                            response = patient_client.observe(frame, observation)
                            identity = response.get("identity", {})
                            name = identity.get("profile", {}).get("alias", "Identity unconfirmed")
                            patient_caption = name + (" (recent match)" if identity.get("status") == "tracked" else "")
                            print(f"patient: {name} ({identity.get('status', 'unknown')})", flush=True)
                        except (OSError, ValueError) as exc:
                            patient_client.close()
                            if time.monotonic()-last_warning >= 5:
                                print(f"Recognition unavailable; sending unnamed observation: {exc}", file=sys.stderr)
                                last_warning = time.monotonic()
                    if response is None:
                        response = post(args.backend, "/ingest/observations", {"observations": [observation.model_dump(mode="json")]})
                    if response.get("stale", 0) or response.get("future", 0):
                        if time.monotonic()-last_warning >= 5:
                            print(f"Backend dropped frame outside live timestamp limits: {response}. "
                                  "Check device clock and inference latency.", file=sys.stderr)
                            last_warning = time.monotonic()
                    elif response.get("accepted") != 1:
                        raise RuntimeError(f"Observation was not accepted: {response}")
                except HTTPError as exc:
                    raise RuntimeError(f"Backend rejected observation ({exc.code}): {exc.read().decode()}") from exc
                except (OSError, URLError) as exc:
                    if time.monotonic()-last_warning >= 5:
                        print(f"Backend unavailable; frame dropped, will retry with fresh observations: {exc}", file=sys.stderr)
                        last_warning = time.monotonic()
                prediction = observation.posture_prediction
                print(f"{context['source_type']} {args.room}: {prediction.label} "
                      f"score={prediction.confidence} detections={prediction.detection_count} "
                      f"input={prediction.input_processing} inference_ms={metrics['inference_ms']:.0f} frame_gap_s={frame_gap:.2f} pipeline_ms={(time.monotonic()-tick)*1000:.0f}", flush=True)
                if observation.hand_gesture_prediction:
                    gesture = observation.hand_gesture_prediction
                    print(f"hand: {gesture.label} score={gesture.confidence} detections={gesture.detection_count}", flush=True)
            if args.preview or args.dashboard_preview:
                annotated = result.plot(img=frame.copy())
                if hand_result is not None:
                    annotated = hand_result.plot(img=annotated)
                annotated = draw_status(annotated, observation,
                                        posture_threshold=args.posture_retry_threshold, help_label=args.help_gesture)
                if patient_client:
                    cv2.putText(annotated, patient_caption, (12, annotated.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
                if args.dashboard_preview and time.monotonic()-last_preview >= .3:
                    last_preview = time.monotonic()
                    preview_sender.submit(annotated)
                if args.preview:
                    cv2.imshow("Hearth local model preview - Q to quit", annotated)
                    if cv2.waitKey(1) & 0xff == ord("q"):
                        break
            if args.image or (args.max_frames is not None and processed >= args.max_frames):
                break
            if not (recorder is not None and args.dry_run and args.video):  # offline: no real-time pacing
                time.sleep(max(0, 1/args.max_fps - (time.monotonic()-tick)))
    except KeyboardInterrupt:
        pass
    finally:
        if preview_sender:
            preview_sender.close()
        if camera_reader:
            camera_reader.close()
        if args.dashboard_preview:
            from urllib.parse import quote
            from urllib.request import build_opener, ProxyHandler
            try:
                request = Request(args.backend.rstrip('/')+f"/camera-preview/{quote(args.room, safe='')}/{quote(args.source_id, safe='')}", method='DELETE')
                with build_opener(ProxyHandler({})).open(request, timeout=1) as reply:
                    reply.read()
            except OSError:
                pass
        if patient_client:
            patient_client.close()
        if recorder is not None:
            recorder.close()
            print(f"Recorded {processed} observations to {recorder.path}", file=sys.stderr)
        stop.set()
        if thread:
            thread.join(timeout=3)
        if cap is not None and camera_reader is None:
            cap.release()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
