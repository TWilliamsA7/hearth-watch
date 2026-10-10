"""Camera/video/image inference. Frames stay local; only structured observations are sent."""
import argparse
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


def post(base, endpoint, body):
    request = Request(base.rstrip("/") + endpoint, data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=2) as response:
        return json.load(response)


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
    parser.add_argument("--preview", action="store_true", help="Show local annotated video; Q closes it")
    parser.add_argument("--dry-run", action="store_true", help="Print observations without contacting backend")
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_fps) or args.max_fps <= 0:
        parser.error("--max-fps must be positive and finite")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.download_only:
        path = download_weights(args.weights, config)
        print(f"Verified weights: {path.resolve()}\nRevision: {config.revision}")
        return
    model = load_model(args.weights, config, args.runtime_dir)
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
        return
    cap = None
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
    elif args.video:
        if not args.video.is_file():
            parser.error("Video file does not exist")
        cap = cv2.VideoCapture(str(args.video))
    if cap is not None and not cap.isOpened():
        cap.release()
        raise RuntimeError("Cannot open camera/video source")
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
    processed, frame_index = 0, -1
    last_warning = float("-inf")
    started = time.monotonic()
    try:
        while True:
            tick = time.monotonic()
            if args.image:
                frame = cv2.imread(str(args.image))
                if frame is None:
                    raise RuntimeError("Cannot read image")
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
                         if args.video else datetime.now(timezone.utc))
            inference_start = time.perf_counter()
            result = model.predict(frame, device=args.device, imgsz=config.image_size,
                                   conf=config.detection_confidence_min, agnostic_nms=True,
                                   save=False, verbose=False)[0]
            metrics["inference_ms"] = (time.perf_counter()-inference_start)*1000
            observation = observation_from_result(result, timestamp=timestamp, settings=config, **context)
            processed += 1
            metrics["fps"] = processed / max(time.monotonic()-started, .001)
            if args.dry_run:
                print(observation.model_dump_json(), flush=True)
            else:
                try:
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
                      f"inference_ms={metrics['inference_ms']:.0f}", flush=True)
            if args.preview:
                cv2.imshow("Hearth local model preview - Q to quit", result.plot())
                if cv2.waitKey(1) & 0xff == ord("q"):
                    break
            if args.image or (args.max_frames is not None and processed >= args.max_frames):
                break
            time.sleep(max(0, 1/args.max_fps - (time.monotonic()-tick)))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if thread:
            thread.join(timeout=3)
        if cap is not None:
            cap.release()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
