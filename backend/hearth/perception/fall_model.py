"""Pinned local YOLO weights and a sensor-only adapter to Hearth observations."""
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

from hearth_schema import BoundingBox, Observation, PosturePrediction


@dataclass(frozen=True)
class ModelSettings:
    repo_id: str
    revision: str
    filename: str
    sha256: str
    image_size: int
    detection_confidence_min: float
    device: str
    max_fps: float
    torch_threads: int

    @classmethod
    def load(cls):
        return cls(**json.loads(Path(__file__).with_name("model.json").read_text(encoding="utf-8")))


def verify_weights(path: Path, settings: ModelSettings) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Weights missing: {path}. Run edge/run_fall_model.py --download-only first.")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != settings.sha256:
        raise ValueError("Weights do not match the pinned Hugging Face model checksum")


def download_weights(path: Path, settings: ModelSettings) -> Path:
    if path.exists():
        verify_weights(path, settings)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".part")
    url = f"https://huggingface.co/{settings.repo_id}/resolve/{settings.revision}/{settings.filename}"
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        verify_weights(temporary, settings)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def canonical_label(name: str) -> str:
    name = name.strip().lower().replace("_", " ").replace("-", " ")
    return {"standing": "standing", "sitting": "sitting", "fallen": "fallen",
            "fall": "fallen", "falling": "fallen", "fall detected": "fallen"}.get(name, "unknown")


def prediction_from_detections(detections: list[dict], width: int, height: int,
                              settings: ModelSettings) -> PosturePrediction:
    """No identity tracking: multiple boxes are uncertainty, not a selected 'safe' person."""
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    candidates = []
    for detection in detections:
        score = float(detection["confidence"])
        coordinates = [float(value) for value in detection["xyxy"]]
        if (len(coordinates) != 4 or not all(math.isfinite(v) for v in [score, *coordinates])
                or not 0 <= score <= 1):
            raise ValueError("Invalid model detection")
        if score < settings.detection_confidence_min:
            continue
        x1, y1, x2, y2 = coordinates
        box = [max(0., min(1., x1 / width)), max(0., min(1., y1 / height)),
               max(0., min(1., x2 / width)), max(0., min(1., y2 / height))]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        candidates.append((canonical_label(detection["label"]), score, box))
    count = len(candidates)
    base = dict(model_id=settings.repo_id, model_revision=settings.revision, detection_count=count)
    if count != 1:
        return PosturePrediction(**base, label="unknown",
                                 uncertainty_reason="multiple_people" if count else "no_detection")
    label, score, box = candidates[0]
    return PosturePrediction(**base, label=label, confidence=score,
                             bbox=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
                             uncertainty_reason="unsupported_label" if label == "unknown" else None)


def observation_from_result(result, *, timestamp: datetime, source: str, source_type: str,
                            room_id: str, run_id: str, settings: ModelSettings) -> Observation:
    detections = []
    if result.boxes is not None:
        for xyxy, cls, conf in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(),
                                   result.boxes.conf.cpu().tolist()):
            detections.append(dict(label=result.names[int(cls)], confidence=conf, xyxy=xyxy))
    height, width = result.orig_shape
    prediction = prediction_from_detections(detections, width, height, settings)
    return Observation(obs_id=str(uuid4()), timestamp=timestamp, source=source, source_type=source_type,
                       room_id=room_id, run_id=run_id, person_detected=prediction.detection_count > 0,
                       posture_prediction=prediction, face_present=None)


def load_model(path: Path, settings: ModelSettings, runtime_dir: Path, *, expected_labels=None):
    verify_weights(path, settings)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(runtime_dir.resolve())
    os.environ["MPLCONFIGDIR"] = str((runtime_dir / "matplotlib").resolve())
    os.environ["TORCH_HOME"] = str((runtime_dir / "torch").resolve())
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    try:
        import torch
        from ultralytics import YOLO, settings as yolo_settings
    except ImportError as exc:
        raise RuntimeError("Install local vision dependencies: python -m pip install -e './backend[vision]'") from exc
    yolo_settings.update({"sync": False})
    torch.set_num_threads(settings.torch_threads)
    model = YOLO(str(path), task="detect")
    labels = ({name.strip().lower() for name in model.names.values()} if expected_labels
              else {canonical_label(name) for name in model.names.values()})
    if labels != (expected_labels or {"standing", "sitting", "fallen"}):
        raise ValueError(f"Unexpected checkpoint labels: {model.names}")
    return model
