"""Sensor-only hand gesture predictions; help decisions belong to the monitor."""
import json
import math
from pathlib import Path

from hearth_schema import BoundingBox, HandGesturePrediction
from .fall_model import ModelSettings

LABELS = {"like", "palm", "fist", "peace", "ok", "one", "three", "four",
          "call", "point", "grip", "middle_finger"}


def settings():
    return ModelSettings(**json.loads(Path(__file__).with_name("hand_model.json").read_text()))


def prediction_from_result(result, config):
    height, width = result.orig_shape
    candidates = []
    if result.boxes is not None:
        for xyxy, cls, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(),
                                   result.boxes.conf.cpu().tolist()):
            if not all(math.isfinite(v) for v in [score, *xyxy]) or not 0 <= score <= 1:
                raise ValueError("Invalid hand detection")
            if score < config.detection_confidence_min:
                continue
            x1, y1, x2, y2 = [max(0., min(1., v / dimension)) for v, dimension in zip(xyxy, [width,height,width,height])]
            if x2 <= x1 or y2 <= y1:
                continue
            candidates.append((result.names[int(cls)].strip().lower(), score,
                               BoundingBox(x1=x1,y1=y1,x2=x2,y2=y2)))
    base = dict(model_id=config.repo_id, model_revision=config.revision, detection_count=len(candidates))
    if len(candidates) != 1:
        return HandGesturePrediction(**base, label="unknown", uncertainty_reason="multiple_hands" if candidates else "no_detection")
    label, score, box = candidates[0]
    return HandGesturePrediction(**base, label=label if label in LABELS else "unknown", confidence=score,
                                 bbox=box, uncertainty_reason=None if label in LABELS else "unsupported_label")
