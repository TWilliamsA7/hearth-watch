"""Person 2's posture_prediction rules, for the Jetson's Python 3.6. Pure functions, no numpy.

A faithful port of backend/hearth/perception/fall_model.py (canonical_label and
prediction_from_detections). The backend can't be imported on the Jetson (Python 3.11, Pydantic),
so this mirrors it and returns a plain dict matching the contract's PosturePrediction.
edge/tests/test_fall_rules.py runs the same detections through both and checks they agree.

The rules (Person 2's): a detection counts if its score is >= detection_confidence_min (0.25)
and its box has positive size; exactly one such detection gives its label and score; zero gives
label "unknown" (no_detection); several give "unknown" (multiple_people) - the model is never
used to pick one person as "safe". A class name outside standing/sitting/fallen is "unknown"
(unsupported_label).

model_id / model_revision are the BASE model's (melihuzunoglu/human-fall-detection @ 97261b0):
the Jetson runs the same weights in ONNX/TensorRT form, and the backend only trusts predictions
whose model_id matches its configured fall_model_id.
"""

import math

MODEL_ID = "melihuzunoglu/human-fall-detection"
MODEL_REVISION = "97261b03632a9715a27481b2cfdac6a78e26d2d9"
DETECTION_CONFIDENCE_MIN = 0.25   # backend/hearth/perception/model.json


def canonical_label(name):
    name = name.strip().lower().replace("_", " ").replace("-", " ")
    return {"standing": "standing", "sitting": "sitting", "fallen": "fallen",
            "fall": "fallen", "falling": "fallen", "fall detected": "fallen"}.get(name, "unknown")


def _unit(v):
    return max(0.0, min(1.0, v))


def prediction_from_detections(detections, width, height, min_confidence=DETECTION_CONFIDENCE_MIN,
                               model_id=MODEL_ID, model_revision=MODEL_REVISION):
    """detections: [{"label": str, "confidence": float, "xyxy": [x1, y1, x2, y2] (pixels)}].
    Returns the contract's PosturePrediction as a dict."""
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    candidates = []
    for d in detections:
        score = float(d["confidence"])
        coords = [float(v) for v in d["xyxy"]]
        if (len(coords) != 4 or not all(math.isfinite(v) for v in [score] + coords)
                or not 0 <= score <= 1):
            raise ValueError("Invalid model detection")
        if score < min_confidence:
            continue
        x1, y1, x2, y2 = coords
        box = [_unit(x1 / width), _unit(y1 / height), _unit(x2 / width), _unit(y2 / height)]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        candidates.append((canonical_label(d["label"]), score, box))
    # input_processing "original": the Jetson uses the frame as captured (no contrast retry,
    # which stays a feature of Person 2's laptop runner).
    base = {"model_id": model_id, "model_revision": model_revision,
            "input_processing": "original", "detection_count": len(candidates)}
    if len(candidates) != 1:
        base.update(label="unknown", confidence=None, bbox=None,
                    uncertainty_reason="multiple_people" if candidates else "no_detection")
        return base
    label, score, box = candidates[0]
    base.update(label=label, confidence=score,
                bbox={"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
                uncertainty_reason="unsupported_label" if label == "unknown" else None)
    return base
