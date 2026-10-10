"""Builds Observation and Heartbeat messages as plain dicts matching shared/ schema v1.

The Jetson cannot run Pydantic, so this mirrors shared/python/hearth_schema/models.py by hand.
edge/tests/test_contract.py validates these dicts against the real models on the laptop;
if the contract changes, that test fails until this file is updated.
"""

import time
from datetime import datetime, timezone

SCHEMA_VERSION = "1"

# Same names and order as hearth_schema.KeypointName (poseNet resnet18-body / COCO-18).
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "neck",
)
_KEYPOINT_ORDER = dict((n, i) for i, n in enumerate(KEYPOINT_NAMES))


def utc_iso(ts):
    """Unix time -> ISO 8601 UTC with milliseconds, e.g. 2026-10-10T05:12:03.481Z."""
    dt = datetime.fromtimestamp(ts, timezone.utc)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def parse_utc_iso(text):
    """ISO 8601 UTC ('2026-10-10T05:12:03.481Z') -> Unix time; fractional seconds optional."""
    s = text.replace("+00:00", "").rstrip("Z")
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in s else "%Y-%m-%dT%H:%M:%S"
    dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
    return (dt - _EPOCH).total_seconds()


def _unit(v):
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


class ObservationIds(object):
    """Unique, ordered ids: <source>-<start epoch>-<sequence>, unique across restarts."""

    def __init__(self, source, start=None):
        self.prefix = "%s-%d" % (source, int(start if start is not None else time.time()))
        self.seq = 0

    def next(self):
        self.seq += 1
        return "%s-%06d" % (self.prefix, self.seq)


def normalize_keypoints(kps_px, width, height, confidence):
    """{name: (x_px, y_px)} -> contract Keypoint dicts in 0..1 frame coordinates.

    Unknown names are dropped. confidence=None omits the field (for when the contract makes
    it optional); otherwise the given value is sent for every keypoint.
    """
    out = []
    for name in sorted((n for n in kps_px if n in _KEYPOINT_ORDER), key=_KEYPOINT_ORDER.get):
        x, y = kps_px[name]
        kp = {
            "name": name,
            "x": round(_unit(x / float(width)), 4),
            "y": round(_unit(y / float(height)), 4),
        }
        if confidence is not None:
            kp["confidence"] = confidence
        out.append(kp)
    return out


def build_observation(obs_id, ts, source, room_id, person_detected, keypoints=None,
                      features=None, face_present=None, face_confidence=None,
                      source_type="edge", scenario_id=None):
    """One Observation. face_present=None means face detection did not run on this frame."""
    return {
        "schema_version": SCHEMA_VERSION,
        "obs_id": obs_id,
        "timestamp": utc_iso(ts),
        "source": source,
        "source_type": source_type,
        "room_id": room_id,
        "scenario_id": scenario_id,
        "person_detected": bool(person_detected),
        "keypoints": keypoints if person_detected and keypoints else None,
        "features": features if person_detected and features else None,
        "face_present": face_present,
        "face_confidence": None if face_confidence is None else round(_unit(face_confidence), 4),
    }


def build_heartbeat(ts, source, room_id, fps=None, inference_ms=None, mem_used_mb=None,
                    source_type="edge"):
    def r(v):
        return None if v is None else round(max(0.0, v), 2)

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": utc_iso(ts),
        "source": source,
        "source_type": source_type,
        "room_id": room_id,
        "fps": r(fps),
        "inference_ms": r(inference_ms),
        "mem_used_mb": r(mem_used_mb),
    }
