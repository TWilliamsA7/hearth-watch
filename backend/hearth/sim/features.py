"""Per-frame posture features from normalized keypoints.

Simulated and live features must be computed the same way, or simulation results say
nothing about the live system. This module is deliberately standard-library only and
Python 3.6 compatible (no annotations syntax beyond comments, no f-string '=', no
dataclasses) so the edge client can use an identical copy.

Keypoints are plain dicts: {"name", "x", "y", "confidence"}, normalized to the frame.
Angles and aspect ratios are computed in pixel proportions, so the frame's width/height
ratio must be supplied; normalized x and y are not the same length.
"""

import math

MIN_CONFIDENCE = 0.3
HEAD_POINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")


def _usable(keypoints, min_confidence):
    return dict((k["name"], k) for k in keypoints if k["confidence"] >= min_confidence)


def _mean_point(pts, names):
    found = [pts[n] for n in names if n in pts]
    if not found:
        return None
    return (sum(p["x"] for p in found) / len(found), sum(p["y"] for p in found) / len(found))


def compute_features(keypoints, frame_aspect, prev_hip_y=None, dt=None, min_confidence=MIN_CONFIDENCE):
    """Return a PostureFeatures-shaped dict. Fields that can't be measured are omitted.

    prev_hip_y / dt: hip_y from the previous frame and seconds since it; both are needed
    for vertical_velocity.
    """
    pts = _usable(keypoints, min_confidence)
    out = {}

    neck = _mean_point(pts, ("neck",)) or _mean_point(pts, ("left_shoulder", "right_shoulder"))
    hip = _mean_point(pts, ("left_hip", "right_hip"))
    head = _mean_point(pts, ("nose",)) or _mean_point(pts, HEAD_POINTS)

    if neck is not None and hip is not None:
        dx = (hip[0] - neck[0]) * frame_aspect
        dy = hip[1] - neck[1]  # positive when the hips are below the neck
        if dx != 0.0 or dy != 0.0:
            out["torso_angle_deg"] = round(math.degrees(math.atan2(abs(dx), dy)), 2)
    if head is not None:
        out["head_y"] = round(head[1], 4)
    if hip is not None:
        out["hip_y"] = round(hip[1], 4)
        if prev_hip_y is not None and dt:
            out["vertical_velocity"] = round((hip[1] - prev_hip_y) / dt, 4)

    if len(pts) >= 3:
        xs = [p["x"] for p in pts.values()]
        ys = [p["y"] for p in pts.values()]
        height = max(ys) - min(ys)
        if height > 0.0:
            out["bbox_aspect"] = round((max(xs) - min(xs)) * frame_aspect / height, 4)

    return out
