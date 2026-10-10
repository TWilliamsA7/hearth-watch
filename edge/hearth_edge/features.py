"""Per-frame posture features from one poseNet pose. Pure functions, no Jetson imports.

A pose is a dict:
    {"keypoints": {name: (x_px, y_px), ...},   # only the keypoints poseNet reported
     "bbox": (left, top, right, bottom)}        # pixels
Angles are computed in pixel space so a 4:3 frame does not distort them. Heights are
returned normalized to the frame (0 = top), matching the shared contract.
"""

import math

HEAD_POINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")


def clamp01(v):
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _mean_point(kps, names):
    pts = [kps[n] for n in names if n in kps]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def neck_point(kps):
    """poseNet's neck keypoint, else the shoulder midpoint (needs both shoulders)."""
    if "neck" in kps:
        return kps["neck"]
    if "left_shoulder" in kps and "right_shoulder" in kps:
        return _mean_point(kps, ("left_shoulder", "right_shoulder"))
    return None


def hip_point(kps):
    """Mid-hip from whichever hips were detected."""
    return _mean_point(kps, ("left_hip", "right_hip"))


def torso_angle_deg(kps):
    """Angle of the hip->neck line from vertical: 0 upright, 90 horizontal, 180 inverted."""
    neck = neck_point(kps)
    hip = hip_point(kps)
    if neck is None or hip is None:
        return None
    dx = neck[0] - hip[0]
    dy = neck[1] - hip[1]  # image y grows downward, so upright gives dy < 0
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(abs(dx), -dy))


def head_y(kps, height):
    p = _mean_point(kps, HEAD_POINTS)
    return None if p is None else clamp01(p[1] / float(height))


def hip_y(kps, height):
    p = hip_point(kps)
    return None if p is None else clamp01(p[1] / float(height))


def bbox_aspect(bbox):
    left, top, right, bottom = bbox
    w = right - left
    h = bottom - top
    if w < 0 or h <= 0:
        return None
    return w / float(h)


class VerticalVelocity(object):
    """Change in smoothed hip_y per second (positive = moving down).

    Smooths hip_y with an exponential moving average and divides by the real time between
    samples. Resets after a gap or a frame without hips, so a person reappearing elsewhere
    does not look like a sudden drop.
    """

    def __init__(self, alpha=0.5, reset_gap_s=1.0):
        self.alpha = alpha
        self.reset_gap_s = reset_gap_s
        self._ema = None
        self._t = None

    def reset(self):
        self._ema = None
        self._t = None

    def update(self, hip_y_value, t):
        if hip_y_value is None:
            self.reset()
            return None
        if self._ema is None or t - self._t > self.reset_gap_s or t <= self._t:
            self._ema = hip_y_value
            self._t = t
            return None
        prev_ema, prev_t = self._ema, self._t
        self._ema = self.alpha * hip_y_value + (1.0 - self.alpha) * prev_ema
        self._t = t
        return (self._ema - prev_ema) / (t - prev_t)


def compute_features(pose, width, height, velocity, t):
    """The contract's PostureFeatures as a dict, omitting anything not measurable."""
    kps = pose["keypoints"]
    hy = hip_y(kps, height)
    out = {
        "torso_angle_deg": torso_angle_deg(kps),
        "head_y": head_y(kps, height),
        "hip_y": hy,
        "bbox_aspect": bbox_aspect(pose["bbox"]),
        "vertical_velocity": velocity.update(hy, t),
    }
    return dict((k, round(v, 4)) for k, v in out.items() if v is not None)


def select_pose(poses):
    """The largest pose by bounding-box area (one person per room in the MVP), or None."""
    best, best_area = None, -1.0
    for p in poses:
        left, top, right, bottom = p["bbox"]
        area = max(0.0, right - left) * max(0.0, bottom - top)
        if area > best_area:
            best, best_area = p, area
    return best
