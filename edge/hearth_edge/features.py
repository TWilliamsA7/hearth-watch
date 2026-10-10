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


def torso_visible(kps):
    """At least one shoulder (or the neck) and at least one hip were detected.

    poseNet's bounding box only wraps the keypoints it found. With just the face visible the
    box is ear-to-ear wide and eye-to-chin short, so its aspect looks like someone lying down.
    Box-based features are only meaningful when part of the torso is in view.
    """
    upper = "neck" in kps or "left_shoulder" in kps or "right_shoulder" in kps
    lower = "left_hip" in kps or "right_hip" in kps
    return upper and lower


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


def head_point(kps):
    """Average of whichever of nose/eyes/ears were detected (pixels), or None."""
    return _mean_point(kps, HEAD_POINTS)


def head_y(kps, height):
    p = head_point(kps)
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


class KeypointHold(object):
    """Bridges brief keypoint dropouts so one missed frame doesn't change the posture numbers.

    poseNet sometimes loses a joint (often a hip) for a frame or two. The backend restarts its
    floor-level timer on any frame that fails its rule, so such flicker can block an alert.
    For up to `max_age_s`, a missing joint is filled in from where it was last seen, shifted by
    how far the joints visible in both frames moved on average (so it follows the body, e.g.
    during a fall). Held joints only feed the posture features; the Observation's keypoints stay
    exactly what poseNet reported. With no joints in common there is no motion estimate, so
    nothing is held. max_age_s = 0 disables it.
    """

    def __init__(self, max_age_s=0.3):
        self.max_age_s = max_age_s
        self._last = {}  # name -> (x, y, t) where it was last seen or held

    def reset(self):
        self._last = {}

    def update(self, kps, t):
        """kps: this frame's detected keypoints. Returns (merged keypoints, names that were held)."""
        if self.max_age_s <= 0:
            return dict(kps), []
        common = [n for n in kps if n in self._last]
        held = []
        merged = dict(kps)
        if common:
            dx = sum(kps[n][0] - self._last[n][0] for n in common) / float(len(common))
            dy = sum(kps[n][1] - self._last[n][1] for n in common) / float(len(common))
            for name, (x, y, seen) in list(self._last.items()):
                if name not in kps and t - seen <= self.max_age_s:
                    merged[name] = (x + dx, y + dy)
                    held.append(name)
        # Remember detected joints at their new position; held ones keep their original time,
        # so a joint can never be held longer than max_age_s after it was really seen.
        new_last = dict((n, (p[0], p[1], t)) for n, p in kps.items())
        for name in held:
            new_last[name] = (merged[name][0], merged[name][1], self._last[name][2])
        self._last = new_last
        return merged, sorted(held)


def compute_features(pose, width, height, velocity, t):
    """The contract's PostureFeatures as a dict, omitting anything not measurable."""
    kps = pose["keypoints"]
    hy = hip_y(kps, height)
    out = {
        "torso_angle_deg": torso_angle_deg(kps),
        "head_y": head_y(kps, height),
        "hip_y": hy,
        "bbox_aspect": bbox_aspect(pose["bbox"]) if torso_visible(kps) else None,
        "vertical_velocity": velocity.update(hy, t),
    }
    return dict((k, round(v, 4)) for k, v in out.items() if v is not None)


def bbox_center(bbox):
    left, top, right, bottom = bbox
    return ((left + right) / 2.0, (top + bottom) / 2.0)


def select_pose(poses, near=None, max_dist=None):
    """The pose to follow this frame (one person per room in the MVP), or None.

    Without `near`: the largest pose by bounding-box area. With `near` (where the tracked person
    was a moment ago, pixels): the pose whose box centre is closest to it, if within `max_dist`;
    otherwise None, so a person in the background can't take over for a frame (the caller then
    tries harder, e.g. a rotated pass, to find the tracked person).
    """
    if near is not None:
        best, best_d = None, None
        for p in poses:
            cx, cy = bbox_center(p["bbox"])
            d = math.hypot(cx - near[0], cy - near[1])
            if (max_dist is None or d <= max_dist) and (best_d is None or d < best_d):
                best, best_d = p, d
        return best
    best, best_area = None, -1.0
    for p in poses:
        left, top, right, bottom = p["bbox"]
        area = max(0.0, right - left) * max(0.0, bottom - top)
        if area > best_area:
            best, best_area = p, area
    return best


class PersonTrack(object):
    """Where the followed person was, so the next frame picks the same person.

    `near(t)` gives the last position while it is recent (`keep_s`), else None (then the
    largest pose is picked, e.g. when someone new walks in). `brief_dropout(t)` is True when the
    person went missing only moments ago (`dropout_s`): the edge then skips sending that frame
    rather than reporting "no person", because the backend resets its floor timer on an explicit
    "no person" but tolerates short gaps between observations.
    """

    def __init__(self, keep_s=1.0, dropout_s=0.3):
        self.keep_s = keep_s
        self.dropout_s = dropout_s
        self.center = None
        self.t = None

    def update(self, pose, t):
        if pose is not None:
            self.center = bbox_center(pose["bbox"])
            self.t = t

    def near(self, t):
        if self.t is None or t - self.t > self.keep_s:
            return None
        return self.center

    def brief_dropout(self, t):
        return self.t is not None and 0 < t - self.t <= self.dropout_s
