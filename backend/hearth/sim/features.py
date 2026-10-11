"""Per-frame posture features from normalized keypoints, computed the way the edge does.

Simulated and live features must be computed the same way, or simulation results say
nothing about the live system. The formulas, smoothing and dropout handling here follow
edge/hearth_edge/features.py and the per-frame pipeline in edge/run_edge.py;
backend/tests/test_feature_parity.py runs both on the same poses and requires the
same numbers. Standard-library only and Python 3.6 compatible, like the edge module.

Keypoints are plain dicts: {"name", "x", "y", "confidence"}, normalized to the frame.
confidence may be None (poseNet reports none): such a keypoint counts as detected.
Angles and aspect ratios are computed in pixel proportions, so the frame's width/height
ratio must be supplied; normalized x and y are not the same length.

The one deliberate difference: the edge's bbox_aspect comes from poseNet's box, which
wraps the keypoints it found. The simulator has no detector box, so it uses the extent
of the detected keypoints, gated the same way (only when part of the torso is visible).
"""

import math

# Below this a simulated keypoint counts as "not detected" (poseNet's own threshold
# decides that on the edge; its keypoints carry no confidence at all).
MIN_CONFIDENCE = 0.3
HEAD_POINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")


def _clamp01(v):
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _detected(keypoints, min_confidence):
    """name -> (x, y) for keypoints the camera would report."""
    out = {}
    for k in keypoints:
        if k.get("confidence") is None or k["confidence"] >= min_confidence:
            out[k["name"]] = (k["x"], k["y"])
    return out


def _mean_point(pts, names):
    found = [pts[n] for n in names if n in pts]
    if not found:
        return None
    return (sum(p[0] for p in found) / len(found), sum(p[1] for p in found) / len(found))


def _neck(pts):
    """The neck keypoint, else the shoulder midpoint (needs both shoulders)."""
    if "neck" in pts:
        return pts["neck"]
    if "left_shoulder" in pts and "right_shoulder" in pts:
        return _mean_point(pts, ("left_shoulder", "right_shoulder"))
    return None


def _torso_visible(pts):
    upper = "neck" in pts or "left_shoulder" in pts or "right_shoulder" in pts
    lower = "left_hip" in pts or "right_hip" in pts
    return upper and lower


class VerticalVelocity(object):
    """Change in smoothed hip_y per second (positive = moving down). Same as the edge's.

    Exponential moving average of hip_y; resets after a gap or a frame without hips.
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
    """Fills a joint missing for up to max_age_s from where it was last seen, moved by the
    average motion of the joints seen in both frames. Same as the edge's; it only feeds the
    features, never the observation's keypoints."""

    def __init__(self, max_age_s=0.3):
        self.max_age_s = max_age_s
        self._last = {}

    def reset(self):
        self._last = {}

    def update(self, pts, t):
        if self.max_age_s <= 0:
            return dict(pts)
        common = [n for n in pts if n in self._last]
        held = []
        merged = dict(pts)
        if common:
            dx = sum(pts[n][0] - self._last[n][0] for n in common) / float(len(common))
            dy = sum(pts[n][1] - self._last[n][1] for n in common) / float(len(common))
            for name, (x, y, seen) in list(self._last.items()):
                if name not in pts and t - seen <= self.max_age_s:
                    merged[name] = (x + dx, y + dy)
                    held.append(name)
        new_last = dict((n, (p[0], p[1], t)) for n, p in pts.items())
        for name in held:
            new_last[name] = (merged[name][0], merged[name][1], self._last[name][2])
        self._last = new_last
        return merged


def compute_features(keypoints, frame_aspect, velocity=None, t=None, min_confidence=MIN_CONFIDENCE,
                     hold=None):
    """Return a PostureFeatures-shaped dict. Fields that can't be measured are omitted.

    velocity / t: a VerticalVelocity carried from frame to frame for this person, and the
    frame time in seconds; both are needed for vertical_velocity.
    hold: an optional KeypointHold carried the same way.
    """
    detected = _detected(keypoints, min_confidence)
    pts = hold.update(detected, t) if hold is not None and t is not None else detected

    neck = _neck(pts)
    hip = _mean_point(pts, ("left_hip", "right_hip"))
    head = _mean_point(pts, HEAD_POINTS)
    out = {}

    if neck is not None and hip is not None:
        dx = (neck[0] - hip[0]) * frame_aspect
        dy = neck[1] - hip[1]  # negative when the neck is above the hips
        if dx != 0.0 or dy != 0.0:
            out["torso_angle_deg"] = math.degrees(math.atan2(abs(dx), -dy))
    if head is not None:
        out["head_y"] = _clamp01(head[1])
    hip_y = _clamp01(hip[1]) if hip is not None else None
    if hip_y is not None:
        out["hip_y"] = hip_y
    if velocity is not None and t is not None:
        v = velocity.update(hip_y, t)
        if v is not None:
            out["vertical_velocity"] = v

    if _torso_visible(pts):
        # Like poseNet's box: around the joints actually detected, not the held ones.
        xs = [p[0] for p in detected.values()]
        ys = [p[1] for p in detected.values()]
        height = max(ys) - min(ys)
        if height > 0.0:
            out["bbox_aspect"] = (max(xs) - min(xs)) * frame_aspect / height

    return dict((k, round(v, 4)) for k, v in out.items())


class FeatureTracker(object):
    """One person's feature state across frames, as the edge keeps it for its one person."""

    def __init__(self, frame_aspect, hold_s=0.3):
        self.frame_aspect = frame_aspect
        self.velocity = VerticalVelocity()
        self.hold = KeypointHold(hold_s)

    def lost(self):
        """No pose this frame: the edge resets the velocity (the hold ages out on its own)."""
        self.velocity.reset()

    def reset(self):
        self.velocity.reset()
        self.hold.reset()

    def update(self, keypoints, t):
        return compute_features(keypoints, self.frame_aspect, self.velocity, t, hold=self.hold)
