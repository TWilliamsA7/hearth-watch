"""Work out the real camera's height and tilt from a person standing in view.

The live avatar is placed by casting rays from the simulated camera, so that camera must
match the real one. With the field of view known (the Jetson's C270: about 43 degrees
vertical, 4:3), a standing person of known height pins down the rest: for the right camera
height and tilt, the ray through their feet meets the floor at a spot where a head at their
height appears exactly where the camera saw it. A grid search finds the height and tilt that
make that true across a few seconds of frames. Any two body points at known heights work the
same way (the camera often cuts off the feet): knees or hips below, head or neck above.

The "jetson" preset looks straight into the room from the middle of the front wall
(hearth.sim.live.camera.placement), so its view direction depends only on the tilt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


C270_FOV = 43.0  # degrees, vertical (55 degrees diagonal at 4:3)
MIN_FRAMES = 10  # each frame is a separate measurement; the fit drops the worst 20%
# From one spot the camera's height and tilt trade off against each other (the answer can be
# 0.6 m out); frames from around the room pin them down. Required spread of the low mark, in
# image units, across and up/down the frame.
MIN_SPREAD_U, MIN_SPREAD_V = 0.15, 0.05
MAX_RMS = 0.025  # of the frame height (about 12 px at 480): a worse fit isn't trusted
CAMERA_Z = -0.12  # the jetson preset's lens, just in front of the front wall
MIN_TILT, MAX_TILT = -15.0, 60.0  # degrees down; a camera at head height may look level or up


# Body points usable as calibration marks, lowest first / highest first, with their height as a
# fraction of standing height (from the standing template the avatar uses, hearth.sim.pose).
LOW_MARKS = (("ankle", ("left_ankle", "right_ankle")), ("knee", ("left_knee", "right_knee")),
             ("hip", ("left_hip", "right_hip")))  # fmt: skip
HIGH_MARKS = (("eyes", ("left_eye", "right_eye")), ("nose", ("nose",)), ("ears", ("left_ear", "right_ear")),
              ("neck", ("neck",)), ("shoulders", ("left_shoulder", "right_shoulder")))  # fmt: skip
MIN_GAP = 0.25  # of standing height, between the two marks: closer marks constrain too little


def _mark_height(names: tuple[str, ...]) -> float:
    from hearth.sim.pose import KEY_POSES

    return sum(KEY_POSES["standing"][n][1] for n in names) / len(names)


@dataclass(frozen=True)
class CalibrationFrame:
    low: tuple[float, float]  # (u, v) of the lowest mark seen (ankles, else knees, else hips)
    low_frac: float  # its height, as a fraction of standing height
    high: tuple[float, float]  # (u, v) of the highest mark seen (eyes, nose, ears, neck, shoulders)
    high_frac: float


@dataclass(frozen=True)
class Calibration:
    ok: bool
    height: float | None = None
    tilt: float | None = None
    fov: float = C270_FOV
    frames: int = 0
    rms: float | None = None  # fit error, fraction of frame height
    reason: str = ""


def frame_from(keypoints: list[tuple[str, float, float]]) -> CalibrationFrame | None:
    """Two marks at known heights on a standing person, as far apart as the frame allows. The
    Jetson's camera often cuts off the feet: knees or hips with the head work just as well."""
    pts = {n: (u, v) for n, u, v in keypoints}

    def first(marks):
        # Either side of a pair will do: poseNet often sees one hip or one knee, not both,
        # and both sit at the same height.
        for _, names in marks:
            seen = [n for n in names if n in pts]
            if seen:
                uv = (sum(pts[n][0] for n in seen) / len(seen), sum(pts[n][1] for n in seen) / len(seen))
                return uv, _mark_height(tuple(seen))
        return None

    low, high = first(LOW_MARKS), first(HIGH_MARKS)
    if low is None or high is None or high[1] - low[1] < MIN_GAP:
        return None
    return CalibrationFrame(low[0], low[1], high[0], high[1])


class _Pinhole:
    """The jetson preset's camera at a given height and tilt (same maths as live.camera.Camera)."""

    def __init__(self, height: float, tilt_deg: float, fov_deg: float, aspect: float) -> None:
        t = math.radians(tilt_deg)
        self.o = (0.0, height, CAMERA_Z)
        self.f = (0.0, -math.sin(t), -math.cos(t))
        self.r = (1.0, 0.0, 0.0)  # forward x world-up, for a camera looking along -z
        self.u = (0.0, math.cos(t), -math.sin(t))  # right x forward
        self.tv = math.tan(math.radians(fov_deg) / 2)
        self.th = self.tv * aspect

    def ray(self, u: float, v: float) -> tuple[float, float, float]:
        a, b = (2 * u - 1) * self.th, (1 - 2 * v) * self.tv
        return tuple(self.f[i] + a * self.r[i] + b * self.u[i] for i in range(3))  # type: ignore[return-value]

    def project(self, p: tuple[float, float, float]) -> tuple[float, float] | None:
        d = (p[0] - self.o[0], p[1] - self.o[1], p[2] - self.o[2])
        z = sum(d[i] * self.f[i] for i in range(3))
        if z < 0.05:
            return None
        x = sum(d[i] * self.r[i] for i in range(3))
        y = sum(d[i] * self.u[i] for i in range(3))
        return (0.5 + x / z / (2 * self.th), 0.5 - y / z / (2 * self.tv))


def _error(cam: _Pinhole, frames: list[CalibrationFrame], person_height: float) -> float:
    """Mean squared error (frame-height units) of where the high mark should appear, given the
    low mark's ray, over the best 80% of frames (a few bad detections don't spoil the fit)."""
    errs = []
    for fr in frames:
        d = cam.ray(*fr.low)
        low_y = fr.low_frac * person_height
        if abs(d[1]) < 1e-6:
            return math.inf
        s = (low_y - cam.o[1]) / d[1]
        if s <= 0.05:
            return math.inf  # the low mark's ray never reaches its height: this camera can't be right
        spot = (cam.o[0] + d[0] * s, cam.o[2] + d[2] * s)
        q = cam.project((spot[0], fr.high_frac * person_height, spot[1]))
        if q is None:
            return math.inf
        errs.append((q[1] - fr.high[1]) ** 2 + 0.25 * (q[0] - fr.high[0]) ** 2)
    errs.sort()
    kept = errs[: max(1, int(len(errs) * 0.8))]
    return sum(kept) / len(kept)


def solve(frames: list[CalibrationFrame], person_height: float, *, fov: float = C270_FOV,
          aspect: float = 4 / 3) -> Calibration:  # fmt: skip
    """Best camera height (m) and tilt (degrees down) for these frames of a standing person."""
    if len(frames) < MIN_FRAMES:
        return Calibration(False, frames=len(frames), reason=(
            f"only {len(frames)} usable frames (need {MIN_FRAMES}): stay upright with your head and at least "
            "your knees or hips in view"))  # fmt: skip
    us, vs = [f.low[0] for f in frames], [f.low[1] for f in frames]
    if max(us) - min(us) < MIN_SPREAD_U or max(vs) - min(vs) < MIN_SPREAD_V:
        return Calibration(False, frames=len(frames), reason=(
            "you stayed in one place: walk slowly around while it watches (left, right, nearer, further), "
            "staying upright and in view"))  # fmt: skip

    def search(heights, tilts):
        best = (math.inf, None, None)
        for h in heights:
            for t in tilts:
                e = _error(_Pinhole(h, t, fov, aspect), frames, person_height)
                if e < best[0]:
                    best = (e, h, t)
        return best

    def steps(lo, hi, step):
        n = int(round((hi - lo) / step))
        return [lo + i * step for i in range(n + 1)]

    e, h, t = search(steps(1.0, 3.0, 0.05), steps(MIN_TILT, MAX_TILT, 1.0))  # tilt < 0: looking up
    if h is None or t is None:
        return Calibration(False, frames=len(frames), reason="no camera height and tilt fits these frames")
    e, h, t = search(steps(max(1.0, h - 0.06), min(3.0, h + 0.06), 0.01), steps(max(MIN_TILT, t - 1.2), min(MAX_TILT, t + 1.2), 0.1))
    rms = math.sqrt(e)
    if rms > MAX_RMS:
        return Calibration(False, round(h, 2), round(t, 1), fov, len(frames), round(rms, 4),
                           "the fit is poor: stand still and upright, facing the camera, and check the height you entered")  # fmt: skip
    return Calibration(True, round(h, 2), round(t, 1), fov, len(frames), round(rms, 4))
