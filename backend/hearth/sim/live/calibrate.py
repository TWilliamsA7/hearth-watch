"""Work out the real camera's height and tilt from a person standing in view.

The live avatar is placed by casting rays from the simulated camera, so that camera must
match the real one. With the field of view known (the Jetson's C270: about 43 degrees
vertical, 4:3), a standing person of known height pins down the rest: for the right camera
height and tilt, the ray through their feet meets the floor at a spot where a head at their
height appears exactly where the camera saw it. A grid search finds the height and tilt that
make that true across a few seconds of frames.

The "jetson" preset looks straight into the room from the middle of the front wall
(hearth.sim.live.camera.placement), so its view direction depends only on the tilt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hearth.sim.live.lift import ANKLE_Y, HEAD

C270_FOV = 43.0  # degrees, vertical (55 degrees diagonal at 4:3)
HEAD_FRACTION = 0.93  # nose / eyes height as a fraction of standing height
MIN_FRAMES = 15
MAX_RMS = 0.025  # of the frame height (about 12 px at 480): a worse fit isn't trusted
CAMERA_Z = -0.12  # the jetson preset's lens, just in front of the front wall


@dataclass(frozen=True)
class CalibrationFrame:
    feet: tuple[float, float]  # (u, v): mid-ankle in the image
    head: tuple[float, float]  # (u, v): nose, else mean of eyes / ears


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
    """A usable frame has both ankles and a head point."""
    pts = {n: (u, v) for n, u, v in keypoints}
    if "left_ankle" not in pts or "right_ankle" not in pts:
        return None
    head = [pts[n] for n in (("nose",) if "nose" in pts else HEAD) if n in pts]
    if not head:
        return None
    feet = ((pts["left_ankle"][0] + pts["right_ankle"][0]) / 2, (pts["left_ankle"][1] + pts["right_ankle"][1]) / 2)
    return CalibrationFrame(feet, (sum(p[0] for p in head) / len(head), sum(p[1] for p in head) / len(head)))


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


def _error(cam: _Pinhole, frames: list[CalibrationFrame], head_y: float) -> float:
    """Mean squared head-position error (frame-height units), or inf if the geometry fails."""
    total = 0.0
    for fr in frames:
        d = cam.ray(*fr.feet)
        if d[1] >= -1e-6:
            return math.inf  # the feet ray never reaches the floor: this camera can't be right
        s = (ANKLE_Y - cam.o[1]) / d[1]
        foot = (cam.o[0] + d[0] * s, ANKLE_Y, cam.o[2] + d[2] * s)
        q = cam.project((foot[0], head_y, foot[2]))
        if q is None:
            return math.inf
        total += (q[1] - fr.head[1]) ** 2 + 0.25 * (q[0] - fr.head[0]) ** 2
    return total / len(frames)


def solve(frames: list[CalibrationFrame], person_height: float, *, fov: float = C270_FOV,
          aspect: float = 4 / 3) -> Calibration:  # fmt: skip
    """Best camera height (m) and tilt (degrees down) for these frames of a standing person."""
    if len(frames) < MIN_FRAMES:
        return Calibration(False, frames=len(frames), reason=(
            f"only {len(frames)} usable frames (need {MIN_FRAMES}): stand still with your whole body, "
            "head to feet, in view"))  # fmt: skip
    head_y = HEAD_FRACTION * person_height

    def search(heights, tilts):
        best = (math.inf, None, None)
        for h in heights:
            for t in tilts:
                e = _error(_Pinhole(h, t, fov, aspect), frames, head_y)
                if e < best[0]:
                    best = (e, h, t)
        return best

    def steps(lo, hi, step):
        n = int(round((hi - lo) / step))
        return [lo + i * step for i in range(n + 1)]

    e, h, t = search(steps(1.0, 3.0, 0.05), steps(0.0, 45.0, 1.0))
    if h is None or t is None:
        return Calibration(False, frames=len(frames), reason="no camera height and tilt fits these frames")
    e, h, t = search(steps(max(1.0, h - 0.06), min(3.0, h + 0.06), 0.01), steps(max(0.0, t - 1.2), t + 1.2, 0.1))
    rms = math.sqrt(e)
    if rms > MAX_RMS:
        return Calibration(False, round(h, 2), round(t, 1), fov, len(frames), round(rms, 4),
                           "the fit is poor: stand still and upright, whole body in view, and check your height")  # fmt: skip
    return Calibration(True, round(h, 2), round(t, 1), fov, len(frames), round(rms, 4))
