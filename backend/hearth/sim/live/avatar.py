"""The live person's avatar: a rigid body driven by a few smoothed measurements.

A single camera's keypoints are noisy, flicker in and out and carry no depth, so they don't
drive the body directly. Each frame gives at most:

  * a floor position (one depth anchor, outliers checked over time, One-Euro filtered,
    then limited to a controlled walking speed, with a dead zone for small jitter);
  * a heading (from the shoulders when upright, from hips to head on the floor);
  * the posture the real monitor concluded (debounced, then blended over ~0.6 s);
  * the direction of each arm, when its shoulder and elbow are seen.

The body itself is a template pose (hearth.sim.pose.KEY_POSES) with the arms set from those
directions and the template's own bone lengths, placed with hearth.sim.world.body like every
simulated resident: the avatar always has a whole, well-formed skeleton.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from hearth.sim.live.camera import Camera
from hearth.sim.live.lift import (
    ANKLE_Y,
    FALLBACK_DISTANCE_M,
    HEAD,
    _at_distance,
    _mean,
    _on_plane,
)
from hearth.sim.live.room import Room
from hearth.sim.pose import KEY_POSES, BodyPose, blend_body, smoothstep
from hearth.sim.world import body as place_body

Vec3 = tuple[float, float, float]

TEMPLATE = {"standing": "standing", "sitting": "sitting", "lying": "side_lying", "floor": "side_lying"}
HIP_HEIGHT = {"standing": 0.53, "sitting": 0.30}  # of body height, for the hip anchor
FLOOR_BODY_Y = 0.15  # metres: mid-height of a body lying on the floor
SWITCH_ANCHOR_S = 0.5  # another depth anchor must be available this long before it's used
ANCHOR_GAP_S = 0.3  # an anchor missing for less than this (a dropped joint) still counts as available
JUMP_M = 0.8  # a position this far from the filtered one is an outlier...
JUMP_HOLD_S = 0.75  # a distant target must stay consistent this long before following it
JUMP_TARGET_M = 0.35  # wildly different outliers cannot confirm each other
POSITION_DEADBAND_M = 0.06  # ignore small floor-position jitter
MAX_SPEED_M_S = 0.6  # controlled walking pace, including after posture/anchor changes
MAX_TURN_RAD_S = math.radians(60)
MAX_MOTION_DT_S = 0.2  # a camera stall never creates a large catch-up step
POSTURE_HOLD_S = 0.4  # a posture change must hold this long before the body changes
BLEND_S = 0.6  # how long the body takes to change posture
ARM_SMOOTH = 0.35  # EMA weight of each new arm direction
ARM_FORGET_S = 0.5  # an arm unseen this long eases back to the template
OUTSIDE_MARGIN = 0.3  # metres beyond the tracked area's edge...
OUTSIDE_S = 1.0  # ...for this long, and the person has left the area
SHOULDER_WIDTH = 0.24  # of body height, shoulder to shoulder


def _unit(v: Vec3) -> Vec3:
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) or 1e-9
    return (v[0] / n, v[1] / n, v[2] / n)


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _lerp3(a: Vec3, b: Vec3, k: float) -> Vec3:
    return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k)


class OneEuro:
    """One-Euro filter (Casiez et al.): smooth when slow, responsive when fast."""

    def __init__(self, min_cutoff: float = 0.4, beta: float = 2.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: float | None = None
        self.dx = 0.0
        self.t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self, x: float, t: float) -> float:
        self.x, self.dx, self.t = x, 0.0, t
        return x

    def __call__(self, x: float, t: float) -> float:
        if self.x is None or self.t is None or t <= self.t:
            return self.reset(x, t) if self.x is None else self.x
        dt = t - self.t
        dx = (x - self.x) / dt
        self.dx += self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


@dataclass
class _Arm:
    upper: Vec3 | None = None  # unit direction shoulder -> elbow, body frame (f, v, l)
    fore: Vec3 | None = None  # unit direction elbow -> wrist
    seen_at: float = -math.inf


@dataclass
class LiveAvatar:
    height: float = 1.7
    root: tuple[float, float] | None = None
    heading: float = 0.0
    posture: str = "standing"  # the posture the body is in (or blending to)
    _fx: OneEuro = field(default_factory=OneEuro)
    _fz: OneEuro = field(default_factory=OneEuro)
    _anchor: str | None = None
    _avail_since: dict[str, float] = field(default_factory=dict)
    _avail_last: dict[str, float] = field(default_factory=dict)
    _jump_since: float | None = None
    _jump_target: tuple[float, float] | None = None
    _measurement_t: float | None = None
    _wanted: str = "standing"
    _wanted_since: float = 0.0
    _from_pose: BodyPose | None = None
    _blend_start: float = -math.inf
    _arms: dict[str, _Arm] = field(default_factory=lambda: {"left": _Arm(), "right": _Arm()})
    _t: float = 0.0
    _outside_since: float | None = None
    _size_scale: float = 1.0  # size-cue distance correction, learnt against the floor anchor
    outside: bool = False  # they've been out of the tracked area for a while

    @property
    def ready(self) -> bool:
        return self.root is not None

    # --- measurements ---

    def _vertical_plane(self, camera: Camera) -> tuple[float, float, float]:
        """(nx, nz, dist): the vertical plane through the root, square to the camera's view."""
        o, f = camera.position, camera.forward
        n = math.hypot(f[0], f[2])
        nx, nz = (f[0] / n, f[2] / n) if n > 1e-3 else (0.0, -1.0)
        assert self.root is not None
        return nx, nz, (self.root[0] - o[0]) * nx + (self.root[1] - o[2]) * nz

    def _on_vertical(self, camera: Camera, u: float, v: float) -> Vec3 | None:
        nx, nz, dist = self._vertical_plane(camera)
        d, o = camera.ray(u, v), camera.position
        along = d[0] * nx + d[2] * nz
        if along <= 1e-6:
            return None
        s = dist / along
        return (o[0] + d[0] * s, o[1] + d[1] * s, o[2] + d[2] * s)

    def _anchors(self, pts: dict[str, tuple[float, float]], camera: Camera, posture: str) -> dict[str, Vec3]:
        """Floor-point estimates from each depth anchor available this frame."""
        out: dict[str, Vec3] = {}
        if posture in ("floor", "lying"):
            hips = _mean(pts, ("left_hip", "right_hip"))
            if hips is not None and (p := _on_plane(camera, *hips, FLOOR_BODY_Y)) is not None:
                out["floor_hips"] = p
            return out
        if "left_ankle" in pts and "right_ankle" in pts:
            if (p := _on_plane(camera, *_mean(pts, ("left_ankle", "right_ankle")), ANKLE_Y)) is not None:  # type: ignore[misc]
                out["ankles"] = p
        if "left_hip" in pts and "right_hip" in pts:
            y = HIP_HEIGHT.get(posture, 0.53) * self.height
            if (p := _on_plane(camera, *_mean(pts, ("left_hip", "right_hip")), y)) is not None:  # type: ignore[misc]
                out["hips"] = p
        return out

    def _measure_root(self, pts, camera: Camera, room: Room, posture: str, t: float) -> tuple[float, float] | None:
        found = self._anchors(pts, camera, posture)
        for name in list(self._avail_since):
            if name not in found and t - self._avail_last[name] > ANCHOR_GAP_S:
                del self._avail_since[name], self._avail_last[name]
        for name in found:
            self._avail_since.setdefault(name, t)
            self._avail_last[name] = t
        if self._anchor in found:
            p = found[self._anchor]
        else:
            # Switch only to an anchor that has been available for a while (depth estimates
            # from different body parts disagree a little: switching often is jitter). With no
            # anchor in use yet (or only a first guess), take any.
            fresh = self._anchor is None
            steady = [n for n in found if fresh or t - self._avail_since[n] >= SWITCH_ANCHOR_S]
            if not steady:
                p = None
            else:
                self._anchor = "ankles" if "ankles" in steady else steady[0]
                p = found[self._anchor]
        if posture in ("standing", "sitting") and self._anchor != "ankles":
            # Without the feet, the floor anchor rests on a guessed body proportion; the person's
            # apparent size is a second, independent cue.
            q = self._size_cue(pts, camera, posture)
            if q is not None:
                o = camera.position
                if p is not None:
                    # Learn how the size cue's distance compares with the floor anchor's, so it
                    # stays consistent with it when the anchor drops out (real proportions differ
                    # a little from the template's).
                    ratio = math.hypot(p[0] - o[0], p[2] - o[2]) / max(0.1, math.hypot(q[0] - o[0], q[2] - o[2]))
                    if 0.7 < ratio < 1.3:
                        self._size_scale += 0.1 * (ratio - self._size_scale)
                k = self._size_scale
                q = (o[0] + (q[0] - o[0]) * k, 0.0, o[2] + (q[2] - o[2]) * k, q[3])
                # Only when there's no floor anchor: at ~3 px of keypoint noise the size cue is a few
                # percent out (about 10 cm at 3 m), noisier than an anchor, so blending it in every
                # frame made a still person wander. It keeps them placed when the hips drop out.
                anchor_lost = self._anchor is None or t - self._avail_last.get(self._anchor, -math.inf) > ANCHOR_GAP_S
                if p is None and anchor_lost:  # not for a dropped joint or two: hold still instead
                    p = q
        if p is None and self.root is None and len(pts) >= 2:
            p = _at_distance(camera, pts, FALLBACK_DISTANCE_M)  # a first guess only
        if p is None:
            return None
        return (p[0], p[2])

    def _size_cue(self, pts, camera: Camera, posture: str):
        """(x, 0, z, sigma): where the person stands, from how tall they look. The highest and
        lowest marks seen (as calibration uses) are a known height apart; find the distance along
        the low mark's ray at which that height spans what the camera saw."""
        from hearth.sim.live.calibrate import frame_from

        mark = frame_from([(n, u, v) for n, (u, v) in pts.items()])
        if mark is None:
            return None
        scale = self.height if posture == "standing" else self.height * 0.75  # seated: shorter
        span = (mark.high_frac - mark.low_frac) * scale
        d, o = camera.ray(*mark.low), camera.position

        def high_v(s: float) -> float | None:
            low = (o[0] + d[0] * s, o[1] + d[1] * s, o[2] + d[2] * s)
            q = camera.project((low[0], low[1] + span, low[2]))
            return None if q is None else q[1]

        # Further along the ray the person looks smaller: the high mark sits lower in the image.
        # Scan for where the predicted high mark crosses the one seen, then bisect.
        target = mark.high[1]
        prev = None
        bracket = None
        for i in range(1, 80):
            dist = 0.25 * i  # up to 20 m along the ray
            v = high_v(dist)
            if v is not None and prev is not None and (prev[1] - target) * (v - target) <= 0:
                bracket = (prev[0], dist)
                break
            prev = (dist, v) if v is not None else None
        if bracket is None:
            return None
        lo, hi = bracket
        for _ in range(30):
            mid = (lo + hi) / 2
            v = high_v(mid)
            if v is None:
                return None
            if (high_v(lo) - target) * (v - target) <= 0:
                hi = mid
            else:
                lo = mid
        s = (lo + hi) / 2
        horizontal = math.hypot(d[0], d[2]) * s
        # Across the view, follow the middle of the torso, not one knee or one hip.
        torso = _mean(pts, ("neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip"))
        if torso is not None:
            d = camera.ray(torso[0], mark.low[1])
        flat = math.hypot(d[0], d[2]) or 1e-9
        x, z = o[0] + d[0] / flat * horizontal, o[2] + d[2] / flat * horizontal
        pixels = max(1e-3, abs(mark.low[1] - mark.high[1]))
        sigma = math.hypot(x - o[0], z - o[2]) * (0.006 / pixels)  # ~3 px of error in the span
        return (x, 0.0, z, sigma)

    def _measure_heading(self, pts, camera: Camera, posture: str) -> float | None:
        if posture in ("floor", "lying"):
            hips = _mean(pts, ("left_hip", "right_hip"))
            head = _mean(pts, ("neck",)) or _mean(pts, ("left_shoulder", "right_shoulder")) or _mean(pts, HEAD)
            if hips is None or head is None:
                return None
            a, b = _on_plane(camera, *hips, FLOOR_BODY_Y), _on_plane(camera, *head, FLOOR_BODY_Y)
            if a is None or b is None or math.hypot(b[0] - a[0], b[2] - a[2]) < 0.2:
                return None
            return math.atan2(b[0] - a[0], b[2] - a[2])  # the lying body's forward axis points to the head
        if "left_shoulder" not in pts or "right_shoulder" not in pts or self.root is None:
            return None
        ls = self._on_vertical(camera, *pts["left_shoulder"])
        rs = self._on_vertical(camera, *pts["right_shoulder"])
        if ls is None or rs is None:
            return None
        nx, nz, _ = self._vertical_plane(camera)
        lat = (ls[0] - rs[0]) * -nz + (ls[2] - rs[2]) * nx  # the person's left, measured across the view
        width = SHOULDER_WIDTH * self.height
        if abs(lat) < 0.25 * width:
            return None  # side-on: can't tell which way they face
        # Facing the camera, their left appears on the camera's right (lat > 0): heading towards the lens.
        towards = math.atan2(-nx, -nz)
        turn = math.acos(min(1.0, abs(lat) / width))
        base = towards if lat > 0 else towards + math.pi
        candidates = (base + turn, base - turn)
        return min(candidates, key=lambda h: abs(math.remainder(h - self.heading, 2 * math.pi)))

    def _measure_arms(self, pts, camera: Camera, t: float) -> None:
        fwd = (math.sin(self.heading), 0.0, math.cos(self.heading))
        left = (math.cos(self.heading), 0.0, -math.sin(self.heading))

        def body_dir(a: Vec3, b: Vec3) -> Vec3:
            d = _sub(b, a)
            return _unit((d[0] * fwd[0] + d[2] * fwd[2], d[1], d[0] * left[0] + d[2] * left[2]))

        for side, arm in self._arms.items():
            s, e, w = (pts.get(f"{side}_{j}") for j in ("shoulder", "elbow", "wrist"))
            if s is None or e is None:
                continue
            S, E = self._on_vertical(camera, *s), self._on_vertical(camera, *e)
            if S is None or E is None:
                continue
            up = body_dir(S, E)
            arm.upper = up if arm.upper is None else _unit(_lerp3(arm.upper, up, ARM_SMOOTH))
            if w is not None and (W := self._on_vertical(camera, *w)) is not None:
                fo = body_dir(E, W)
                arm.fore = fo if arm.fore is None else _unit(_lerp3(arm.fore, fo, ARM_SMOOTH))
            arm.seen_at = t

    # --- per tick ---

    def update(self, keypoints: list[tuple[str, float, float]] | None, posture: str, camera: Camera, room: Room,
               t: float) -> None:  # fmt: skip
        """keypoints: this tick's new observation (None if there's nothing new); posture: the monitor's."""
        self._t = t
        posture = posture if posture in TEMPLATE else self._wanted
        if posture != self._wanted:
            self._wanted, self._wanted_since = posture, t
        if self._wanted != self.posture and t - self._wanted_since >= POSTURE_HOLD_S:
            self._from_pose = self._pose(t)
            self.posture, self._blend_start = self._wanted, t
            self._anchor = None  # lying down or getting up moves the depth anchor
        if not keypoints:
            return
        pts = {n: (u, v) for n, u, v in keypoints}
        gap = t - self._measurement_t if self._measurement_t is not None else 0.0
        dt = min(MAX_MOTION_DT_S, max(0.0, gap))
        if gap > ANCHOR_GAP_S:
            self._jump_since, self._jump_target = None, None
        self._measurement_t = t
        meas = self._measure_root(pts, camera, room, self.posture, t)
        if meas is not None and all(math.isfinite(v) for v in meas):
            if self.root is None:  # only the first placement may set the position directly
                self.root = (self._fx.reset(meas[0], t), self._fz.reset(meas[1], t))
            else:
                accept = True
                if math.dist(meas, self.root) > JUMP_M:
                    if self._jump_target is None or math.dist(meas, self._jump_target) > JUMP_TARGET_M:
                        self._jump_since, self._jump_target = t, meas
                    # A smoothly moving target is consistent too; don't require a
                    # fast walker to stop before the slower avatar can follow.
                    self._jump_target = meas
                    accept = t - self._jump_since >= JUMP_HOLD_S
                else:
                    self._jump_since, self._jump_target = None, None
                if accept and math.dist(meas, self.root) > POSITION_DEADBAND_M:
                    target = (self._fx(meas[0], t), self._fz(meas[1], t))
                    dist = math.dist(target, self.root)
                    fraction = min(1.0, MAX_SPEED_M_S * dt / max(dist, 1e-9))
                    self.root = tuple(a + (b - a) * fraction for a, b in zip(self.root, target))
                    # Keep the filters at the displayed position when speed-limited.
                    if fraction < 1.0:
                        self._fx.reset(self.root[0], t)
                        self._fz.reset(self.root[1], t)
                else:
                    self._fx.reset(self.root[0], t)
                    self._fz.reset(self.root[1], t)
        else:
            self._jump_since, self._jump_target = None, None
        if self.root is None:
            return
        out = (abs(self.root[0]) > room.width / 2 + OUTSIDE_MARGIN
               or not (-room.depth - OUTSIDE_MARGIN < self.root[1] < OUTSIDE_MARGIN))  # fmt: skip
        self._outside_since = (self._outside_since if self._outside_since is not None else t) if out else None
        self.outside = out and t - self._outside_since >= OUTSIDE_S  # type: ignore[operator]
        h = self._measure_heading(pts, camera, self.posture)
        if h is not None:
            turn = 0.2 * math.remainder(h - self.heading, 2 * math.pi)
            self.heading += max(-MAX_TURN_RAD_S * dt, min(MAX_TURN_RAD_S * dt, turn))
        if self.posture in ("standing", "sitting"):
            self._measure_arms(pts, camera, t)

    # --- the body ---

    def _template(self, posture: str) -> BodyPose:
        return dict(KEY_POSES[TEMPLATE.get(posture, "standing")])

    def _with_arms(self, pose: BodyPose, t: float) -> BodyPose:
        if self.posture not in ("standing", "sitting"):
            return pose
        out = dict(pose)
        for side, arm in self._arms.items():
            if arm.upper is None:
                continue
            sh, el, wr = f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"
            # Seen recently: follow it. Unseen: ease back to the template over ARM_FORGET_S.
            k = 1.0 - min(1.0, max(0.0, (t - arm.seen_at - ARM_FORGET_S) / ARM_FORGET_S))
            if k <= 0.0:
                continue
            upper_len = math.dist(pose[sh], pose[el])
            fore_len = math.dist(pose[el], pose[wr])
            t_up, t_fo = _unit(_sub(pose[el], pose[sh])), _unit(_sub(pose[wr], pose[el]))
            up = _unit(_lerp3(t_up, arm.upper, k))
            fo = _unit(_lerp3(t_fo, arm.fore, k)) if arm.fore is not None else t_fo
            elbow = (pose[sh][0] + up[0] * upper_len, pose[sh][1] + up[1] * upper_len, pose[sh][2] + up[2] * upper_len)
            out[el] = elbow
            out[wr] = (elbow[0] + fo[0] * fore_len, elbow[1] + fo[1] * fore_len, elbow[2] + fo[2] * fore_len)
        return out

    def _pose(self, t: float) -> BodyPose:
        target = self._template(self.posture)
        k = (t - self._blend_start) / BLEND_S
        if self._from_pose is not None and k < 1.0:
            return blend_body(self._from_pose, target, smoothstep(max(0.0, k)))
        return self._with_arms(target, t)

    def joints(self) -> dict[str, Vec3]:
        assert self.root is not None
        return place_body(self._pose(self._t), self.heading, self.root[0], self.root[1], 0.0, self.height)
