"""The live room's camera: a pinhole on the wall, and what it reports.

The camera projects each body's joints into a normalized image (0..1, origin top left).
A joint is reported only if it's in the picture, in front of the lens, and not hidden:
furniture hides what is behind it (a ray from the lens to the joint through any piece's
box), and a person nearer the camera hides joints inside their outline. Noise is the
scripted pipeline's NoiseConfig, applied the same way, and features are computed with
hearth.sim.features, so a live observation looks exactly like a scripted one.

The camera never sees ground truth, labels, activities or ids it couldn't know: an
observation carries the keypoints and face result only. Like the scripted generator, it
tags each person's observation with their person_id (an anonymous track id), which
assumes perfect tracking; see docs/interfaces.md.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from hearth_schema.models import Heartbeat, Observation

from hearth.sim.features import FeatureTracker
from hearth.sim.generator import MIN_VISIBLE_KEYPOINTS, SIM_EPOCH, SOURCE, NoiseConfig
from hearth.sim.pose import JOINTS
from hearth.sim.world import Vec3, WorldPose

from hearth.sim.live.room import Piece, Room
from hearth.sim.live.spec import CameraSpec

Box3 = tuple[float, float, float, float, float, float]  # x0, y0, z0, x1, y1, z1
PERSON_PAD = 0.012  # normalized: a nearer body hides this much around its keypoints
FACE_FRONT = math.radians(50)
FACE_AWAY = math.radians(130)


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(a: Vec3) -> Vec3:
    n = math.sqrt(_dot(a, a)) or 1e-9
    return (a[0] / n, a[1] / n, a[2] / n)


def placement(room: Room, spec: CameraSpec) -> tuple[Vec3, Vec3]:
    """(position, target) for a preset."""
    w, d = room.width, room.depth
    h = min(spec.height, room.height - 0.1)
    look = (0.0, 0.6, -d * 0.55)
    if spec.preset == "corner_left":
        return (-w / 2 + 0.15, h, -0.15), look
    if spec.preset == "corner_right":
        return (w / 2 - 0.15, h, -0.15), look
    if spec.preset == "front":
        return (0.0, h, -0.12), (0.0, 0.5, -d * 0.6)
    if spec.preset == "jetson":
        # The Jetson's mount: centre of the front wall, tilted down, aimed `reach` metres in.
        reach = min(spec.reach, d - 0.2)
        return (0.0, h, -0.12), (0.0, h - reach * math.tan(math.radians(spec.tilt)), -0.12 - reach)
    return (0.0, room.height - 0.05, -d / 2 + 0.001), (0.0, 0.0, -d / 2)  # ceiling, looking straight down


def occluding_parts(piece: Piece) -> list[Box3]:
    """The solid parts of a piece, as boxes: a table hides nothing below its top, a chair
    only its seat and back; couches, beds and cabinets are solid to the floor."""
    name, h = piece.kind.name, piece.kind.h
    if piece.kind.walkable:
        return []

    def part(lx0: float, lx1: float, lz0: float, lz1: float, y0: float, y1: float) -> Box3:
        a, b = piece.local(lx0, lz0), piece.local(lx1, lz1)
        return (min(a[0], b[0]), y0, min(a[1], b[1]), max(a[0], b[0]), y1, max(a[1], b[1]))

    w, d = piece.kind.w / 2, piece.kind.d / 2
    if name in ("table", "side_table"):
        return [part(-w, w, -d, d, h - 0.06, h)]
    if name == "chair":
        return [part(-w, w, -d, d, 0.40, 0.47), part(-w, w, -d, -d + 0.08, 0.47, h)]
    if name in ("couch", "loveseat", "armchair"):
        return [part(-w, w, -d, d, 0.05, 0.45), part(-w, w, -d, -d + 0.2, 0.45, h),
                part(-w, -w + 0.18, -d, d, 0.45, 0.65), part(w - 0.18, w, -d, d, 0.45, 0.65)]  # fmt: skip
    if name == "bed":
        return [part(-w, w, -d, d, 0.1, h), part(-w, w, -d, -d + 0.08, h, 1.0)]
    if name == "plant":
        return [part(-w * 0.5, w * 0.5, -d * 0.5, d * 0.5, 0.0, 0.4), part(-w, w, -d, d, 0.5, h)]
    return [part(-w, w, -d, d, 0.0, h)]


class Camera:
    def __init__(self, room: Room, spec: CameraSpec) -> None:
        self.spec = spec
        self.position, self.target = placement(room, spec)
        # Looking straight down from the ceiling needs a wide lens to see the room.
        self.fov, self.aspect = (max(spec.fov, 100.0) if spec.preset == "ceiling" else spec.fov), spec.aspect
        f = _unit(_sub(self.target, self.position))
        up = (0.0, 1.0, 0.0) if abs(f[1]) < 0.95 else (0.0, 0.0, -1.0)
        r = _unit(_cross(f, up))
        self.forward, self.right, self.up = f, r, _cross(r, f)
        self.tan_v = math.tan(math.radians(self.fov) / 2)
        self.tan_h = self.tan_v * self.aspect
        self.boxes: list[Box3] = [b for p in room.pieces for b in occluding_parts(p)]

    def project(self, p: Vec3) -> tuple[float, float, float] | None:
        """(x, y, depth) in the image, or None if behind the lens."""
        d = _sub(p, self.position)
        z = _dot(d, self.forward)
        if z < 0.05:
            return None
        return (0.5 + _dot(d, self.right) / z / (2 * self.tan_h), 0.5 - _dot(d, self.up) / z / (2 * self.tan_v), z)

    def ray(self, u: float, v: float) -> Vec3:
        """The inverse of project(): the direction through image point (u, v), scaled so its
        forward component is 1, so position + ray * depth is the point at that depth."""
        a, b = (2 * u - 1) * self.tan_h, (1 - 2 * v) * self.tan_v
        f, r, up = self.forward, self.right, self.up
        return (f[0] + a * r[0] + b * up[0], f[1] + a * r[1] + b * up[1], f[2] + a * r[2] + b * up[2])

    def blocked(self, p: Vec3) -> bool:
        """Does a piece of furniture stand between the lens and p?"""
        o = self.position
        d = _sub(p, o)
        for box in self.boxes:
            t0, t1 = 0.0, 0.97  # stop short of the joint: a hand resting on a table is still seen
            for k in range(3):
                lo, hi = box[k], box[k + 3]
                if abs(d[k]) < 1e-9:
                    if not lo <= o[k] <= hi:
                        t0, t1 = 1.0, 0.0
                        break
                    continue
                a, b = (lo - o[k]) / d[k], (hi - o[k]) / d[k]
                if a > b:
                    a, b = b, a
                t0, t1 = max(t0, a), min(t1, b)
                if t0 > t1:
                    break
            if t0 <= t1:
                return True
        return False

    def sees(self, pose: WorldPose) -> dict[str, tuple[float, float, float]]:
        """Joints in the picture and not behind furniture: name -> (x, y, depth)."""
        out = {}
        for j, p in pose.items():
            q = self.project(p)
            if q is None or not (0.0 <= q[0] <= 1.0 and 0.0 <= q[1] <= 1.0):
                continue
            if self.blocked(p):
                continue
            out[j] = q
        return out


@dataclass
class Seen:
    """One person as the camera found them this frame (noise-free), for truth and display."""

    person_id: str
    joints: dict[str, tuple[float, float, float]]  # visible joints only
    box: tuple[float, float, float, float] | None  # outline in the image, all joints in frame
    depth: float  # distance to the hips along the lens axis


class Observer:
    """Turns the world into contract records at the camera's frame rate."""

    def __init__(self, camera: Camera, *, scenario_id: str, seed: int, room_id: str, noise: NoiseConfig,
                 start: datetime = SIM_EPOCH, source: str = SOURCE) -> None:  # fmt: skip
        self.camera = camera
        self.scenario_id, self.seed, self.room_id = scenario_id, seed, room_id
        self.noise, self.start, self.source = noise, start, source
        self.rng = random.Random(f"camera/{seed}")
        self.frame = 0
        self.next_heartbeat = 0.0
        self.trackers: dict[str, FeatureTracker] = {}  # per person, smoothed like the edge's
        self.visible: dict[str, float] = {}  # person_id -> fraction of joints seen (noise-free), last frame

    @property
    def fps(self) -> float:
        return self.camera.spec.fps

    def due(self, t: float) -> bool:
        return t + 1e-9 >= self.frame / self.fps

    def emit(self, t: float, people: list[tuple[str, WorldPose, float, float, bool]]) -> list[Observation | Heartbeat]:
        """people: (person_id, world pose, heading, root distance, lying) for everyone present."""
        out: list[Observation | Heartbeat] = []
        if t + 1e-9 >= self.next_heartbeat:
            out.append(Heartbeat(timestamp=self._ts(t), source=self.source, source_type="simulation",
                                 room_id=self.room_id, fps=self.fps))  # fmt: skip
            self.next_heartbeat += 1.0
        n = self.frame
        self.frame += 1
        seen = [self._see(pid, pose) for pid, pose, *_ in people]
        self.visible = {s.person_id: len(s.joints) / len(JOINTS) for s in seen}
        if self.rng.random() < self.noise.frame_drop:
            return out
        emitted = False
        for (pid, pose, heading, _, lying), s in zip(people, seen):
            in_front = [o.box for o in seen if o is not s and o.box and o.depth < s.depth - 0.3]
            obs = self._observation(n, t, s, in_front, heading, pose, lying)
            if obs is not None:
                out.append(obs)
                emitted = True
        if not emitted:
            out.append(Observation.model_validate(self._base(n, t, None)))
        return out

    def _see(self, pid: str, pose: WorldPose) -> Seen:
        cam = self.camera
        visible = cam.sees(pose)
        projected = [q for q in (cam.project(p) for p in pose.values()) if q is not None]
        box = None
        if projected:
            xs, ys = [q[0] for q in projected], [q[1] for q in projected]
            box = (min(xs) - PERSON_PAD, min(ys) - PERSON_PAD, max(xs) + PERSON_PAD, max(ys) + PERSON_PAD)
        hips = [pose[j] for j in ("left_hip", "right_hip")]
        mid = ((hips[0][0] + hips[1][0]) / 2, (hips[0][1] + hips[1][1]) / 2, (hips[0][2] + hips[1][2]) / 2)
        depth = _dot(_sub(mid, cam.position), cam.forward)
        return Seen(pid, visible, box, depth)

    def _ts(self, t: float) -> datetime:
        return self.start + timedelta(seconds=round(t, 6))

    def _base(self, n: int, t: float, person_id: str | None) -> dict:
        suffix = f"-{person_id}" if person_id else ""
        return {"obs_id": f"sim-{self.scenario_id}-s{self.seed}-{n:06d}{suffix}", "timestamp": self._ts(t),
                "source": self.source, "source_type": "simulation", "room_id": self.room_id,
                "scenario_id": self.scenario_id, "person_id": person_id, "person_detected": False,
                "face_present": False}  # fmt: skip

    def _observation(self, n: int, t: float, s: Seen, in_front: list, heading: float, pose: WorldPose,
                     lying: bool) -> Observation | None:  # fmt: skip
        nz = self.noise
        jitter = nz.jitter * (1.5 if lying else 1.0)
        conf_mean = nz.confidence_mean - (nz.lying_penalty if lying else 0.0)
        keypoints = []
        for joint in JOINTS:
            # Every random number for every joint, seen or not, so the noise stays stable.
            dx, dy = self.rng.gauss(0.0, jitter), self.rng.gauss(0.0, jitter)
            conf = self.rng.gauss(conf_mean, nz.confidence_sd)
            dropped = self.rng.random() < nz.keypoint_dropout
            q = s.joints.get(joint)
            if q is None or dropped:
                continue
            x, y = q[0], q[1]
            if any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in in_front):
                continue
            x, y = x + dx, y + dy
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                continue
            keypoints.append({"name": joint, "x": round(x, 4), "y": round(y, 4),
                              "confidence": round(min(max(conf, 0.05), 0.99), 3)})  # fmt: skip
        tracker = self.trackers.get(s.person_id)
        if tracker is None:
            tracker = self.trackers[s.person_id] = FeatureTracker(self.camera.aspect)
        if len(keypoints) < MIN_VISIBLE_KEYPOINTS:
            tracker.lost()
            return None
        features = tracker.update(keypoints, t)
        face_present, face_conf = self._face(keypoints, heading, pose, lying)
        obs = self._base(n, t, s.person_id)
        obs.update(person_detected=True, keypoints=keypoints, features=features, face_present=face_present,
                   face_confidence=face_conf)  # fmt: skip
        return Observation.model_validate(obs)

    def _face(self, keypoints: list[dict], heading: float, pose: WorldPose, lying: bool) -> tuple[bool, float | None]:
        nz = self.noise
        nose = pose["nose"]
        to_cam = math.atan2(self.camera.position[0] - nose[0], self.camera.position[2] - nose[2])
        off = abs((to_cam - heading + math.pi) % (2 * math.pi) - math.pi)
        if lying:
            rate = nz.face_rate_lying
        elif off < FACE_FRONT:
            rate = nz.face_rate_front
        elif off > FACE_AWAY:
            rate = nz.face_rate_away
        else:
            rate = nz.face_rate_side
        roll, conf = self.rng.random(), self.rng.uniform(0.6, 0.95)
        if not any(k["name"] == "nose" for k in keypoints) or roll >= rate:
            return False, None
        return True, round(conf, 3)
