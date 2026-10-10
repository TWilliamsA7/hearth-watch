"""Lift a live camera observation (2D keypoints) into the simulated room as a 3D body.

One camera gives no depth, so depth comes from the floor: a body part whose height above
the floor is known (feet on the floor, or hips at a typical height for the posture) is
found where its ray from the lens meets that height. Upright bodies are then placed on a
plane at that depth facing the lens. Bodies on the floor lie on the floor, so every joint
gets its own floor intersection.

The result is only as good as the match between the simulated camera (CameraSpec: preset
"jetson", height, tilt, fov) and the real one. The person's posture is not guessed here:
the caller passes what the real monitor concluded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hearth.sim.live.camera import Camera
from hearth.sim.live.room import Room

Vec3 = tuple[float, float, float]

# Heights above the floor, as a fraction of body height unless noted, used as depth anchors.
ANKLE_Y = 0.06  # metres
UPRIGHT_ANCHORS = (("ankle", None), ("knee", 0.28), ("hip", 0.53), ("neck", 0.82))
SITTING_ANCHORS = (("ankle", None), ("knee", 0.30), ("hip", 0.30), ("neck", 0.62))
LYING_Y = {"head": 0.22, "default": 0.18}  # metres: mid-height of a body lying on the floor
HEAD = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
WALL_MARGIN = 0.25


@dataclass(frozen=True)
class LiftedBody:
    joints: dict[str, Vec3]
    root: tuple[float, float]  # floor (x, z) under the hips
    heading: float  # radians: 0 faces +z (towards the camera wall), pi/2 faces +x
    height: float


def _on_plane(camera: Camera, u: float, v: float, y: float) -> Vec3 | None:
    """Where the ray through (u, v) is at height y, if it gets there in front of the lens."""
    d = camera.ray(u, v)
    o = camera.position
    if abs(d[1]) < 1e-6:
        return None
    s = (y - o[1]) / d[1]
    if s <= 0.05:
        return None
    return (o[0] + d[0] * s, y, o[2] + d[2] * s)


def _mean(points: dict[str, tuple[float, float]], names: tuple[str, ...]) -> tuple[float, float] | None:
    found = [points[n] for n in names if n in points]
    if not found:
        return None
    return (sum(p[0] for p in found) / len(found), sum(p[1] for p in found) / len(found))


def _part(points: dict[str, tuple[float, float]], part: str) -> tuple[float, float] | None:
    return _mean(points, (f"left_{part}", f"right_{part}")) if part != "neck" else points.get("neck")


def _depth(camera: Camera, p: Vec3) -> float:
    o, f = camera.position, camera.forward
    return (p[0] - o[0]) * f[0] + (p[1] - o[1]) * f[1] + (p[2] - o[2]) * f[2]


def _heading(joints: dict[str, Vec3]) -> float:
    for left, right in (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip")):
        if left in joints and right in joints:
            lx, lz = joints[left][0] - joints[right][0], joints[left][2] - joints[right][2]
            if math.hypot(lx, lz) > 1e-6:
                # The body's left is (cos h, 0, -sin h) for heading h (see hearth.sim.world).
                return math.atan2(-lz, lx)
    return 0.0  # unknown: face the camera


def lift(keypoints: list[tuple[str, float, float]], camera: Camera, room: Room, *, height: float = 1.7,
         posture: str = "standing") -> LiftedBody | None:
    """keypoints: (name, x, y) normalized to the camera frame (the contract's Keypoint).
    posture: "standing" | "sitting" | "lying" | "floor" (from the monitor). None if nothing usable."""
    points = {name: (x, y) for name, x, y in keypoints}
    if not points:
        return None
    joints: dict[str, Vec3] = {}
    if posture in ("floor", "lying"):
        for name, (u, v) in points.items():
            p = _on_plane(camera, u, v, LYING_Y["head" if name in HEAD else "default"])
            if p is not None:
                joints[name] = p
    else:
        anchors = SITTING_ANCHORS if posture == "sitting" else UPRIGHT_ANCHORS
        anchor = None
        for part, frac in anchors:
            uv = _part(points, part)
            if uv is not None:
                anchor = _on_plane(camera, uv[0], uv[1], ANKLE_Y if frac is None else frac * height)
                if anchor is not None:
                    break
        if anchor is None:
            return None
        if _depth(camera, anchor) < 0.3:
            return None
        # An upright body is vertical: put every joint on the vertical plane through the anchor,
        # square to the camera's horizontal view direction (a tilted camera's own depth planes lean).
        o, f = camera.position, camera.forward
        n = math.hypot(f[0], f[2])
        nx, nz = (f[0] / n, f[2] / n) if n > 1e-3 else (0.0, -1.0)
        dist = (anchor[0] - o[0]) * nx + (anchor[2] - o[2]) * nz
        for name, (u, v) in points.items():
            d = camera.ray(u, v)
            along = d[0] * nx + d[2] * nz
            if along <= 1e-6:
                continue
            s = dist / along
            joints[name] = (o[0] + d[0] * s, o[1] + d[1] * s, o[2] + d[2] * s)
    if not joints:
        return None

    hip = _mean({k: (p[0], p[2]) for k, p in joints.items()}, ("left_hip", "right_hip"))
    if hip is None:
        hip = _mean({k: (p[0], p[2]) for k, p in joints.items()}, tuple(joints))
    assert hip is not None
    # Keep the body inside the room: a bad calibration shouldn't put someone through a wall.
    x0, x1 = -room.width / 2 + WALL_MARGIN, room.width / 2 - WALL_MARGIN
    z0, z1 = -room.depth + WALL_MARGIN, -WALL_MARGIN
    dx = min(max(hip[0], x0), x1) - hip[0]
    dz = min(max(hip[1], z0), z1) - hip[1]
    if dx or dz:
        joints = {k: (p[0] + dx, p[1], p[2] + dz) for k, p in joints.items()}
    root = (hip[0] + dx, hip[1] + dz)
    return complete(LiftedBody(joints, root, _heading(joints), height), posture)


# Body template per posture, for the joints the camera didn't see.
TEMPLATES = {"standing": "standing", "sitting": "sitting", "lying": "side_lying", "floor": "side_lying"}


def complete(body: LiftedBody, posture: str) -> LiftedBody:
    """Fill in joints the camera didn't see (a close camera often misses the head or legs) from
    a body in this posture, standing where the seen joints are: the avatar is always a whole
    person. Seen joints are kept exactly as lifted."""
    from hearth.sim.pose import KEY_POSES
    from hearth.sim.world import body as pose_body

    tpl = pose_body(KEY_POSES[TEMPLATES.get(posture, "standing")], body.heading, body.root[0], body.root[1], 0.0,
                    body.height)  # fmt: skip
    seen = [n for n in body.joints if n in tpl]
    if not seen or len(seen) == len(tpl):
        return body
    # Line the template up with what was seen, across the floor only: its height is the body's own.
    dx = sum(body.joints[n][0] - tpl[n][0] for n in seen) / len(seen)
    dz = sum(body.joints[n][2] - tpl[n][2] for n in seen) / len(seen)
    joints = {n: (p[0] + dx, p[1], p[2] + dz) for n, p in tpl.items()}
    joints.update(body.joints)
    return LiftedBody(joints, body.root, body.heading, body.height)
