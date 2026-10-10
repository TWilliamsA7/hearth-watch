"""Stick-figure body model for the simulated resident.

Key poses are defined once in body units, side-on: f = forward distance from the root,
v = height above the support surface, 1.0 = standing height. They are then projected
into the camera frame for whichever way the person faces. Facing the camera, the
forward axis points into the lens and drops out, and each joint's lateral offset takes
its place. That is the same foreshortening a real camera produces, so a forward bend
seen head-on shows a shorter torso rather than a large torso angle.

Poses are blended in image space, so a transition between a front view and a side view
(or into a lying pose) stays continuous.
"""

from __future__ import annotations

import math

from hearth_schema.models import KeypointName

from hearth.sim.scenario import Facing, PoseName, Scene

JOINTS: tuple[KeypointName, ...] = KeypointName.__args__  # type: ignore[attr-defined]

BodyPose = dict[str, tuple[float, float]]
"""joint name -> (f, v) in body units."""
ImagePose = dict[str, tuple[float, float]]
"""joint name -> (x, y) in normalized frame coordinates (may fall outside the frame)."""

# Lateral offset of each joint from the body's midline; left side positive.
_HALF_WIDTH = {"eye": 0.03, "ear": 0.06, "shoulder": 0.12, "elbow": 0.15, "wrist": 0.16,
               "hip": 0.08, "knee": 0.08, "ankle": 0.08}  # fmt: skip
LATERAL: dict[str, float] = {"nose": 0.0, "neck": 0.0}
for _part, _w in _HALF_WIDTH.items():
    LATERAL[f"left_{_part}"] = _w
    LATERAL[f"right_{_part}"] = -_w

# Seen side-on, the far half of the body sits mostly behind the near half.
_SIDE_LATERAL = 0.15


def _sym(head, eyes, ears, neck, shoulders, elbows, wrists, hips, knees, ankles, **asym) -> BodyPose:
    """Build a pose whose left and right joints coincide side-on, with optional per-joint overrides."""
    pose = {"nose": head, "neck": neck}
    for part, fv in (("eye", eyes), ("ear", ears), ("shoulder", shoulders), ("elbow", elbows),
                     ("wrist", wrists), ("hip", hips), ("knee", knees), ("ankle", ankles)):  # fmt: skip
        pose[f"left_{part}"] = pose[f"right_{part}"] = fv
    pose.update(asym)
    return pose


KEY_POSES: dict[str, BodyPose] = {
    "standing": _sym(
        head=(0.05, 0.93), eyes=(0.04, 0.95), ears=(-0.01, 0.94), neck=(0.0, 0.84),
        shoulders=(0.0, 0.82), elbows=(-0.01, 0.64), wrists=(0.0, 0.47),
        hips=(0.0, 0.52), knees=(0.01, 0.28), ankles=(0.0, 0.03),
    ),
    "sitting": _sym(
        head=(0.04, 0.71), eyes=(0.03, 0.73), ears=(-0.02, 0.72), neck=(-0.02, 0.62),
        shoulders=(-0.02, 0.60), elbows=(0.02, 0.44), wrists=(0.15, 0.36),
        hips=(-0.05, 0.30), knees=(0.20, 0.30), ankles=(0.20, 0.03),
    ),
    "bending": _sym(
        head=(0.40, 0.64), eyes=(0.40, 0.66), ears=(0.35, 0.67), neck=(0.30, 0.70),
        shoulders=(0.28, 0.69), elbows=(0.32, 0.50), wrists=(0.36, 0.33),
        hips=(-0.08, 0.50), knees=(0.03, 0.27), ankles=(0.0, 0.03),
    ),
    # On the back, arms by the sides. Root is under the hips. Heights include the body's
    # thickness as seen from a camera mounted above the floor.
    "lying": _sym(
        head=(0.42, 0.18), eyes=(0.42, 0.20), ears=(0.40, 0.15), neck=(0.32, 0.14),
        shoulders=(0.30, 0.15), elbows=(0.14, 0.10), wrists=(0.0, 0.07),
        hips=(0.0, 0.10), knees=(-0.24, 0.09), ankles=(-0.48, 0.04),
    ),
    # Sprawled on the side: one arm thrown out, one knee bent.
    "fallen": _sym(
        head=(0.44, 0.12), eyes=(0.44, 0.15), ears=(0.41, 0.14), neck=(0.33, 0.15),
        shoulders=(0.31, 0.17), elbows=(0.20, 0.20), wrists=(0.12, 0.06),
        hips=(0.0, 0.14), knees=(-0.25, 0.10), ankles=(-0.48, 0.04),
        left_elbow=(0.46, 0.08), left_wrist=(0.58, 0.04),
        left_knee=(-0.18, 0.24), left_ankle=(-0.36, 0.05),
    ),
}

SIDE_ONLY_POSES: frozenset[str] = frozenset({"lying", "fallen"})
"""Lying poses are always seen along the body's length; there is no head-on view of them."""

STEP_HZ = 0.9  # full gait cycles per second


def body_pose(name: PoseName, gait_phase: float | None = None, gait_amount: float = 1.0) -> BodyPose:
    """Key pose in body units, with a walking cycle layered on when gait_phase is given."""
    pose = dict(KEY_POSES[name])
    if gait_phase is None or gait_amount <= 0.0:
        return pose

    s, c, a = math.sin(gait_phase), math.cos(gait_phase), gait_amount
    bob = 0.012 * (abs(c) - 0.5)

    def shift(joint: str, df: float = 0.0, dv: float = 0.0) -> None:
        f, v = pose[joint]
        pose[joint] = (f + df * a, v + dv * a)

    for joint in pose:
        shift(joint, dv=bob)
    for joint in ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "neck"):
        shift(joint, df=0.02)
    shift("left_ankle", df=0.13 * s, dv=0.04 * max(0.0, c))
    shift("right_ankle", df=-0.13 * s, dv=0.04 * max(0.0, -c))
    shift("left_knee", df=0.07 * s, dv=0.02 * max(0.0, c))
    shift("right_knee", df=-0.07 * s, dv=0.02 * max(0.0, -c))
    shift("left_wrist", df=-0.09 * s)
    shift("right_wrist", df=0.09 * s)
    shift("left_elbow", df=-0.04 * s)
    shift("right_elbow", df=0.04 * s)
    return pose


def project(pose: BodyPose, facing: Facing, root_x: float, base_y: float, scene: Scene) -> ImagePose:
    """Place a body-unit pose in the frame with its root at (root_x, base_y)."""
    sx = scene.person_height / scene.frame_aspect  # body unit -> normalized x
    sy = scene.person_height  # body unit -> normalized y
    out: ImagePose = {}
    for joint, (f, v) in pose.items():
        lat = LATERAL[joint]
        if facing == "camera":
            u = lat  # facing the lens, the person's left side appears on the image's right
        else:
            u = (1.0 if facing == "right" else -1.0) * f + _SIDE_LATERAL * lat
        out[joint] = (root_x + u * sx, base_y - v * sy)
    return out


def blend(a: ImagePose, b: ImagePose, alpha: float) -> ImagePose:
    if alpha <= 0.0:
        return dict(a)
    if alpha >= 1.0:
        return dict(b)
    return {j: (ax + (b[j][0] - ax) * alpha, ay + (b[j][1] - ay) * alpha) for j, (ax, ay) in a.items()}


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def ease_in(x: float) -> float:
    """Accelerating, like a body under gravity."""
    x = min(max(x, 0.0), 1.0)
    return x * x
