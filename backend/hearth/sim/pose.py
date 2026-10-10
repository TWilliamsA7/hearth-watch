"""Body model for the simulated resident.

Key poses are defined once in body units: f = forward from the root, v = height above
the support surface, l = towards the person's left; 1.0 = standing height. The root is
the point on the floor under the hips. hearth.sim.world turns a pose, a heading and a
place on the floor into metres in the room, and the camera image is taken from that
(world.to_image), so the 2D keypoints and the 3D view are always the same body.

Poses are blended bone by bone (blend_body): each bone turns steadily from one direction
to the other and keeps its standing length, and the feet move straight with the knees
bending to suit (so feet stay planted while someone sits down). Blending every joint in
a straight line would shorten limbs half way, and flip a shin over when it points
opposite ways in the two poses; it shows most when someone goes from upright to lying
down. Poses built by hand are put on the skeleton the same way (fit_bones).

Lying poses are built rather than written out: an upright arrangement of the limbs is
laid down head first in some direction and rolled onto the back, side or front (lay()).
"""

from __future__ import annotations

import math
import random

from hearth_schema.models import KeypointName

JOINTS: tuple[KeypointName, ...] = KeypointName.__args__  # type: ignore[attr-defined]

Vec = tuple[float, float, float]
BodyPose = dict[str, Vec]
"""joint name -> (f, v, l) in body units."""

# Lateral offset of each joint from the body's midline; left side positive.
_HALF_WIDTH = {"eye": 0.03, "ear": 0.06, "shoulder": 0.12, "elbow": 0.15, "wrist": 0.16,
               "hip": 0.08, "knee": 0.08, "ankle": 0.08}  # fmt: skip
LATERAL: dict[str, float] = {"nose": 0.0, "neck": 0.0}
for _part, _w in _HALF_WIDTH.items():
    LATERAL[f"left_{_part}"] = _w
    LATERAL[f"right_{_part}"] = -_w

SIDES = ("left", "right")
HEAD_JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
LYING_POSES: frozenset[str] = frozenset({"lying", "fallen", "supine", "prone", "side_lying"})
"""Poses with the body along the floor (or a bed): pose nets do worse on them."""

FLOOR_POSES: frozenset[str] = LYING_POSES | {"floor_sit", "hands_knees", "kneel", "half_kneel"}
"""Poses with the hips near the floor: where get_up starts from."""


def _sym(head, eyes, ears, neck, shoulders, elbows, wrists, hips, knees, ankles, **asym) -> BodyPose:
    """A pose from side-on (f, v) values; left and right differ only by LATERAL, unless overridden.

    An override is (f, v) (keeping the joint's lateral offset) or (f, v, l).
    """
    flat = {"nose": head, "neck": neck}
    for part, fv in (("eye", eyes), ("ear", ears), ("shoulder", shoulders), ("elbow", elbows),
                     ("wrist", wrists), ("hip", hips), ("knee", knees), ("ankle", ankles)):  # fmt: skip
        flat[f"left_{part}"] = flat[f"right_{part}"] = fv
    flat.update(asym)
    return {j: (p[0], p[1], p[2] if len(p) > 2 else LATERAL[j]) for j, p in flat.items()}


# --- the skeleton ---

# (parent, child) in an order where every parent is placed before its children. "hips" is
# the point between the hips, which the skeleton hangs from.
BONES: tuple[tuple[str, str], ...] = (
    ("hips", "left_hip"), ("hips", "right_hip"), ("hips", "neck"),
    ("neck", "left_shoulder"), ("neck", "right_shoulder"), ("neck", "nose"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("nose", "left_eye"), ("nose", "right_eye"), ("nose", "left_ear"), ("nose", "right_ear"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)  # fmt: skip


def _add(a: Vec, b: Vec, k: float = 1.0) -> Vec:
    return (a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k)


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(a: Vec) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def hips_of(pose: BodyPose) -> Vec:
    (a, b) = pose["left_hip"], pose["right_hip"]
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)


def _point(pose: BodyPose, name: str) -> Vec:
    return hips_of(pose) if name == "hips" else pose[name]


def bone_lengths(pose: BodyPose) -> dict[tuple[str, str], float]:
    return {(p, c): _norm(_sub(pose[c], _point(pose, p))) for p, c in BONES}


def fit_bones(pose: BodyPose, lengths: dict[tuple[str, str], float] | None = None) -> BodyPose:
    """Put a pose back on the skeleton: same bone directions, standing bone lengths.

    The hips stay where they are; everything else hangs from them.
    """
    lengths = REST if lengths is None else lengths
    out: BodyPose = {}
    hips = hips_of(pose)
    for parent, child in BONES:
        if child == "right_hip":  # the hips stay either side of their centre
            out[child] = _add(hips, _sub(hips, out["left_hip"]))
            continue
        base = hips if parent == "hips" else out[parent]
        d = _sub(pose[child], _point(pose, parent))
        n = _norm(d)
        if n < 1e-9:  # degenerate: fall back to the standing direction
            d = _sub(STANDING[child], _point(STANDING, parent))
            n = _norm(d)
        out[child] = _add(base, d, lengths[(parent, child)] / n)
    return out


# --- rotations ---


def rotate(p: Vec, axis: Vec, angle: float, about: Vec = (0.0, 0.0, 0.0)) -> Vec:
    """Rotate p by angle (radians, right-handed in (l, v, f)) about an axis through `about`.

    Axes are given in (f, v, l). About +l, a positive angle tips +v towards +f (so a
    forward raise of a hanging arm is negative); about +f, it tips +v towards -l (so
    raising the left arm out sideways is positive); about +v, it turns +f towards +l.
    """
    # Work in (x, y, z) = (l, v, f), a right-handed frame.
    x, y, z = p[2] - about[2], p[1] - about[1], p[0] - about[0]
    ax, ay, az = axis[2], axis[1], axis[0]
    n = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / n, ay / n, az / n
    c, s = math.cos(angle), math.sin(angle)
    dot = ax * x + ay * y + az * z
    cx, cy, cz = ay * z - az * y, az * x - ax * z, ax * y - ay * x
    rx = x * c + cx * s + ax * dot * (1 - c)
    ry = y * c + cy * s + ay * dot * (1 - c)
    rz = z * c + cz * s + az * dot * (1 - c)
    return (rz + about[0], ry + about[1], rx + about[2])


F_AXIS: Vec = (1.0, 0.0, 0.0)
V_AXIS: Vec = (0.0, 1.0, 0.0)
L_AXIS: Vec = (0.0, 0.0, 1.0)

_CHILDREN: dict[str, list[str]] = {}
for _p, _c in BONES:
    _CHILDREN.setdefault(_p, []).append(_c)


def below(joint: str) -> list[str]:
    """A joint and everything that hangs from it."""
    out, todo = [], [joint]
    while todo:
        j = todo.pop()
        out.append(j)
        todo.extend(_CHILDREN.get(j, []))
    return out


def turn_limb(pose: BodyPose, pivot: str, axis: Vec, angle: float) -> BodyPose:
    """Rotate everything below `pivot` (not the pivot itself) about it."""
    centre = _point(pose, pivot)
    moved = set(below(pivot)) - {pivot}
    return {j: rotate(p, axis, angle, centre) if j in moved else p for j, p in pose.items()}


def lean(pose: BodyPose, axis: Vec, angle: float) -> BodyPose:
    """Tip the upper body (neck and up, arms included) about the hips."""
    centre, moved = hips_of(pose), set(below("neck"))
    return {j: rotate(p, axis, angle, centre) if j in moved else p for j, p in pose.items()}


def tilt(pose: BodyPose, axis: Vec, angle: float, about: Vec = (0.0, 0.0, 0.0)) -> BodyPose:
    """Rotate the whole body about a point (by default the root)."""
    return {j: rotate(p, axis, angle, about) for j, p in pose.items()}


# --- key poses ---

STANDING: BodyPose = _sym(
    head=(0.05, 0.93), eyes=(0.04, 0.95), ears=(-0.01, 0.94), neck=(0.0, 0.84),
    shoulders=(0.0, 0.82), elbows=(-0.01, 0.64), wrists=(0.0, 0.47),
    hips=(0.0, 0.52), knees=(0.01, 0.28), ankles=(0.0, 0.03),
)  # fmt: skip
REST = bone_lengths(STANDING)


LIMB_ENDS = frozenset(f"{side}_{part}" for side in SIDES for part in ("elbow", "wrist", "knee", "ankle"))
LIMB_TOP = 0.2  # body units: how high an elbow, wrist, knee or ankle can be on someone lying down


def lay(upright: BodyPose, *, head: float = 1.0, roll: float = 0.0, thickness: float = 0.06) -> BodyPose:
    """Lay an upright pose on the floor, head towards f = head (+1 or -1).

    roll turns the chest from facing up (0: on the back) towards the body's own left
    (+pi/2: lying on the right side) and down (pi: face down). The result is rooted
    under the hips, with the torso `thickness` above the floor when on the back or front.
    """
    # Tip the body forwards (head to +f, chest down) or backwards (head to -f, chest up),
    # then roll about the body's long axis so the chest faces where it should.
    hips = hips_of(upright)
    centred = {j: _sub(p, (hips[0], 0.0, hips[2])) for j, p in upright.items()}
    laid = tilt(centred, L_AXIS, math.copysign(math.pi / 2, head), (0.0, hips[1], 0.0))
    laid = tilt(laid, F_AXIS, roll + (math.pi if head > 0 else 0.0), (0.0, hips[1], 0.0))
    torso = [laid[j] for j in ("left_hip", "right_hip", "left_shoulder", "right_shoulder")]
    lift = -min(p[1] for p in torso) + 0.02 + (thickness - 0.02) * abs(math.cos(roll))
    # Limbs don't stay up in the air: they flop down to the floor (fit_bones then keeps
    # their lengths, swinging them out along the floor instead).
    out = {j: (p[0], min(max(p[1] + lift, 0.02), LIMB_TOP if j in LIMB_ENDS else math.inf), p[2])
           for j, p in laid.items()}  # fmt: skip
    centre = hips_of(out)
    return {j: (p[0] - centre[0], p[1], p[2] - centre[2]) for j, p in out.items()}


def lying_limbs(rng: random.Random | None, kind: str) -> BodyPose:
    """An upright arrangement of the limbs for someone about to be laid down as `kind`.

    kind: supine (arms by the sides), prone (arms up by the head, as after a trip),
    side (curled: hips and knees bent, arms in front), side_out (on the side, legs
    straighter), sprawl (on the back, limbs out).
    rng varies it; None gives the plain version.
    """

    def u(lo: float, hi: float) -> float:
        return (lo + hi) / 2 if rng is None else rng.uniform(lo, hi)

    p = dict(STANDING)
    if kind == "supine":
        for side, sign in (("left", 1), ("right", -1)):
            p = turn_limb(p, f"{side}_shoulder", F_AXIS, sign * math.radians(u(5, 25)))
            p = turn_limb(p, f"{side}_hip", F_AXIS, sign * math.radians(u(0, 8)))
    elif kind == "prone":
        for side, sign in (("left", 1), ("right", -1)):
            p = turn_limb(p, f"{side}_shoulder", F_AXIS, sign * math.radians(u(70, 150)))
            p = turn_limb(p, f"{side}_elbow", F_AXIS, -sign * math.radians(u(20, 80)))
            p = turn_limb(p, f"{side}_hip", F_AXIS, sign * math.radians(u(2, 15)))
        side = "left" if rng is None or rng.random() < 0.5 else "right"
        p = turn_limb(p, f"{side}_knee", L_AXIS, math.radians(u(0, 50)))  # foot comes up behind
    elif kind in ("side", "side_out"):
        lo = 0.0 if kind == "side_out" else 1.0  # side_out: legs out straighter, as when knocked over
        hip_bend, knee_bend = math.radians(u(25 * lo, 70 if lo else 25)), math.radians(u(30 * lo, 80 if lo else 25))
        for side in SIDES:
            p = turn_limb(p, f"{side}_hip", L_AXIS, -hip_bend * u(0.6, 1.2))
            p = turn_limb(p, f"{side}_knee", L_AXIS, knee_bend * u(0.6, 1.2))
            p = turn_limb(p, f"{side}_shoulder", L_AXIS, -math.radians(u(30, 100)))
            p = turn_limb(p, f"{side}_elbow", L_AXIS, -math.radians(u(10, 70)))
        p = turn_limb(p, "neck", L_AXIS, math.radians(u(5, 25)))
    elif kind == "sprawl":
        for side, sign in (("left", 1), ("right", -1)):
            p = turn_limb(p, f"{side}_shoulder", F_AXIS, sign * math.radians(u(20, 110)))
            p = turn_limb(p, f"{side}_elbow", F_AXIS, sign * math.radians(u(-30, 40)))
            p = turn_limb(p, f"{side}_hip", F_AXIS, sign * math.radians(u(3, 20)))
        side = "left" if rng is None or rng.random() < 0.5 else "right"
        p = turn_limb(p, f"{side}_hip", L_AXIS, -math.radians(u(0, 40)))
        p = turn_limb(p, f"{side}_knee", L_AXIS, math.radians(u(0, 60)))
    else:
        raise ValueError(kind)
    if rng is not None:  # the head turns a little, wherever the body ends up
        p = turn_limb(p, "neck", V_AXIS, math.radians(rng.uniform(-30, 30)))
    return fit_bones(p)


def lying_pose(kind: str, *, head: float = 1.0, roll: float | None = None, rng: random.Random | None = None) -> BodyPose:
    """A body on the floor: kind as lying_limbs; roll defaults to what the kind means."""
    default_roll = {"supine": 0.0, "sprawl": 0.0, "prone": math.pi, "side": math.pi / 2, "side_out": math.pi / 2}[kind]
    return fit_bones(lay(lying_limbs(rng, kind), head=head, roll=default_roll if roll is None else roll))


KEY_POSES: dict[str, BodyPose] = {
    "standing": STANDING,
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
    # Down on both knees, upright.
    "kneel": _sym(
        head=(0.08, 0.71), eyes=(0.08, 0.73), ears=(0.03, 0.72), neck=(0.03, 0.62),
        shoulders=(0.03, 0.60), elbows=(0.04, 0.42), wrists=(0.10, 0.28),
        hips=(0.0, 0.30), knees=(0.08, 0.05), ankles=(-0.17, 0.03),
    ),
    # One knee down, the other foot planted: halfway up off the floor.
    "half_kneel": _sym(
        head=(0.13, 0.72), eyes=(0.13, 0.74), ears=(0.08, 0.73), neck=(0.08, 0.63),
        shoulders=(0.08, 0.61), elbows=(0.14, 0.45), wrists=(0.21, 0.33),
        hips=(0.0, 0.32), knees=(0.0, 0.06), ankles=(-0.25, 0.03),
        right_knee=(0.22, 0.30), right_ankle=(0.22, 0.03),
    ),
    # On hands and knees.
    "hands_knees": _sym(
        head=(0.42, 0.40), eyes=(0.43, 0.41), ears=(0.37, 0.42), neck=(0.32, 0.36),
        shoulders=(0.32, 0.34), elbows=(0.33, 0.17), wrists=(0.34, 0.02),
        hips=(0.0, 0.30), knees=(0.0, 0.05), ankles=(-0.25, 0.03),
    ),
    # Squatting on the way down to a seat that isn't there.
    "crouch": _sym(
        head=(0.16, 0.70), eyes=(0.16, 0.72), ears=(0.11, 0.71), neck=(0.10, 0.62),
        shoulders=(0.10, 0.60), elbows=(0.18, 0.45), wrists=(0.28, 0.40),
        hips=(-0.12, 0.32), knees=(0.14, 0.27), ankles=(0.02, 0.03),
    ),
    # Sitting on the floor, legs out in front, propped up on the hands.
    "floor_sit": _sym(
        head=(-0.04, 0.47), eyes=(-0.03, 0.49), ears=(-0.08, 0.48), neck=(-0.08, 0.39),
        shoulders=(-0.08, 0.37), elbows=(-0.14, 0.21), wrists=(-0.18, 0.04),
        hips=(0.0, 0.08), knees=(0.22, 0.14), ankles=(0.45, 0.03),
    ),
    # Rising out of a chair, weight forward over the feet.
    "half_stand": _sym(
        head=(0.24, 0.74), eyes=(0.24, 0.76), ears=(0.19, 0.75), neck=(0.18, 0.66),
        shoulders=(0.18, 0.64), elbows=(0.12, 0.48), wrists=(0.18, 0.38),
        hips=(-0.02, 0.40), knees=(0.18, 0.29), ankles=(0.12, 0.03),
    ),
    # Tripped: pitched forward over a planted foot, arms coming up.
    "stumble": _sym(
        head=(0.21, 0.87), eyes=(0.21, 0.89), ears=(0.15, 0.89), neck=(0.14, 0.80),
        shoulders=(0.14, 0.78), elbows=(0.28, 0.70), wrists=(0.40, 0.68),
        hips=(0.0, 0.50), knees=(0.12, 0.28), ankles=(0.20, 0.04),
        right_knee=(-0.08, 0.29), right_ankle=(-0.20, 0.06),
    ),
    # Going down forwards, arms out to break the fall.
    "lunge": _sym(
        head=(0.38, 0.66), eyes=(0.39, 0.68), ears=(0.33, 0.69), neck=(0.30, 0.62),
        shoulders=(0.30, 0.60), elbows=(0.42, 0.46), wrists=(0.50, 0.30),
        hips=(0.05, 0.40), knees=(0.18, 0.20), ankles=(0.18, 0.03),
        right_knee=(-0.10, 0.25), right_ankle=(-0.25, 0.12),
    ),
    # Knees giving way, head dropping.
    "buckle": _sym(
        head=(0.12, 0.77), eyes=(0.12, 0.78), ears=(0.07, 0.79), neck=(0.06, 0.73),
        shoulders=(0.06, 0.71), elbows=(0.05, 0.54), wrists=(0.08, 0.38),
        hips=(-0.03, 0.42), knees=(0.10, 0.24), ankles=(0.0, 0.03),
    ),
    # Losing balance backwards, arms thrown forward.
    "lean_back": _sym(
        head=(-0.15, 0.90), eyes=(-0.14, 0.92), ears=(-0.19, 0.90), neck=(-0.16, 0.80),
        shoulders=(-0.16, 0.78), elbows=(0.0, 0.70), wrists=(0.12, 0.72),
        hips=(-0.03, 0.50), knees=(0.03, 0.27), ankles=(0.06, 0.03),
    ),
    # On the back, arms by the sides: lying down on purpose, which means lying back, so the
    # head ends up behind where the person was facing.
    "lying": lying_pose("supine", head=-1.0),
    "supine": lying_pose("sprawl", head=-1.0),
    "prone": lying_pose("prone"),
    "side_lying": lying_pose("side"),
}
KEY_POSES["fallen"] = KEY_POSES["side_lying"]  # the name hand-written scenarios use
KEY_POSES = {name: fit_bones(p) for name, p in KEY_POSES.items()}


# --- walking ---

GAIT_SWING = 0.13  # ankle swing either side of the hips, body units, at gait amount 1


def stride_length(speed: float, size: float) -> float:
    """Metres covered by one full gait cycle (two steps) at a walking speed (m/s)."""
    return size * min(max(0.25 + 0.45 * speed, 0.45), 0.8)


def gait_amount(stride: float, size: float) -> float:
    """How big the swing must be for the planted foot to keep pace with the ground."""
    return stride / (2.0 * math.pi * size * GAIT_SWING)


def body_pose(name: str, gait_phase: float | None = None, amount: float = 1.0) -> BodyPose:
    """Key pose in body units, with a walking cycle layered on when gait_phase is given."""
    pose = dict(KEY_POSES[name])
    if gait_phase is None or amount <= 0.0:
        return pose

    s, c, a = math.sin(gait_phase), math.cos(gait_phase), amount
    bob = 0.012 * (abs(c) - 0.5)

    def shift(joint: str, df: float = 0.0, dv: float = 0.0) -> None:
        f, v, l = pose[joint]
        pose[joint] = (f + df * a, v + dv * min(a, 1.0), l)

    for joint in pose:
        shift(joint, dv=bob)
    for joint in ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "neck"):
        shift(joint, df=0.02)
    shift("left_ankle", df=GAIT_SWING * s, dv=0.04 * max(0.0, c))
    shift("right_ankle", df=-GAIT_SWING * s, dv=0.04 * max(0.0, -c))
    shift("left_knee", df=0.07 * s, dv=0.02 * max(0.0, c))
    shift("right_knee", df=-0.07 * s, dv=0.02 * max(0.0, -c))
    shift("left_wrist", df=-0.09 * s)
    shift("right_wrist", df=0.09 * s)
    shift("left_elbow", df=-0.04 * s)
    shift("right_elbow", df=0.04 * s)
    return fit_bones(pose)


def _slerp(a: Vec, b: Vec, alpha: float) -> Vec:
    """Turn unit vector a part way towards unit vector b, at a steady rate."""
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
    angle = math.acos(dot)
    if angle < 1e-6:
        return a
    if math.pi - angle < 1e-3:  # opposite: turn about any axis square to it
        ref = (0.0, 1.0, 0.0) if abs(a[1]) < 0.9 else (1.0, 0.0, 0.0)
        axis = (a[1] * ref[2] - a[2] * ref[1], a[2] * ref[0] - a[0] * ref[2], a[0] * ref[1] - a[1] * ref[0])
        return rotate(a, axis, angle * alpha)
    k1, k2 = math.sin((1 - alpha) * angle) / math.sin(angle), math.sin(alpha * angle) / math.sin(angle)
    return (a[0] * k1 + b[0] * k2, a[1] * k1 + b[1] * k2, a[2] * k1 + b[2] * k2)


def _unit(a: Vec) -> Vec:
    n = _norm(a)
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-9 else (0.0, 1.0, 0.0)


FLOOR_V = 0.015  # body units: the lowest a joint centre goes


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _knee(hip: Vec, ankle: Vec, pole: Vec, thigh: float, shin: float) -> tuple[Vec, Vec]:
    """(knee, ankle) for a leg from hip reaching for ankle, bending towards pole (two-bone IK).

    An ankle out of reach is pulled in along the line from the hip.
    """
    reach = _sub(ankle, hip)
    dist = _norm(reach)
    u = _unit(reach) if dist > 1e-9 else (0.0, -1.0, 0.0)
    longest = thigh + shin - 1e-4
    dist = min(max(dist, abs(thigh - shin) + 1e-4), longest)
    w = _sub(pole, (u[0] * _dot(pole, u), u[1] * _dot(pole, u), u[2] * _dot(pole, u)))
    # With the pole along the leg there is no telling which way the knee goes; straighten
    # the leg as that happens, so the knee can't visibly flip from one side to the other.
    dist += (longest - dist) * (1.0 - smoothstep(_norm(w) / 0.35))
    ankle = _add(hip, u, dist)
    if _norm(w) < 1e-9:
        w = (1.0, 0.0, 0.0) if abs(u[0]) < 0.9 else (0.0, 1.0, 0.0)
        w = _sub(w, (u[0] * _dot(w, u), u[1] * _dot(w, u), u[2] * _dot(w, u)))
    w = _unit(w)
    along = (thigh * thigh - shin * shin + dist * dist) / (2 * dist)
    up = math.sqrt(max(thigh * thigh - along * along, 0.0))
    return _add(_add(hip, u, along), w, up), ankle


def _planted(a: Vec, b: Vec) -> bool:
    """Is a foot on the floor in both poses, and in much the same place?"""
    return a[1] < 0.08 and b[1] < 0.08 and math.hypot(b[0] - a[0], b[2] - a[2]) < 0.35


def blend_body(a: BodyPose, b: BodyPose, alpha: float) -> BodyPose:
    """Part way from pose a to pose b.

    The hips move straight; every bone of the trunk, arms and head turns at a steady rate
    from its direction in a to its direction in b, keeping its standing length. Feet move
    straight too (so they stay planted while someone sits down) and the knees bend to
    match.
    """
    if alpha <= 0.0:
        return a
    if alpha >= 1.0:
        return b
    ha, hb = hips_of(a), hips_of(b)
    hips = _add(ha, _sub(hb, ha), alpha)
    out: BodyPose = {}
    for parent, child in BONES:
        if child == "right_hip":  # the hips stay either side of their centre
            out[child] = _add(hips, _sub(hips, out["left_hip"]))
            continue
        if child.endswith(("_knee", "_ankle")):
            continue
        base = hips if parent == "hips" else out[parent]
        da = _unit(_sub(a[child], _point(a, parent)))
        db = _unit(_sub(b[child], _point(b, parent)))
        x, v, l = _add(base, _slerp(da, db, alpha), REST[(parent, child)])
        out[child] = (x, max(v, FLOOR_V), l)  # an arm turning through the floor stops at it
    for side in SIDES:
        hip, knee, ankle = f"{side}_hip", f"{side}_knee", f"{side}_ankle"
        thigh, shin = REST[(hip, knee)], REST[(knee, ankle)]
        # The knee goes the way the thigh would point if it simply turned from a to b, and
        # the foot follows the shin the same way...
        pole = _slerp(_unit(_sub(a[knee], a[hip])), _unit(_sub(b[knee], b[hip])), alpha)
        swung = _add(_add(out[hip], pole, thigh), _slerp(_unit(_sub(a[ankle], a[knee])), _unit(_sub(b[ankle], b[knee])), alpha), shin)
        # ...unless it is planted on the floor in both poses: then it stays planted.
        if _planted(a[ankle], b[ankle]):
            target = _add(a[ankle], _sub(b[ankle], a[ankle]), alpha)
        else:
            target = swung
        target = (target[0], max(target[1], FLOOR_V), target[2])
        out[knee], out[ankle] = _knee(out[hip], target, pole, thigh, shin)
    return out


def is_pose(name: str) -> bool:
    return name in KEY_POSES


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def ease_in(x: float) -> float:
    """Accelerating, like a body under gravity."""
    x = min(max(x, 0.0), 1.0)
    return x * x


def ease_out(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1.0 - (1.0 - x) * (1.0 - x)
