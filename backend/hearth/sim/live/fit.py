"""How a posed body sits against furniture: for tests, and for checking new poses and slots.

Keypoints are joint centres, but a body has volume: the 3D view draws a mannequin round
them, and its seat is a hand's breadth below the hip joints. Here the body is a chain of
spheres along its bones (radii in body units, so they scale with height), and a piece is
its collision_boxes(). penetration() is how far the worst sphere goes into a solid part;
a body resting on a seat just touches it (about 0).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from hearth.sim.pose import KEY_POSES, BodyPose, curled_lying, long_sit, supine
from hearth.sim.world import WorldPose, body

from hearth.sim.live.room import Kind, Piece, collision_boxes

Vec3 = tuple[float, float, float]

RADIUS: dict[str, float] = {
    "head": 0.055, "neck": 0.04, "shoulder": 0.045, "elbow": 0.03, "wrist": 0.025,
    "hips": 0.06, "hip": 0.055, "knee": 0.035, "ankle": 0.03,
}  # fmt: skip
"""Body units: half the thickness of the body at each point."""

HIP_SEAT_GAP = RADIUS["hip"]  # = pose.HIP_DEPTH
"""Body units from a hip joint down to what it sits on."""

SEGMENTS: tuple[tuple[str, str], ...] = (
    ("hips", "neck"), ("neck", "head"), ("left_hip", "right_hip"), ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)  # fmt: skip
STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _radius(point: str) -> float:
    return RADIUS[point.removeprefix("left_").removeprefix("right_")]


def _points(joints: WorldPose) -> dict[str, Vec3]:
    pts = dict(joints)
    lh, rh = joints["left_hip"], joints["right_hip"]
    pts["hips"] = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2, (lh[2] + rh[2]) / 2)
    le, re = joints["left_ear"], joints["right_ear"]
    pts["head"] = ((le[0] + re[0]) / 2, (le[1] + re[1]) / 2, (le[2] + re[2]) / 2)
    return pts


def spheres(joints: WorldPose, size: float, build: float = 1.0) -> list[tuple[str, Vec3, float]]:
    """(where, centre, radius) along every bone. `build` widens the body as the 3D view does."""
    pts = _points(joints)
    out = []
    for a, b in SEGMENTS:
        pa, pb = pts[a], pts[b]
        ra, rb = _radius(a) * size * build, _radius(b) * size * build
        for t in STEPS:
            c = (pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t, pa[2] + (pb[2] - pa[2]) * t)
            out.append((a if t < 0.5 else b, c, ra + (rb - ra) * t))
    return out


@dataclass(frozen=True)
class Overlap:
    depth: float  # metres the worst sphere goes into a part (negative: clear by that much)
    where: str  # body point
    part: str  # furniture part


def penetration(joints: WorldPose, piece: Piece, size: float, build: float = 1.0) -> Overlap:
    """How far the body goes into the piece's solid parts, at worst."""
    worst = Overlap(-1e9, "", "")
    boxes = collision_boxes(piece)
    for where, c, r in spheres(joints, size, build):
        local = piece.to_local(c)
        for box in boxes:
            d = box.depth(local) + r
            if d > worst.depth:
                worst = Overlap(d, where, box.part)
    return worst


def lowest(joints: WorldPose, size: float) -> float:
    """Height of the underside of the body: its lowest sphere's bottom."""
    return min(c[1] - r for _, c, r in spheres(joints, size))


# --- lying on things ---

LIE_MARGIN_M = 0.03  # kept clear at each end and side of the space a body lies in
BENDS = tuple(round(0.2 + 0.1 * i, 2) for i in range(9))  # 0.2 .. 1.0


@dataclass(frozen=True)
class LieFit:
    """How someone lies on a kind of piece, in its own axes (as Box): the pose, where its root
    goes (lx, lz), the heading in the piece's frame, and the base (metres up)."""

    pose: BodyPose
    root: tuple[float, float]
    heading: float
    base: float
    bend: float  # 0 = legs straight; else knees drawn up (on the back) or curled (on the side) this much
    sit_root: tuple[float, float]  # where sit_pose goes: sitting up, centred on the piece
    sit_base: float

    @property
    def sit_pose(self) -> BodyPose:
        """Half way: sitting up with the legs along the piece, as bent as when lying."""
        return long_sit(self.bend)


def _space(kind: Kind) -> tuple[float, float, float, float]:
    """The free top of a piece, in its own axes (lx0, lx1, lz0, lz1): between the arms, in
    front of the backrest or headboard."""
    w, d = kind.w / 2, kind.d / 2
    inner = w - kind.arm
    return (-inner, inner, -d + kind.back, d)


def _extent(joints: WorldPose, size: float) -> tuple[float, float, float, float, float]:
    sph = spheres(joints, size)
    return (min(c[0] - r for _, c, r in sph), max(c[0] + r for _, c, r in sph),
            min(c[2] - r for _, c, r in sph), max(c[2] + r for _, c, r in sph),
            min(c[1] - r for _, c, r in sph))  # fmt: skip


@functools.cache
def _candidates() -> list[tuple[float, BodyPose]]:
    """Ways to lie, straightest and most spread out first."""
    out = [(0.0, KEY_POSES["lying"]), (0.0, supine())]
    out += [(bend, supine(bend)) for bend in BENDS]
    for bend in BENDS:
        for roll in (math.pi / 2, -math.pi / 2):
            out.append((bend, curled_lying(bend, roll)))
    return out


@functools.lru_cache(maxsize=256)
def _lie_fit(kind: Kind, heading: float, size: float) -> LieFit | None:
    x0, x1, z0, z1 = _space(kind)
    piece = Piece(0, kind, (0.0, 0.0), 0.0)
    for bend, pose in _candidates():
        joints = body(pose, heading, 0.0, 0.0, 0.0, size)
        ex0, ex1, ez0, ez1, ey = _extent(joints, size)
        if ex1 - ex0 > x1 - x0 - 2 * LIE_MARGIN_M or ez1 - ez0 > z1 - z0 - 2 * LIE_MARGIN_M:
            continue
        root = ((x0 + x1) / 2 - (ex0 + ex1) / 2, (z0 + z1) / 2 - (ez0 + ez1) / 2)
        base = kind.surface - ey + 0.003  # resting on the surface, not sunk into it
        placed = body(pose, heading, root[0], root[1], base, size)
        if penetration(placed, piece, size).depth > 0.01:
            continue
        # Sitting up on the way, legs along the piece: centred too (a curl lies further back).
        sit = long_sit(bend)
        sx0, sx1, sz0, sz1, sy = _extent(body(sit, heading, 0.0, 0.0, 0.0, size), size)
        sit_root = ((x0 + x1) / 2 - (sx0 + sx1) / 2, (z0 + z1) / 2 - (sz0 + sz1) / 2)
        return LieFit(pose, (round(root[0], 4), round(root[1], 4)), heading, round(base, 4), bend,
                      (round(sit_root[0], 4), round(sit_root[1], 4)), round(kind.surface - sy + 0.003, 4))  # fmt: skip
    return None


def lie_fit(kind: Kind, heading: float, size: float) -> LieFit | None:
    """The straightest way someone `size` metres tall lies on a kind of piece, head to feet
    along `heading` (in the piece's frame), centred in the space between its arms or ends.
    None if they can't fit even curled up."""
    h = (heading + math.pi) % (2 * math.pi) - math.pi
    return _lie_fit(kind, round(h, 4), round(size, 3))


def seat_gap(joints: WorldPose, size: float, seat: float) -> float:
    """Metres between the bottom of the hips (the lower hip, lying on one side) and a seat
    or lying surface at height `seat`; negative = sunk into it."""
    hip = min(joints["left_hip"][1], joints["right_hip"][1])
    return hip - RADIUS["hip"] * size - seat
