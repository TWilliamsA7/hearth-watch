"""How a posed body sits against furniture: for tests, and for checking new poses and slots.

Keypoints are joint centres, but a body has volume: the 3D view draws a mannequin round
them, and its seat is a hand's breadth below the hip joints. Here the body is a chain of
spheres along its bones (radii in body units, so they scale with height), and a piece is
its collision_boxes(). penetration() is how far the worst sphere goes into a solid part;
a body resting on a seat just touches it (about 0).
"""

from __future__ import annotations

from dataclasses import dataclass

from hearth.sim.world import WorldPose

from hearth.sim.live.room import Piece, collision_boxes

Vec3 = tuple[float, float, float]

RADIUS: dict[str, float] = {
    "head": 0.055, "neck": 0.04, "shoulder": 0.045, "elbow": 0.03, "wrist": 0.025,
    "hips": 0.06, "hip": 0.055, "knee": 0.035, "ankle": 0.03,
}  # fmt: skip
"""Body units: half the thickness of the body at each point."""

HIP_SEAT_GAP = RADIUS["hip"]
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


def seat_gap(joints: WorldPose, size: float, seat: float) -> float:
    """Metres between the bottom of the hips and a seat (or lying surface) at height `seat`;
    negative = sunk into it."""
    hips = _points(joints)["hips"]
    return hips[1] - RADIUS["hip"] * size - seat
