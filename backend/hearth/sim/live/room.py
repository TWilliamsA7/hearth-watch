"""Procedural rooms in metres: walls, doors, a window and furniture people can use.

Coordinates are the world track's (hearth.sim.world): x to the right, y up, z towards
the front. The room's floor spans x in [-width/2, width/2] and z in [-depth, 0]; the
front wall (z = 0) is the camera side. Headings follow world.HEADING: 0 faces +z,
pi/2 faces +x; a piece of furniture's `rotation` is the heading its front faces.

Furniture is functional. Each piece carries slots: where a person's root goes to sit
or lie on it, which way they face, and a group (a couch's seats and its lying slot
share one, so someone lying on it blocks the seats). Where a person stands before
sitting is worked out per person (stand_point), so that their feet stay planted as they
lower themselves (pose.blend_body moves planted feet in step with the root).

The generator places pieces one at a time and keeps a piece only if every slot and door
can still be reached from the first door (a flood fill on the room's NavGrid), so a
generated room never traps anyone. Same spec and seed, same room.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Literal

from hearth.sim.nav import BODY_RADIUS_M, Floor, Footprint, NavGrid, distance
from hearth.sim.pose import seated_pose
from hearth.sim.live.spec import RoomSpec

Wall = Literal["back", "left", "right", "front"]
SlotKind = Literal["sit", "lie"]

WALL_HEADING: dict[str, float] = {"back": 0.0, "left": math.pi / 2, "right": -math.pi / 2, "front": math.pi}
"""Which way something against each wall faces (into the room)."""

APPROACH_GAP_M = 0.08  # approach points sit this far outside a piece's grown footprint
WALKWAY_M = 0.55  # pieces keep this much floor between them, unless grouped (chairs at a table)
DOOR_WIDTH_M = 0.95
DOOR_KEEPOUT_M = 1.2  # nothing stands this close in front of a door
SIT_FEET_F = 0.20  # body units: how far in front of the root the feet are in pose "sitting"


@dataclass(frozen=True)
class Kind:
    """A kind of furniture: size (local width x, depth z, height) and what it's for.

    The parts people touch are measured here, and only here: the 3D view draws them from
    these numbers (WorldObject.seat_height...) and bodies are fitted to them, so what's
    drawn and where people sit can't drift apart. Heights are metres above the floor;
    depths are from the back of the piece.
    """

    name: str
    w: float
    d: float
    h: float
    walkable: bool = False  # a rug
    view_blocking: bool = False  # tall enough to hide people from the camera
    seat: float = 0.0  # top of the seat (what someone sitting rests on); 0 = not for sitting
    surface: float = 0.0  # top of what someone lies on; 0 = not for lying
    arm: float = 0.0  # width of each arm
    arm_h: float = 0.0  # top of the arms
    back: float = 0.0  # how far the backrest (or headboard) comes forward from the back


KINDS: dict[str, Kind] = {
    k.name: k
    for k in (
        Kind("chair", 0.48, 0.5, 0.9, seat=0.485, back=0.05),
        Kind("armchair", 0.92, 0.85, 0.95, seat=0.43, arm=0.09, arm_h=0.62, back=0.31),
        Kind("couch", 2.0, 0.9, 0.85, seat=0.43, surface=0.43, arm=0.16, arm_h=0.62, back=0.31),
        Kind("loveseat", 1.5, 0.88, 0.85, seat=0.43, surface=0.43, arm=0.16, arm_h=0.62, back=0.31),
        Kind("bed", 1.05, 2.05, 0.55, seat=0.565, surface=0.565, back=0.06),
        Kind("table", 1.4, 0.85, 0.75),
        Kind("side_table", 0.5, 0.5, 0.55),
        Kind("cabinet", 1.4, 0.45, 0.65),
        Kind("bookshelf", 0.9, 0.35, 1.85, view_blocking=True),
        Kind("plant", 0.5, 0.5, 1.3, view_blocking=True),
        Kind("rug", 2.2, 1.5, 0.01, walkable=True),
    )
}

LEGACY_KIND: dict[str, str] = {
    "chair": "chair", "armchair": "chair", "couch": "couch", "loveseat": "couch", "bed": "bed",
    "rug": "mat", "table": "occluder", "side_table": "occluder", "cabinet": "occluder",
    "bookshelf": "occluder", "plant": "occluder",
}  # fmt: skip
"""The nearest of the scripted pipeline's ObjectKinds (WorldObject.kind); the full kind is WorldObject.variant."""

HEADBOARD_H_M = 0.95  # top of a bed's headboard


@dataclass(frozen=True)
class Box:
    """A solid part of a piece, in its own axes: lx towards its left, lz towards its front, y up."""

    part: str
    lx: tuple[float, float]
    y: tuple[float, float]
    lz: tuple[float, float]

    def depth(self, p: tuple[float, float, float]) -> float:
        """How far a local point is inside (positive) or outside (negative)."""
        return min(p[0] - self.lx[0], self.lx[1] - p[0], p[1] - self.y[0], self.y[1] - p[1],
                   p[2] - self.lz[0], self.lz[1] - p[2])  # fmt: skip


def collision_boxes(piece: Piece) -> list[Box]:
    """The solid parts of a piece that bodies must stay out of, from the same Kind numbers the
    3D view draws. Soft parts (cushion tops, pillows, blankets) are the boxes' tops."""
    k = piece.kind
    w, d = k.w / 2, k.d / 2
    if k.walkable:
        return []
    if k.name == "chair":
        return [Box("seat", (-w, w), (0.0, k.seat), (-d, d)), Box("back", (-w, w), (k.seat, k.h), (-d, -d + k.back))]
    if k.name in ("armchair", "couch", "loveseat"):
        inner = w - k.arm
        return [
            Box("seat", (-inner, inner), (0.0, k.seat), (-d, d)),
            Box("back", (-inner, inner), (k.seat, k.h), (-d, -d + k.back)),
            Box("arm", (-w, -inner), (0.0, k.arm_h), (-d, d)),
            Box("arm", (inner, w), (0.0, k.arm_h), (-d, d)),
        ]
    if k.name == "bed":
        return [Box("mattress", (-w, w), (0.0, k.surface), (-d, d)),
                Box("headboard", (-w, w), (k.surface, HEADBOARD_H_M), (-d, -d + k.back))]  # fmt: skip
    return [Box("body", (-w, w), (0.0, k.h), (-d, d))]


def forward(heading: float) -> Floor:
    return (math.sin(heading), math.cos(heading))


def _add(p: Floor, d: Floor, k: float = 1.0) -> Floor:
    return (p[0] + d[0] * k, p[1] + d[1] * k)


def _r(p: Floor) -> Floor:
    return (round(p[0], 3), round(p[1], 3))


@dataclass
class Piece:
    index: int
    kind: Kind
    at: Floor  # centre of the footprint
    rotation: float  # heading its front faces
    group: int | None = None  # pieces placed together (a table and its chairs)

    @property
    def size(self) -> tuple[float, float]:
        """World (x, z) extent: width and depth swap when turned sideways."""
        side = abs(math.sin(self.rotation)) > 0.5
        return (self.kind.d, self.kind.w) if side else (self.kind.w, self.kind.d)

    def footprint(self, grow: float = 0.0) -> Footprint:
        sx, sz = self.size
        return (self.at[0] - sx / 2 - grow, self.at[1] - sz / 2 - grow,
                self.at[0] + sx / 2 + grow, self.at[1] + sz / 2 + grow)  # fmt: skip

    def local(self, lx: float, lz: float) -> Floor:
        """A point in the piece's own axes: lx towards its left (as world.body's l axis), lz towards its front."""
        fx, fz = forward(self.rotation)
        rx, rz = fz, -fx  # the piece's own left, as world.body's "l" axis
        return (self.at[0] + rx * lx + fx * lz, self.at[1] + rz * lx + fz * lz)

    def to_local(self, p: tuple[float, float, float]) -> tuple[float, float, float]:
        """A world point (x, y, z) in the piece's own axes (lx, y, lz), as Box uses."""
        fx, fz = forward(self.rotation)
        dx, dz = p[0] - self.at[0], p[2] - self.at[1]
        return (dx * fz - dz * fx, p[1], dx * fx + dz * fz)


@dataclass
class Slot:
    slot_id: str
    piece: int
    kind: SlotKind
    at: Floor  # where the person's root goes
    heading: float  # sitting: the way they face; lying: from head to feet
    base: float  # metres above the floor of the surface the pose rests on
    group: str  # slots in one group can't be used at once
    entry: str | None = None  # lie slots: the sit slot to sit on first
    occupant: str | None = None  # person_id sitting or lying here, or on the way to
    seat: float = 0.0  # sit slots: metres up to the top of the seat
    front: float = 0.0  # sit slots: metres from `at` forward to the seat's front edge

    def pose(self, size: float):
        """How someone `size` metres tall sits here (pose.seated_pose)."""
        return seated_pose(self.seat, self.front, size)

    def stand_point(self, size: float) -> Floor:
        """Where someone `size` metres tall stands, facing away, just before sitting here: where
        their feet will be once seated, so they stay planted as they sit down."""
        pose = self.pose(size)
        feet = (pose["left_ankle"][0] + pose["right_ankle"][0]) / 2
        return _add(self.at, forward(self.heading), max(feet, SIT_FEET_F) * size)


@dataclass(frozen=True)
class Door:
    door_id: str
    wall: Wall
    at: Floor  # centre of the opening, on the wall line
    inside: Floor  # a walkable point just inside
    outside: Floor  # where someone stands out of the room before coming in
    heading_in: float  # facing into the room
    width: float = DOOR_WIDTH_M


@dataclass(frozen=True)
class Window:
    wall: Wall
    at: Floor  # centre, on the wall line
    width: float
    sill: float
    height: float


@dataclass(frozen=True)
class Place:
    """Somewhere to stand on purpose (no furniture used), and which way to face there."""

    name: str  # window, browse, middle, door
    at: Floor
    heading: float
    piece: int | None = None


@dataclass
class Room:
    width: float
    depth: float
    height: float
    pieces: list[Piece]
    slots: list[Slot]
    doors: list[Door]
    windows: list[Window]
    places: list[Place]
    nav: NavGrid
    seed: int
    radius: float = BODY_RADIUS_M
    _slot_index: dict[str, Slot] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._slot_index = {s.slot_id: s for s in self.slots}

    def slot(self, slot_id: str) -> Slot:
        return self._slot_index[slot_id]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (-self.width / 2, -self.depth, self.width / 2, 0.0)

    def group_busy(self, group: str, *, besides: str | None = None) -> bool:
        return any(s.group == group and s.occupant is not None and s.occupant != besides for s in self.slots)

    def free_slots(self, kind: SlotKind, person_id: str | None = None) -> list[Slot]:
        """Slots nobody is in or headed for; a lie slot needs its whole group free."""
        out = []
        for s in self.slots:
            if s.kind != kind or s.occupant is not None:
                continue
            if kind == "lie" and self.group_busy(s.group, besides=person_id):
                continue
            if kind == "sit" and any(o.kind == "lie" and o.group == s.group and o.occupant for o in self.slots):
                continue
            out.append(s)
        return out

    def raw_free(self, p: Floor, ignore: set[int] = frozenset(), margin: float = 0.12) -> bool:  # type: ignore[assignment]
        """Is p clear of every piece's real footprint (not grown by a body radius)?"""
        x0, z0, x1, z1 = self.bounds
        if not (x0 + margin <= p[0] <= x1 - margin and z0 + margin <= p[1] <= z1 - margin):
            return False
        for piece in self.pieces:
            if piece.index in ignore or piece.kind.walkable:
                continue
            f = piece.footprint(margin)
            if f[0] <= p[0] <= f[2] and f[1] <= p[1] <= f[3]:
                return False
        return True

    def step_clear(self, a: Floor, b: Floor, ignore: set[int]) -> bool:
        steps = max(1, int(distance(a, b) / 0.04))
        return all(self.raw_free((a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps), ignore, 0.1)
                   for k in range(steps + 1))  # fmt: skip

    def approach(self, slot: Slot, size: float) -> Floor | None:
        """A point on open floor from which a short straight step reaches stand_point."""
        piece = self.pieces[slot.piece]
        stand = slot.stand_point(size)
        own = {piece.index} | {p.index for p in self.pieces if piece.group is not None and p.group == piece.group}
        fwd = forward(slot.heading)
        left = (fwd[1], -fwd[0])
        reach = max(piece.kind.d, piece.kind.w) / 2 + self.radius + APPROACH_GAP_M
        for ahead, aside in ((1.0, 0.0), (0.7, 0.7), (0.7, -0.7), (0.2, 1.0), (0.2, -1.0)):
            for k in (0.6, 0.9, 1.2):
                p = _add(_add(slot.at, fwd, ahead * reach * k + 0.15), left, aside * reach * k)
                if self.nav.free(p) and self.step_clear(p, stand, own):
                    return _r(p)
        return None

    def door(self, door_id: str) -> Door:
        return next(d for d in self.doors if d.door_id == door_id)


# --- generation ---


def generate_room(spec: RoomSpec, seed: int) -> Room:
    rng = random.Random(f"room/{seed}")
    width = spec.width if spec.width is not None else round(rng.uniform(5.0, 8.5), 1)
    depth = spec.depth if spec.depth is not None else round(rng.uniform(4.5, 7.5), 1)
    doors = _doors(rng, width, depth, spec.doors)
    windows = _windows(rng, width, depth) if spec.window else []
    builder = _Builder(rng, width, depth, doors)
    builder.furnish(spec.furniture_density)
    pieces = builder.pieces
    nav = builder.nav()
    room = Room(width=width, depth=depth, height=spec.height, pieces=pieces, slots=_slots(pieces), doors=doors,
                windows=windows, places=[], nav=nav, seed=seed)  # fmt: skip
    # Drop slots nobody can get to (a seat boxed in by its neighbours).
    reach = nav.reachable(doors[0].inside)
    room.slots = [s for s in room.slots if (a := room.approach(s, 1.7)) is not None and nav._cell(a) in reach]
    room.__post_init__()
    room.places = _places(room, reach)
    return room


def _doors(rng: random.Random, width: float, depth: float, count: int) -> list[Door]:
    walls: list[Wall] = ["left", "right", "back", "front"]
    rng.shuffle(walls)
    # The camera hangs in a front corner: keep at least one door off the front wall.
    walls.sort(key=lambda w: w == "front")
    out: list[Door] = []
    for i, wall in enumerate(walls[:count]):
        length = depth if wall in ("left", "right") else width
        offset = rng.uniform(0.8, length - 0.8)  # from the wall's start, away from corners
        if wall == "left":
            at, n = (-width / 2, -depth + offset), (1.0, 0.0)
        elif wall == "right":
            at, n = (width / 2, -depth + offset), (-1.0, 0.0)
        elif wall == "back":
            at, n = (-width / 2 + offset, -depth), (0.0, 1.0)
        else:
            at, n = (-width / 2 + offset, 0.0), (0.0, -1.0)
        inside = _add(at, n, BODY_RADIUS_M + 0.25)
        outside = _add(at, n, -0.7)
        out.append(Door(f"door-{i + 1}", wall, _r(at), _r(inside), _r(outside), math.atan2(n[0], n[1])))
    return out


def _windows(rng: random.Random, width: float, depth: float) -> list[Window]:
    w = min(1.8, width * 0.35)
    x = rng.uniform(-width / 2 + w / 2 + 0.4, width / 2 - w / 2 - 0.4)
    return [Window("back", (round(x, 3), -depth), round(w, 2), 0.9, 1.3)]


class _Builder:
    def __init__(self, rng: random.Random, width: float, depth: float, doors: list[Door]) -> None:
        self.rng = rng
        self.width, self.depth = width, depth
        self.doors = doors
        self.pieces: list[Piece] = []
        self._groups = 0

    # The room as it stands, for checking a placement.

    def nav(self, pieces: list[Piece] | None = None) -> NavGrid:
        r = BODY_RADIUS_M
        pieces = self.pieces if pieces is None else pieces
        return NavGrid(-self.width / 2 + r, self.width / 2 - r, -self.depth + r, -r,
                       [None if p.kind.walkable else p.footprint(r) for p in pieces])  # fmt: skip

    def _fits(self, piece: Piece, others: list[Piece]) -> bool:
        x0, z0, x1, z1 = piece.footprint()
        if x0 < -self.width / 2 - 1e-6 or x1 > self.width / 2 + 1e-6 or z0 < -self.depth - 1e-6 or z1 > 1e-6:
            return False
        for d in self.doors:  # keep doorways clear
            dx0, dz0 = d.at[0] - DOOR_WIDTH_M / 2 - 0.1, d.at[1] - DOOR_WIDTH_M / 2 - 0.1
            reach = _add(d.at, forward(d.heading_in), DOOR_KEEPOUT_M)
            kx0, kz0 = min(dx0, reach[0] - DOOR_WIDTH_M / 2), min(dz0, reach[1] - DOOR_WIDTH_M / 2)
            kx1, kz1 = max(d.at[0], reach[0]) + DOOR_WIDTH_M / 2, max(d.at[1], reach[1]) + DOOR_WIDTH_M / 2
            if not piece.kind.walkable and x0 < kx1 and x1 > kx0 and z0 < kz1 and z1 > kz0:
                return False
        for o in others:
            if piece.kind.walkable or o.kind.walkable:
                if piece.kind.walkable and o.kind.walkable and _overlap(piece.footprint(), o.footprint()):
                    return False
                continue
            gap = 0.04 if piece.group is not None and piece.group == o.group else WALKWAY_M
            if _overlap(piece.footprint(gap / 2), o.footprint(gap / 2)):
                return False
        return True

    def _connected(self, pieces: list[Piece]) -> bool:
        """Every door and every other piece's surroundings still reachable from the first door."""
        nav = self.nav(pieces)
        reach = nav.reachable(self.doors[0].inside)
        if any(nav._cell(d.inside) not in reach for d in self.doors):
            return False
        for p in pieces:
            if p.kind.walkable:
                continue
            f = p.footprint(BODY_RADIUS_M + 0.12)
            sides = [(p.at[0], f[1]), (p.at[0], f[3]), (f[0], p.at[1]), (f[2], p.at[1])]
            if not any(nav.free(s) and nav._cell(s) in reach for s in sides):
                return False
        return True

    def _try(self, pieces: list[Piece]) -> bool:
        """Add pieces (one group) if they fit and leave the room connected."""
        placed = list(self.pieces)
        for p in pieces:
            if not self._fits(p, placed):
                return False
            placed.append(p)
        if not self._connected(placed):
            return False
        for p in pieces:
            p.index = len(self.pieces)
            self.pieces.append(p)
        return True

    # Placement strategies.

    def against_wall(self, kind: str, walls: tuple[Wall, ...] = ("back", "left", "right"), tries: int = 30) -> bool:
        k = KINDS[kind]
        for _ in range(tries):
            wall = self.rng.choice(walls)
            rot = WALL_HEADING[wall]
            along = self.width if wall in ("back", "front") else self.depth
            half = k.w / 2
            if along < k.w + 0.2:
                continue
            s = self.rng.uniform(-along / 2 + half + 0.05, along / 2 - half - 0.05)
            off = k.d / 2 + 0.02
            at = {"back": (s, -self.depth + off), "front": (s, -off),
                  "left": (-self.width / 2 + off, -self.depth / 2 + s),
                  "right": (self.width / 2 - off, -self.depth / 2 + s)}[wall]  # fmt: skip
            if self._try([Piece(-1, k, _r(at), rot)]):
                return True
        return False

    def in_corner(self, kind: str, tries: int = 12) -> bool:
        k = KINDS[kind]
        for _ in range(tries):
            sx, sz = self.rng.choice((-1, 1)), self.rng.choice((-1, 0))
            x = sx * (self.width / 2 - k.w / 2 - 0.05)
            z = -self.depth + k.d / 2 + 0.05 if sz else -k.d / 2 - 0.05
            rot = math.pi / 2 * -sx if self.rng.random() < 0.5 else (0.0 if sz else math.pi)
            if self._try([Piece(-1, k, _r((x, z)), rot)]):
                return True
        return False

    def table_group(self, tries: int = 20) -> bool:
        table = KINDS["table"]
        for _ in range(tries):
            rot = self.rng.choice((0.0, math.pi / 2))
            x = self.rng.uniform(-self.width / 2 + 1.4, self.width / 2 - 1.4)
            z = self.rng.uniform(-self.depth + 1.4, -1.6)
            self._groups += 1
            g = self._groups
            t = Piece(-1, table, _r((x, z)), rot, group=g)
            group = [t]
            n_side = 2 if table.w >= 1.3 else 1
            gap = KINDS["chair"].d / 2 + 0.2
            for side in (1, -1):  # the table's front and back, chairs facing it
                for i in range(n_side):
                    lx = (i - (n_side - 1) / 2) * (table.w / n_side)
                    if self.rng.random() < 0.15:
                        continue
                    at = t.local(lx, side * (table.d / 2 + gap))
                    group.append(Piece(-1, KINDS["chair"], _r(at), rot + (math.pi if side > 0 else 0.0), group=g))
            if self.rng.random() < 0.5:  # one at the end
                end = self.rng.choice((1, -1))
                at = t.local(end * (table.w / 2 + gap), 0.0)
                group.append(Piece(-1, KINDS["chair"], _r(at), rot + end * math.pi / 2 * -1, group=g))
            if len(group) >= 3 and self._try(group):
                return True
        return False

    def facing_middle(self, kind: str, tries: int = 25) -> bool:
        """An armchair or couch standing off the walls, turned towards the middle of the room."""
        k = KINDS[kind]
        for _ in range(tries):
            x = self.rng.uniform(-self.width / 2 + 0.7, self.width / 2 - 0.7)
            z = self.rng.uniform(-self.depth + 0.7, -0.9)
            to_mid = math.atan2(-x, -self.depth / 2 - z)
            rot = round(to_mid / (math.pi / 2)) * (math.pi / 2)
            if self._try([Piece(-1, k, _r((x, z)), rot)]):
                return True
        return False

    def rug(self) -> bool:
        k = KINDS["rug"]
        for _ in range(10):
            x = self.rng.uniform(-self.width / 2 + 1.5, self.width / 2 - 1.5)
            z = self.rng.uniform(-self.depth + 1.2, -1.2)
            if self._try([Piece(-1, k, _r((x, z)), self.rng.choice((0.0, math.pi / 2)))]):
                return True
        return False

    def furnish(self, density: float) -> None:
        rng = self.rng
        # The essentials of a day room, then extras as density allows.
        self.against_wall(rng.choice(("couch", "couch", "loveseat")))
        if density > 0.15:
            self.table_group()
        if rng.random() < 0.35 + 0.5 * density:
            self.against_wall("bed", walls=("back", "left", "right"))
        for _ in range(1 + round(density * 2)):
            self.facing_middle("armchair")
        if density > 0.3 and rng.random() < 0.8:
            self.rug()
        if rng.random() < density:
            self.against_wall("cabinet")
        if rng.random() < density:
            self.against_wall("bookshelf")
        for _ in range(round(density * 2)):
            self.in_corner("plant")
        if rng.random() < density:
            self.against_wall("side_table")
        if not any(p.kind.name in ("chair", "armchair", "couch", "loveseat") for p in self.pieces):
            self.facing_middle("armchair", tries=60)  # always somewhere to sit
        self.pieces.sort(key=lambda p: p.index)


def _overlap(a: Footprint, b: Footprint) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _slots(pieces: list[Piece]) -> list[Slot]:
    out: list[Slot] = []
    for p in pieces:
        name, k, rot = p.kind.name, p.kind, p.rotation
        g = f"piece-{p.index}"
        if name in ("chair", "armchair"):
            lz = 0.03 if name == "chair" else 0.08
            out.append(Slot(f"{g}-sit", p.index, "sit", _r(p.local(0.0, lz)), rot, 0.0, g, seat=k.seat,
                            front=k.d / 2 - lz))  # fmt: skip
        elif name in ("couch", "loveseat"):
            n = 3 if name == "couch" else 2
            spacing = (k.w - 0.4) / n
            for i in range(n):
                lx = (i - (n - 1) / 2) * spacing
                out.append(Slot(f"{g}-sit{i + 1}", p.index, "sit", _r(p.local(lx, 0.12)), rot, 0.0, g, seat=k.seat,
                                front=k.d / 2 - 0.12))  # fmt: skip
            # Lying along it, head on the first seat's arm.
            out.append(Slot(f"{g}-lie", p.index, "lie", _r(p.local(0.0, 0.05)), rot + math.pi / 2,
                            k.surface, g, entry=f"{g}-sit1"))  # fmt: skip
        elif name == "bed":
            # Sit on the long side away from the nearer wall, then lie with the head at the headboard.
            for side, sign in (("a", 1.0), ("b", -1.0)):
                at = p.local(sign * (k.w / 2 - 0.2), 0.15)
                out.append(Slot(f"{g}-sit{side}", p.index, "sit", _r(at), rot + sign * math.pi / 2, 0.0, g, seat=k.seat,
                                front=0.2))  # fmt: skip
            out.append(Slot(f"{g}-lie", p.index, "lie", _r(p.local(0.0, 0.05)), rot, k.surface, g,
                            entry=f"{g}-sita"))  # fmt: skip
    return out


def _places(room: Room, reach: set) -> list[Place]:
    out: list[Place] = []
    nav = room.nav

    def ok(p: Floor) -> bool:
        return nav.free(p) and nav._cell(p) in reach

    for w in room.windows:
        for dx in (-0.4, 0.4):
            p = _r((w.at[0] + dx, w.at[1] + room.radius + 0.35))
            if ok(p):
                out.append(Place("window", p, math.pi))
    for piece in room.pieces:
        if piece.kind.name in ("bookshelf", "cabinet"):
            p = _r(_add(piece.at, forward(piece.rotation), piece.kind.d / 2 + room.radius + 0.3))
            if ok(p):
                out.append(Place("browse", p, piece.rotation + math.pi, piece.index))
    # The open middle: the roomiest reachable spots, a little apart.
    cells = sorted((c for c in reach if nav._clearance[c[0]][c[1]] >= 0.45),
                   key=lambda c: (abs(nav._centre(c)[0]) + abs(nav._centre(c)[1] + room.depth / 2), c))  # fmt: skip
    mids: list[Floor] = []
    for c in cells:
        p = nav._centre(c)
        if all(distance(p, q) > 1.4 for q in mids):
            mids.append(p)
        if len(mids) == 4:
            break
    rng = random.Random(f"places/{room.seed}")
    out.extend(Place("middle", _r(p), round(rng.uniform(-math.pi, math.pi), 3)) for p in mids)
    for d in room.doors:
        out.append(Place("door", d.inside, d.heading_in + math.pi))
    return out
