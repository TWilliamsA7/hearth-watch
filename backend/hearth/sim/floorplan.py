"""Floor plan: where people can walk in a scene, and routes between places.

Planning happens on the room floor in metres, in the same coordinates as the world track
(hearth.sim.world: x right, z = -depth * ROOM_DEPTH_M), not in frame coordinates, where
perspective would bend distances. Routes are converted back to frame x and depth to
become ordinary walking segments.

The room is the box the 3D view draws: as wide as the camera sees at the back wall and
ROOM_DEPTH_M deep. Its front corners are outside the camera's view. That is where
people come in and leave (DOORS), as the hand-written scenarios do with x < 0 or x > 1.

Furniture blocks walking, grown by a body radius, except mats (walked and lain on).
Occluders block too: they stand in front of what they hide (OCCLUDER_GAP_M).

Routes prefer open floor: a cell close to furniture or a wall costs more to cross, so
people walk down the middle of a gap rather than brushing past a corner, and the corners
of a route are rounded off. Places (places()) are the spots people actually go to: in
front of or beside furniture, at the window on the back wall, in the open middle of the
room, each with the way someone there would face.
"""

from __future__ import annotations

import math
import random
from collections.abc import Collection
from dataclasses import dataclass

from hearth.sim.nav import (  # noqa: F401  (re-exported: older callers import these from here)
    BODY_RADIUS_M,
    CELL_M,
    CLEARANCE_COST,
    CLEARANCE_M,
    CORNER_ROUND_M,
    ROOM_TOLERANCE_M,
    ROOMY_MARGIN_M,
    SHORTCUT_MARGIN_M,
    Cell,
    Floor,
    NavGrid,
    distance,
)
from hearth.sim.scenario import ObjectKind, Scene, SceneObject, Segment
from hearth.sim.world import ROOM_DEPTH_M, metres_per_x, room_geometry, world_x, world_z

FRONT_DEPTH = 0.02  # nobody stands right at the frame's bottom edge
BACK_DEPTH = 0.95  # nor flat against the back wall
WALKABLE: frozenset[ObjectKind] = frozenset({"mat"})
DOOR_DEPTH = 0.08
PLACE_GAP_M = 0.3  # people stand this far from the furniture they've come to


def to_world(scene: Scene, x: float, depth: float) -> Floor:
    return world_x(scene, x, depth), world_z(depth)


def to_frame(scene: Scene, p: Floor) -> tuple[float, float]:
    """(frame x, depth) of a floor point; inverse of to_world."""
    depth = -p[1] / ROOM_DEPTH_M
    return p[0] / metres_per_x(scene, depth) + 0.5, depth


@dataclass(frozen=True)
class Place:
    """Somewhere a person goes on purpose, and the way they face when they get there."""

    name: str  # e.g. "window", "front-couch", "middle"
    at: Floor
    heading: float  # radians, as Segment.heading
    near: int | None = None  # index into scene.objects of what it is beside


@dataclass(frozen=True)
class Seat:
    """Where a person sits or lies on a piece of furniture, in the terms segments use."""

    index: int  # into scene.objects
    kind: ObjectKind
    x: float  # frame x
    depth: float
    surface_y: float | None  # frame y to lie on; None = on the floor (sitting poses carry their own seat)

    def at(self, scene: Scene) -> Floor:
        return to_world(scene, self.x, self.depth)


def object_depth(scene: Scene, o: SceneObject) -> float:
    """The depth an object stands at: where its bottom edge meets the floor (as world._object)."""
    if not scene.depth_rise:
        return 0.0
    return min(max((scene.floor_y - (o.y + o.h)) / scene.depth_rise, 0.0), 1.0)


def seat_of(scene: Scene, index: int) -> Seat:
    o = scene.objects[index]
    depth = object_depth(scene, o)
    # Lie a little below the top edge, as the hand-written couch scenarios do (y 0.64 -> 0.68).
    surface = None if o.kind in ("chair", "mat") else round(o.y + 0.2 * o.h, 4)
    return Seat(index, o.kind, round(o.x + o.w / 2, 4), round(depth, 4), surface)


class FloorPlan(NavGrid):
    def __init__(self, scene: Scene, *, cell: float = CELL_M, radius: float = BODY_RADIUS_M) -> None:
        self.scene = scene
        room = room_geometry(scene)
        self.half_width = room.width_back / 2
        # Footprints as (x0, z0, x1, z1), grown by the body radius.
        footprints: list[tuple[float, float, float, float] | None] = []
        for o in room.objects:
            if o.kind in WALKABLE:
                footprints.append(None)
                continue
            (cx, _, cz), (sx, _, sz) = o.center, o.size
            footprints.append((cx - sx / 2 - radius, cz - sz / 2 - radius, cx + sx / 2 + radius, cz + sz / 2 + radius))
        super().__init__(-self.half_width + radius, self.half_width - radius, world_z(BACK_DEPTH), world_z(FRONT_DEPTH),
                         footprints, cell=cell, radius=radius)  # fmt: skip
        self.doors: dict[str, Floor] = {
            "left": (self.x_min, world_z(DOOR_DEPTH)),
            "right": (self.x_max, world_z(DOOR_DEPTH)),
        }
        self._places: list[Place] | None = None

    # --- geometry ---

    def visible(self, p: Floor, margin: float = 0.05) -> bool:
        """Would the camera see someone standing here (frame x inside the picture)?"""
        x, _ = to_frame(self.scene, p)
        return margin <= x <= 1.0 - margin

    def seats(self, kinds: Collection[ObjectKind] = ("chair", "couch", "bed", "mat")) -> list[Seat]:
        return [seat_of(self.scene, i) for i, o in enumerate(self.scene.objects) if o.kind in kinds]

    def approach(self, seat: Seat) -> Floor:
        """Where to stand before sitting or lying down: just in front of the furniture, or on it (a mat)."""
        at = seat.at(self.scene)
        f = self.footprints[seat.index]
        if f is None:
            return at
        return at[0], min(f[3] + self.cell / 2, self.z_max)

    def route_to_seat(self, start: Floor, seat: Seat, **kw) -> list[Floor] | None:
        """Walk to the front of a seat, then step back into it."""
        route = self.find_path(start, self.approach(seat), **kw)
        return None if route is None else route + [seat.at(self.scene)]

    def hiding_spots(self) -> list[tuple[Floor, float]]:
        """(where to stand, frame y below which the body is hidden) behind each occluder."""
        spots = []
        for o in self.scene.objects:
            if o.kind != "occluder":
                continue
            depth = object_depth(self.scene, o)
            for d in (depth, depth + 0.05, depth + 0.1):
                p = to_world(self.scene, o.x + o.w / 2, d)
                if self.free(p):
                    spots.append((p, o.y))
                    break
        return spots

    def random_point(self, rng: random.Random, *, visible: bool = True, tries: int = 200) -> Floor:
        for _ in range(tries):
            p = (rng.uniform(self.x_min, self.x_max), rng.uniform(self.z_min, self.z_max))
            if self.free(p) and (not visible or self.visible(p)):
                return (round(p[0], 3), round(p[1], 3))
        raise RuntimeError("no free floor found; is the room full of furniture?")

    # --- places ---

    def places(self) -> list[Place]:
        """Where people go: beside furniture facing it, the window, the open middle of the room."""
        if self._places is None:
            self._places = self._find_places()
        return self._places

    def _find_places(self) -> list[Place]:
        out: list[Place] = []
        room = room_geometry(self.scene)
        for index, (o, f) in enumerate(zip(room.objects, self.footprints)):
            (cx, _, cz) = o.center
            if f is None:  # a mat: stand on it
                if self.visible((cx, cz), margin=0.1):
                    out.append(Place(f"on-{o.kind}", (round(cx, 3), round(cz, 3)), math.pi, index))
                continue
            gap = PLACE_GAP_M  # from the grown footprint: a body radius plus this from the furniture
            sides = [
                ("front", (cx, f[3] + gap), math.pi),
                ("left-of", (f[0] - gap, cz), math.pi / 2),
                ("right-of", (f[2] + gap, cz), -math.pi / 2),
            ]
            for name, p, heading in sides:
                if self.free(p) and self.visible(p, margin=0.08):
                    out.append(Place(f"{name}-{o.kind}", (round(p[0], 3), round(p[1], 3)), heading, index))
        # The window: along the back wall, looking out.
        z = self.z_min + 0.05
        for frac in (0.3, 0.5, 0.7):
            p = (self.x_min + (self.x_max - self.x_min) * frac, z)
            if self.free(p, margin=0.1) and self.visible(p, margin=0.1):
                out.append(Place("window", (round(p[0], 3), round(p[1], 3)), math.pi))
        # The open middle: the roomiest visible spots, a little apart, facing the camera side.
        roomy = sorted(
            ((i, j) for i in range(self.nx) for j in range(self.nz)
             if self._clearance[i][j] >= CLEARANCE_M and self.visible(self._centre((i, j)), margin=0.15)),
            key=lambda c: (abs(self._centre(c)[0]) + abs(self._centre(c)[1] - (self.z_min + self.z_max) / 2), c),
        )  # fmt: skip
        middles: list[Floor] = []
        for c in roomy:
            p = self._centre(c)
            if all(distance(p, q) > 1.2 for q in middles):
                middles.append(p)
            if len(middles) == 3:
                break
        out.extend(Place("middle", (round(p[0], 3), round(p[1], 3)), 0.0) for p in middles)
        return out


def route_segments(scene: Scene, route: list[Floor], speed: float, *, first_transition: float = 0.3) -> list[Segment]:
    """One walking segment per leg of a route, timed for a walking speed in metres per second."""
    segments = []
    for a, b in zip(route, route[1:]):
        length = distance(a, b)
        if length < 0.01:
            continue
        x, depth = to_frame(scene, b)
        duration = round(max(length / speed, 0.25), 3)
        transition = min(first_transition if not segments else 0.2, duration)
        segments.append(Segment(label="walking", duration=duration, transition=transition,
                                x=round(x, 4), depth=round(min(max(depth, 0.0), 1.0), 4)))  # fmt: skip
    return segments
