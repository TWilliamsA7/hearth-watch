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
"""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Callable, Collection
from dataclasses import dataclass

from hearth.sim.scenario import ObjectKind, Scene, SceneObject, Segment
from hearth.sim.world import ROOM_DEPTH_M, metres_per_x, room_geometry, world_x, world_z

Floor = tuple[float, float]
"""A point on the floor: (x, z) in metres, world coordinates."""
Cell = tuple[int, int]

CELL_M = 0.1
BODY_RADIUS_M = 0.25
FRONT_DEPTH = 0.02  # nobody stands right at the frame's bottom edge
BACK_DEPTH = 0.95  # nor flat against the back wall
WALKABLE: frozenset[ObjectKind] = frozenset({"mat"})
DOOR_DEPTH = 0.08
SHORTCUT_MARGIN_M = 0.02  # smoothed routes keep this clear of furniture, so rounding can't clip a corner


def to_world(scene: Scene, x: float, depth: float) -> Floor:
    return world_x(scene, x, depth), world_z(depth)


def to_frame(scene: Scene, p: Floor) -> tuple[float, float]:
    """(frame x, depth) of a floor point; inverse of to_world."""
    depth = -p[1] / ROOM_DEPTH_M
    return p[0] / metres_per_x(scene, depth) + 0.5, depth


def distance(a: Floor, b: Floor) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


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


class FloorPlan:
    def __init__(self, scene: Scene, *, cell: float = CELL_M, radius: float = BODY_RADIUS_M) -> None:
        self.scene = scene
        self.cell = cell
        self.radius = radius
        room = room_geometry(scene)
        self.half_width = room.width_back / 2
        self.x_min, self.x_max = -self.half_width + radius, self.half_width - radius
        self.z_min, self.z_max = world_z(BACK_DEPTH), world_z(FRONT_DEPTH)
        self.nx = int((self.x_max - self.x_min) / cell) + 1
        self.nz = int((self.z_max - self.z_min) / cell) + 1
        # Footprints as (x0, z0, x1, z1), grown by the body radius.
        self.footprints: list[tuple[float, float, float, float] | None] = []
        for o in room.objects:
            if o.kind in WALKABLE:
                self.footprints.append(None)
                continue
            (cx, _, cz), (sx, _, sz) = o.center, o.size
            self.footprints.append((cx - sx / 2 - radius, cz - sz / 2 - radius, cx + sx / 2 + radius, cz + sz / 2 + radius))
        self.doors: dict[str, Floor] = {
            "left": (self.x_min, world_z(DOOR_DEPTH)),
            "right": (self.x_max, world_z(DOOR_DEPTH)),
        }

    # --- geometry ---

    def blockers(self, p: Floor, margin: float = 0.0) -> set[int]:
        """Indices of the objects whose (grown) footprint contains p."""
        x, z, m = p[0], p[1], margin
        return {i for i, f in enumerate(self.footprints)
                if f and f[0] - m <= x <= f[2] + m and f[1] - m <= z <= f[3] + m}  # fmt: skip

    def in_room(self, p: Floor) -> bool:
        x, z = p
        return self.x_min - 1e-9 <= x <= self.x_max + 1e-9 and self.z_min - 1e-9 <= z <= self.z_max + 1e-9

    def free(self, p: Floor, ignore: Collection[int] = (), margin: float = 0.0) -> bool:
        return self.in_room(p) and not (self.blockers(p, margin) - set(ignore))

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

    def route_from_seat(self, seat: Seat, goal: Floor, **kw) -> list[Floor] | None:
        """Step forward out of a seat, then walk to goal."""
        route = self.find_path(self.approach(seat), goal, **kw)
        return None if route is None else [seat.at(self.scene)] + route

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

    # --- grid ---

    def _cell(self, p: Floor) -> Cell:
        i = round((p[0] - self.x_min) / self.cell)
        j = round((p[1] - self.z_min) / self.cell)
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.nz - 1)

    def _centre(self, c: Cell) -> Floor:
        return self.x_min + c[0] * self.cell, self.z_min + c[1] * self.cell

    def line_free(self, a: Floor, b: Floor, ignore: Collection[int] = (), margin: float = 0.0) -> bool:
        steps = max(1, int(distance(a, b) / (self.cell / 4)))
        return all(
            self.free((a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps), ignore, margin)
            for k in range(steps + 1)
        )

    def find_path(
        self, start: Floor, goal: Floor, *, ignore: Collection[int] = (), cost: Callable[[Floor], float] | None = None
    ) -> list[Floor] | None:
        """A smoothed route from start to goal (both included), or None if there is none.

        Furniture under the start or goal is ignored, so routes can begin on the couch and
        end in the chair. cost(p) adds an extra cost per metre walked through p (e.g. to
        keep away from other people); it must be >= 0.
        """
        ignore = set(ignore) | self.blockers(start) | self.blockers(goal)
        if not (self.in_room(start) and self.in_room(goal)):
            return None
        s, g = self._cell(start), self._cell(goal)
        frontier: list[tuple[float, float, Cell]] = [(0.0, 0.0, s)]
        came: dict[Cell, Cell | None] = {s: None}
        best: dict[Cell, float] = {s: 0.0}
        found = False
        while frontier:
            _, d, c = heapq.heappop(frontier)
            if c == g:
                found = True
                break
            if d > best[c]:
                continue
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    n = (c[0] + di, c[1] + dj)
                    if n == c or not (0 <= n[0] < self.nx and 0 <= n[1] < self.nz):
                        continue
                    p = self._centre(n)
                    if n != g and not self.free(p, ignore):
                        continue
                    if di and dj and not (self._open((c[0] + di, c[1]), ignore) and self._open((c[0], c[1] + dj), ignore)):
                        continue  # don't cut a corner of furniture
                    step = self.cell * (math.sqrt(2.0) if di and dj else 1.0)
                    nd = d + step * (1.0 + (cost(p) if cost else 0.0))
                    if nd < best.get(n, math.inf):
                        best[n], came[n] = nd, c
                        heapq.heappush(frontier, (nd + self._octile(n, g), nd, n))
        if not found:
            return None
        cells: list[Cell] = []
        c: Cell | None = g
        while c is not None:
            cells.append(c)
            c = came[c]
        points = [start] + [self._centre(c) for c in reversed(cells[1:-1])] + [goal]
        return self._smooth(points, ignore) if cost is None else self._thin(points)

    def _open(self, c: Cell, ignore: Collection[int]) -> bool:
        return 0 <= c[0] < self.nx and 0 <= c[1] < self.nz and self.free(self._centre(c), ignore)

    def _octile(self, a: Cell, b: Cell) -> float:
        dx, dz = abs(a[0] - b[0]), abs(a[1] - b[1])
        return self.cell * (max(dx, dz) + (math.sqrt(2.0) - 1.0) * min(dx, dz))

    def _smooth(self, points: list[Floor], ignore: Collection[int]) -> list[Floor]:
        """String-pulling: keep only the corners a straight line can't cut."""
        out = [points[0]]
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self.line_free(points[i], points[j], ignore, SHORTCUT_MARGIN_M):
                j -= 1
            out.append(points[j])
            i = j
        return out

    def _thin(self, points: list[Floor]) -> list[Floor]:
        """Drop points in the middle of straight runs (cost-shaped routes aren't cut short)."""
        out = [points[0]]
        for prev, p, nxt in zip(points, points[1:], points[2:]):
            if (p[0] - prev[0]) * (nxt[1] - p[1]) != (p[1] - prev[1]) * (nxt[0] - p[0]):
                out.append(p)
        out.append(points[-1])
        return out


def route_segments(scene: Scene, route: list[Floor], speed: float, *, first_transition: float = 0.3) -> list[Segment]:
    """One walking segment per leg of a route, timed for a walking speed in metres per second."""
    segments = []
    for a, b in zip(route, route[1:]):
        length = distance(a, b)
        if length < 0.05:
            continue
        x, depth = to_frame(scene, b)
        duration = round(max(length / speed, 0.4), 3)
        transition = min(first_transition if not segments else 0.2, duration)
        segments.append(Segment(label="walking", duration=duration, transition=transition,
                                x=round(x, 4), depth=round(min(max(depth, 0.0), 1.0), 4)))  # fmt: skip
    return segments
