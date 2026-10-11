"""Navigation grid: where a body can stand in a rectangle of floor, and routes across it.

Coordinates are metres on the floor, (x, z), as the world track uses (hearth.sim.world).
Furniture is a list of axis-aligned footprints already grown by the body radius; None
marks furniture that can be walked on (a mat or rug). The scripted generator's FloorPlan
and the live engine's rooms (hearth.sim.live.room) both plan on one of these.

Routes prefer open floor: a cell close to furniture or a wall costs more to cross, so
people walk down the middle of a gap rather than brushing past a corner, and the corners
of a route are rounded off.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Collection, Sequence

Floor = tuple[float, float]
"""A point on the floor: (x, z) in metres, world coordinates."""
Cell = tuple[int, int]
Footprint = tuple[float, float, float, float]
"""(x0, z0, x1, z1), grown by the body radius."""
Disc = tuple[float, float, float]
"""(x, z, radius): a temporary obstacle, such as another person standing in the way."""

CELL_M = 0.1
BODY_RADIUS_M = 0.25
ROOM_TOLERANCE_M = 0.005  # routes hugging a wall drift this far past it through frame-coordinate rounding
SHORTCUT_MARGIN_M = 0.02  # smoothed routes keep this clear of furniture, so rounding can't clip a corner
ROOMY_MARGIN_M = 0.15  # ...and this much, where there is room for it
CLEARANCE_M = 0.45  # cells closer than this to furniture or a wall cost more to walk through
CLEARANCE_COST = 2.0  # how much more, right up against it
CORNER_ROUND_M = 0.35  # corners of a route are rounded off over up to this distance either side


def distance(a: Floor, b: Floor) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


class NavGrid:
    def __init__(
        self,
        x_min: float,
        x_max: float,
        z_min: float,
        z_max: float,
        footprints: Sequence[Footprint | None],
        *,
        cell: float = CELL_M,
        radius: float = BODY_RADIUS_M,
    ) -> None:
        """Bounds are where a body's centre may be (walls already pulled in by the radius)."""
        self.cell = cell
        self.radius = radius
        self.x_min, self.x_max, self.z_min, self.z_max = x_min, x_max, z_min, z_max
        self.nx = int((self.x_max - self.x_min) / cell) + 1
        self.nz = int((self.z_max - self.z_min) / cell) + 1
        self.footprints: list[Footprint | None] = list(footprints)
        self._avoid: tuple[Disc, ...] = ()
        self._clearance = self._clearances()

    # --- geometry ---

    def blockers(self, p: Floor, margin: float = 0.0) -> set[int]:
        """Indices of the objects whose (grown) footprint contains p."""
        x, z, m = p[0], p[1], margin
        return {i for i, f in enumerate(self.footprints)
                if f and f[0] - m <= x <= f[2] + m and f[1] - m <= z <= f[3] + m}  # fmt: skip

    def in_room(self, p: Floor) -> bool:
        x, z = p
        e = ROOM_TOLERANCE_M
        return self.x_min - e <= x <= self.x_max + e and self.z_min - e <= z <= self.z_max + e

    def free(self, p: Floor, ignore: Collection[int] = (), margin: float = 0.0) -> bool:
        if not self.in_room(p) or (self.blockers(p, margin) - set(ignore)):
            return False
        return not any(math.hypot(p[0] - x, p[1] - z) < r for x, z, r in self._avoid)

    def line_free(self, a: Floor, b: Floor, ignore: Collection[int] = (), margin: float = 0.0) -> bool:
        steps = max(1, int(distance(a, b) / (self.cell / 4)))
        return all(
            self.free((a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps), ignore, margin)
            for k in range(steps + 1)
        )

    # --- grid ---

    def _cell(self, p: Floor) -> Cell:
        i = round((p[0] - self.x_min) / self.cell)
        j = round((p[1] - self.z_min) / self.cell)
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.nz - 1)

    def _centre(self, c: Cell) -> Floor:
        return self.x_min + c[0] * self.cell, self.z_min + c[1] * self.cell

    def find_path(
        self, start: Floor, goal: Floor, *, ignore: Collection[int] = (), avoid: Sequence[Disc] = ()
    ) -> list[Floor] | None:
        """A smoothed route from start to goal (both included), or None if there is none.

        Furniture under the start or goal is ignored, so routes can begin on the couch and
        end in the chair. `avoid` adds temporary round obstacles (other people), except
        any covering the start or goal.
        """
        self._avoid = tuple(d for d in avoid
                            if math.hypot(start[0] - d[0], start[1] - d[1]) >= d[2]
                            and math.hypot(goal[0] - d[0], goal[1] - d[1]) >= d[2])  # fmt: skip
        try:
            return self._find_path(start, goal, set(ignore) | self.blockers(start) | self.blockers(goal))
        finally:
            self._avoid = ()

    def _find_path(self, start: Floor, goal: Floor, ignore: set[int]) -> list[Floor] | None:
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
                    nd = d + step * self._cost(n)
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
        route = self._smooth(points, ignore)
        # The grid can't see everything between a point and the cell it snapped to.
        if not all(self.line_free(a, b, ignore) for a, b in zip(route, route[1:])):
            return None
        return self._round(route, ignore)

    def reachable(self, start: Floor) -> set[Cell]:
        """Every cell a body can walk to from start (8-connected, not cutting corners)."""
        s = self._cell(start)
        seen = {s}
        todo = [s]
        while todo:
            c = todo.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    n = (c[0] + di, c[1] + dj)
                    if n in seen or not self._open(n, ()):
                        continue
                    if di and dj and not (self._open((c[0] + di, c[1]), ()) and self._open((c[0], c[1] + dj), ())):
                        continue
                    seen.add(n)
                    todo.append(n)
        return seen

    def _open(self, c: Cell, ignore: Collection[int]) -> bool:
        return 0 <= c[0] < self.nx and 0 <= c[1] < self.nz and self.free(self._centre(c), ignore)

    def _octile(self, a: Cell, b: Cell) -> float:
        dx, dz = abs(a[0] - b[0]), abs(a[1] - b[1])
        return self.cell * (max(dx, dz) + (math.sqrt(2.0) - 1.0) * min(dx, dz))

    def _smooth(self, points: list[Floor], ignore: Collection[int]) -> list[Floor]:
        """String-pulling: keep only the corners a straight line can't cut. A shortcut must
        give furniture a wide berth, unless the grid's own path squeezes past it anyway."""
        out = [points[0]]
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self._shortcut(points, i, j, ignore):
                j -= 1
            out.append(points[j])
            i = j
        return out

    def _shortcut(self, points: list[Floor], i: int, j: int, ignore: Collection[int]) -> bool:
        if self.line_free(points[i], points[j], ignore, ROOMY_MARGIN_M):
            return True
        squeezed = not all(self.free(p, ignore, ROOMY_MARGIN_M) for p in points[i : j + 1])
        return squeezed and self.line_free(points[i], points[j], ignore, SHORTCUT_MARGIN_M)

    def _round(self, route: list[Floor], ignore: Collection[int]) -> list[Floor]:
        """Round each corner off: a few points on a curve from before it to after it."""
        if len(route) < 3:
            return route
        out = [route[0]]
        for a, b, c in zip(route, route[1:], route[2:]):
            ab, bc = distance(a, b), distance(b, c)
            r = min(CORNER_ROUND_M, ab / 2, bc / 2)
            if r < 0.05:
                out.append(b)
                continue
            p0 = (b[0] + (a[0] - b[0]) * r / ab, b[1] + (a[1] - b[1]) * r / ab)
            p2 = (b[0] + (c[0] - b[0]) * r / bc, b[1] + (c[1] - b[1]) * r / bc)
            n = max(1, min(4, int(r / 0.08)))  # legs of a few centimetres at least
            curve = [p0] + [_bezier(p0, b, p2, k / n) for k in range(1, n)] + [p2]
            if all(self.line_free(p, q, ignore) for p, q in zip([out[-1]] + curve, curve + [c])):
                out.extend(curve)
            else:
                out.append(b)
        out.append(route[-1])
        return out

    # --- open floor ---

    def _clearances(self) -> list[list[float]]:
        """For every cell, metres to the nearest blocked cell or wall (capped at CLEARANCE_M)."""
        dist = [[math.inf] * self.nz for _ in range(self.nx)]
        queue: list[tuple[float, int, int]] = []
        for i in range(self.nx):
            for j in range(self.nz):
                if self.free(self._centre((i, j))):
                    d = min(i + 1, j + 1, self.nx - i, self.nz - j) * self.cell
                else:
                    d = 0.0
                dist[i][j] = d
                queue.append((d, i, j))
        heapq.heapify(queue)
        while queue:
            d, i, j = heapq.heappop(queue)
            if d > dist[i][j] or d >= CLEARANCE_M:
                continue
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if (di or dj) and 0 <= ni < self.nx and 0 <= nj < self.nz:
                        nd = d + self.cell * (math.sqrt(2.0) if di and dj else 1.0)
                        if nd < dist[ni][nj]:
                            dist[ni][nj] = nd
                            heapq.heappush(queue, (nd, ni, nj))
        return dist

    def clearance(self, p: Floor) -> float:
        """Metres of open floor around p (capped at CLEARANCE_M)."""
        i, j = self._cell(p)
        return min(self._clearance[i][j], CLEARANCE_M)

    def _cost(self, c: Cell) -> float:
        tight = 1.0 - min(self._clearance[c[0]][c[1]], CLEARANCE_M) / CLEARANCE_M
        return 1.0 + CLEARANCE_COST * tight * tight


def _bezier(a: Floor, b: Floor, c: Floor, t: float) -> Floor:
    u = 1.0 - t
    return (u * u * a[0] + 2 * u * t * b[0] + t * t * c[0], u * u * a[1] + 2 * u * t * b[1] + t * t * c[1])
