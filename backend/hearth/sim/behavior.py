"""Resident behaviour: goals, personas and faults, compiled into ordinary segments.

An Actor is one resident being planned. It keeps where they are, what they're doing and
the clock, and each action (walk somewhere, sit, lie down, bend, fall, leave) appends
segments to their script. live() picks goals from the persona's weights until the
scenario's time is used up. All of this happens before the run: the result is a plain
Resident, which the generator plays like any hand-written one.

Several residents share a Household: the floor plan, which seats are taken when, and
(see hearth.sim.generate) where everyone planned so far will be.

Randomness comes only from the rng each Actor is given, so the same seed always gives the
same script.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Literal

from hearth.sim.floorplan import Floor, FloorPlan, Seat, distance, route_segments, to_frame, to_world
from hearth.sim.generator import ResidentTrack
from hearth.sim.scenario import Facing, Resident, Segment, TagValue

State = Literal["standing", "sitting", "lying", "fallen", "absent"]
Goal = Literal["sit", "rest", "pick_up", "idle", "wander", "leave", "browse"]
FallKind = Literal["trip", "sit_miss", "stand_up", "faint"]
FallOutcome = Literal["long_lie", "recover"]

FALL_KINDS: tuple[FallKind, ...] = ("trip", "sit_miss", "stand_up", "faint")
FALL_CLEARANCE_M = 0.55  # floor needed either side of where someone lands, along their body
LONG_LIE_MIN_S = 15.0
PERSONAL_SPACE_M = 0.5  # closest two people come (body centres, or a standing body and a lying one)
LYING_HALF_LENGTH_M = 0.75  # someone lying on the floor reaches this far either side, along x
CLASH_STEP_S = 0.1
LOOKAHEAD_S = 2.5


@dataclass(frozen=True)
class Persona:
    name: str
    speed: tuple[float, float]  # walking speed, m/s
    height: tuple[float, float]  # relative to Scene.person_height
    slowness: float  # scales how long getting up and sitting down take
    fall_weight: float  # relative chance of being the one who falls
    goals: dict[Goal, float]


PERSONAS: dict[str, Persona] = {
    "steady": Persona("steady", (0.85, 1.15), (0.95, 1.06), 1.0, 1.0,
                      {"sit": 3, "rest": 1, "pick_up": 2, "idle": 2, "wander": 3, "leave": 1, "browse": 1}),
    "frail": Persona("frail", (0.45, 0.7), (0.9, 1.0), 1.6, 2.5,
                     {"sit": 4, "rest": 3, "pick_up": 1, "idle": 2, "wander": 1, "leave": 0.5, "browse": 0.5}),
    "restless": Persona("restless", (0.9, 1.3), (0.95, 1.08), 0.8, 0.8,
                        {"sit": 1, "rest": 0.5, "pick_up": 2, "idle": 1, "wander": 5, "leave": 2, "browse": 2}),
}  # fmt: skip


@dataclass(frozen=True)
class Leg:
    """One straight stretch of walking, for drawing routes and checking for collisions."""

    t0: float
    t1: float
    a: Floor
    b: Floor


@dataclass(frozen=True)
class FallPlan:
    kind: FallKind
    at: float  # scenario seconds; the fall happens at the first chance after this
    outcome: FallOutcome


@dataclass
class Household:
    """What residents share while they're planned: the room, its seats, and each other.

    Residents are planned one at a time. Everyone planned so far is fixed (`planned`);
    the one being planned must keep out of their way (clash()).
    """

    plan: FloorPlan
    bookings: dict[int, list[tuple[float, float]]] = field(default_factory=dict)  # seat index -> busy intervals
    planned: dict[str, ResidentTrack] = field(default_factory=dict)

    def seat_free(self, seat: Seat, t0: float, t1: float) -> bool:
        return all(b1 <= t0 or t1 <= b0 for b0, b1 in self.bookings.get(seat.index, []))

    def book(self, seat: Seat, t0: float, t1: float) -> None:
        self.bookings.setdefault(seat.index, []).append((t0, t1))

    def settle(self, actor: Actor) -> None:
        """Fix an actor's plan; everyone planned after them works around it."""
        if actor.seat is not None:  # still seated at the end: the seat stays taken
            self.book(actor.seat, actor.seated_since, math.inf)
        self.planned[actor.resident_id] = ResidentTrack(actor.resident(), self.plan.scene)

    def body(self, track: ResidentTrack, t: float) -> tuple[Floor, float] | None:
        """Where a body is on the floor at t, as (centre, half length along x); None if not in the way."""
        seg = track.plans[track.segment_index(t)].segment
        if seg.label in ("absent", "sitting") or (seg.label == "lying_intentional" and seg.surface_y is not None):
            return None  # out of the room, or on furniture that walkers already avoid
        x, depth = track.position_at(t)
        lying = seg.label in ("fall", "lying_intentional")
        return to_world(self.plan.scene, x, depth), LYING_HALF_LENGTH_M if lying else 0.0

    def clash(self, actor: Actor, t0: float, t1: float, *, allow: Collection[str] = ()) -> float | None:
        """First time in [t0, t1] the actor comes too close to anyone already planned, or None."""
        others = [tr for rid, tr in self.planned.items() if rid != actor.resident_id and rid not in allow]
        if not others or t1 <= t0:
            return None
        mine = ResidentTrack(actor.resident(), self.plan.scene)
        steps = max(1, math.ceil((t1 - t0) / CLASH_STEP_S))
        for k in range(steps + 1):
            t = t0 + (t1 - t0) * k / steps
            me = self.body(mine, t)
            if me is None:
                continue
            for other in others:
                them = self.body(other, t)
                if them is not None and gap(me, them) < PERSONAL_SPACE_M:
                    return t
        return None


def gap(a: tuple[Floor, float], b: tuple[Floor, float]) -> float:
    """Distance between two bodies, each a point or a stretch along x."""
    (pa, ea), (pb, eb) = a, b
    return math.hypot(max(0.0, abs(pa[0] - pb[0]) - ea - eb), pa[1] - pb[1])


def depth_bin(depth: float) -> str:
    return "front" if depth < 1 / 3 else "middle" if depth < 2 / 3 else "back"


class Actor:
    def __init__(
        self, home: Household, resident_id: str, persona: Persona, rng: random.Random, *,
        height: float | None = None, speed: float | None = None,
    ) -> None:  # fmt: skip
        self.home = home
        self.plan = home.plan
        self.resident_id = resident_id
        self.persona = persona
        self.rng = rng
        self.height = round(height if height is not None else rng.uniform(*persona.height), 3)
        self.speed = round(speed if speed is not None else rng.uniform(*persona.speed), 3)
        self.segments: list[Segment] = []
        self.legs: list[Leg] = []
        self.t = 0.0
        self.state: State = "standing"
        self.seat: Seat | None = None
        self.seated_since = 0.0
        self.pos: Floor = (0.0, 0.0)
        self.facing: Facing = "camera"
        self.start: tuple[Floor, Facing] = (self.pos, self.facing)
        self.falls: list[Segment] = []
        self.fall_events: list[tuple[float, Floor]] = []  # (when the fall starts, where they land)
        self.allow: set[str] = set()  # residents this one may come close to (a responder and the faller)
        self.clashes = 0  # actions dropped because they'd have walked into someone
        self.unresolved = 0  # clashes it couldn't avoid

    # --- setup and output ---

    def place(self, pos: Floor, *, facing: Facing = "camera") -> None:
        self.pos, self.facing = pos, facing
        self.start = (pos, facing)

    def resident(self) -> Resident:
        (x, depth), facing = to_frame(self.plan.scene, self.start[0]), self.start[1]
        return Resident(resident_id=self.resident_id, height=self.height, start_x=round(x, 4),
                        start_depth=round(min(max(depth, 0.0), 1.0), 4), start_facing=facing,
                        segments=self.segments)  # fmt: skip

    def _u(self, lo: float, hi: float) -> float:
        return self.rng.uniform(lo, hi)

    def _seg(self, label: str, duration: float, transition: float = 0.0, **kw) -> Segment:
        transition = round(max(transition, 0.0), 3)
        duration = round(max(duration, transition, 0.1), 3)
        seg = Segment(label=label, duration=duration, transition=transition, **kw)  # type: ignore[arg-type]
        self.segments.append(seg)
        self.t = round(self.t + duration, 3)
        return seg

    # --- moving ---

    def get_up(self) -> None:
        slow = self.persona.slowness
        if self.state == "absent":
            return
        if self.state == "lying":
            tr = self._u(2.0, 3.0) * slow
            self._seg("sitting", tr + self._u(0.5, 1.5), tr, facing="camera")
            self.state = "sitting"
        if self.state == "fallen":
            tr = self._u(2.5, 4.0) * slow
            self._seg("standing", tr + self._u(0.5, 1.5), tr)
            self.state = "standing"
        if self.state == "sitting":
            tr = self._u(1.0, 1.6) * slow
            self._seg("standing", tr + self._u(0.3, 0.8), tr)
            self.state = "standing"
        if self.seat is not None:
            self.home.book(self.seat, self.seated_since, self.t)

    def _route(self, goal: Floor, seat: Seat | None = None, **kw) -> list[Floor] | None:
        """Route from wherever the actor is (stepping out of a seat first) to goal or into seat."""
        start = self.plan.approach(self.seat) if self.seat is not None else self.pos
        target = self.plan.approach(seat) if seat is not None else goal
        route = self.plan.find_path(start, target, **kw)
        if route is None:
            return None
        if self.seat is not None and route[0] != self.pos:
            route = [self.pos] + route
        if seat is not None:
            route = route + [seat.at(self.plan.scene)]
        return route

    def walk(self, route: list[Floor]) -> None:
        """Walk a route (route[0] must be where the actor is)."""
        self.get_up()
        self.seat = None
        t = self.t
        for seg in route_segments(self.plan.scene, route, self.speed):
            a = self.pos
            b = _floor_of(self.plan, seg)
            x_from = to_frame(self.plan.scene, a)[0]
            if seg.x is not None and abs(seg.x - x_from) > 1e-6:
                self.facing = "right" if seg.x > x_from else "left"
            self.segments.append(seg)
            self.legs.append(Leg(t, t + seg.duration, a, b))
            t = round(t + seg.duration, 3)
            self.pos = b
        self.t = t
        self.state = "standing"

    def go(self, goal: Floor) -> bool:
        if self.state == "absent":
            return False
        route = self._route(goal)
        if route is None:
            return False
        self.walk(route)
        return True

    def go_sit(self, seat: Seat) -> bool:
        route = self._route(seat.at(self.plan.scene), seat)
        if route is None:
            return False
        self.walk(route)
        return True

    # --- activities ---

    def stand(self, duration: float, facing: Facing | None = None) -> None:
        self.get_up()
        if facing is not None:
            self.facing = facing
        self._seg("standing", duration, min(0.4, duration), facing=facing)

    def sit(self, seat: Seat, duration: float, *, fast: bool = False) -> None:
        slow = self.persona.slowness
        tr = self._u(0.3, 0.45) if fast else self._u(0.6, 1.0) * slow
        tags: dict[str, TagValue] = {"hard_negative": "fast_sit"} if fast else {}
        self.facing = "camera"
        self._seg("sitting", duration + tr, tr, facing="camera", tags=tags)
        self.state, self.seat, self.seated_since = "sitting", seat, self.t - duration - tr

    def lie(self, seat: Seat, duration: float) -> None:
        """Lie down on a couch or bed (sitting on it first) or on a mat."""
        slow = self.persona.slowness
        since = self.t
        if seat.kind in ("couch", "bed"):
            self.sit(seat, self._u(1.0, 3.0))
            since = self.seated_since
        tr = self._u(2.5, 3.5) * slow
        self.facing = self.rng.choice(["left", "right"])
        self._seg("lying_intentional", duration + tr, tr, facing=self.facing, surface_y=seat.surface_y,
                  tags={"hard_negative": "lie_down", "on": seat.kind})  # fmt: skip
        self.state, self.seat, self.seated_since = "lying", seat, since

    def bend(self, duration: float) -> None:
        self.get_up()
        self._seg("bending", duration, self._u(0.8, 1.1), tags={"hard_negative": "bend"})
        self._seg("standing", self._u(1.0, 2.0), 0.9)

    def leave(self, door: str, away: float, come_back: str | None) -> bool:
        if not self.go(self.plan.doors[door]):
            return False
        x, depth = to_frame(self.plan.scene, self.plan.doors[come_back or door])
        self._seg("absent", away, x=round(x, 4), depth=round(depth, 4))
        self.pos = self.plan.doors[come_back or door]
        self.state = "absent" if come_back is None else "standing"
        return True

    def browse(self, spot: Floor, top: float, duration: float) -> bool:
        """Stand behind a tall piece of furniture, partly hidden from the camera."""
        if not self.go(spot):
            return False
        self.facing = "camera"
        self._seg("occluded", duration, 0.4, facing="camera", occluder_top=top,
                  tags={"hard_negative": "occluded"})  # fmt: skip
        return True

    # --- falls ---

    def can_land(self, p: Floor) -> bool:
        """Is there floor for a body here, in the camera's view?"""
        sides = ((p[0] - FALL_CLEARANCE_M, p[1]), p, (p[0] + FALL_CLEARANCE_M, p[1]))
        return all(self.plan.free(q) for q in sides) and self.plan.visible(p, margin=0.1)

    def fall(self, kind: FallKind, outcome: FallOutcome, until: float, *, at: Floor | None = None,
             lunge: float = 0.0) -> None:  # fmt: skip
        """Fall from wherever the actor is, landing `lunge` metres ahead (forward falls)."""
        direction = {"right": 1.0, "left": -1.0}.get(self.facing, 0.0)
        start = at or self.pos
        land = (start[0] + direction * lunge, start[1])
        if not self.can_land(land):
            land = start
        x, depth = to_frame(self.plan.scene, land)
        floor_time = max(until - self.t, LONG_LIE_MIN_S) if outcome == "long_lie" else self._u(3.0, 10.0)
        tags: dict[str, TagValue] = {
            "fault": kind, "outcome": outcome, "depth_bin": depth_bin(depth), "persona": self.persona.name,
            "facing": self.facing,
        }
        self.fall_events.append((self.t, land))
        seg = self._seg("fall", floor_time, self._u(0.6, 1.0), x=round(x, 4), depth=round(depth, 4), tags=tags)
        self.falls.append(seg)
        if self.seat is not None:
            self.home.book(self.seat, self.seated_since, self.t)
        self.pos, self.state, self.seat = land, "fallen", None

    def do_fall(self, fall: FallPlan, until: float) -> bool:
        """Carry out a planned fall. False if this room offers no way to do it right now."""
        if fall.kind == "trip":
            for _ in range(12):
                goal = self.plan.random_point(self.rng)
                route = self._route(goal)
                if route is None or _length(route) < 1.5:
                    continue
                cut = _cut(route, self._u(0.35, 0.8) * _length(route))
                if not self.can_land(cut[-1]):
                    continue
                self.walk(cut)
                self.fall("trip", fall.outcome, until, lunge=self._u(0.3, 0.6))
                return True
            return False
        if fall.kind == "sit_miss":
            seat = self.free_seat(("chair", "couch"), 10.0)
            if seat is None or not self.can_land(self.plan.approach(seat)):
                return False
            route = self._route(self.plan.approach(seat))
            if route is None:
                return False
            self.walk(route)
            self.facing = "camera"
            self.fall("sit_miss", fall.outcome, until)
            return True
        if fall.kind == "stand_up":
            if self.state not in ("sitting", "lying"):
                seat = self.free_seat(("chair", "couch"), 20.0)
                if seat is None or not self.can_land(self.plan.approach(seat)) or not self.go_sit(seat):
                    return False
                self.sit(seat, self._u(3.0, 8.0))
            assert self.seat is not None
            if not self.can_land(self.plan.approach(self.seat)):
                return False
            if self.state == "lying":
                tr = self._u(2.0, 3.0) * self.persona.slowness
                self._seg("sitting", tr + 0.5, tr, facing="camera")
            self._seg("standing", 1.0, 1.0)  # and straight back down as they come upright
            self.fall("stand_up", fall.outcome, until, at=self.plan.approach(self.seat))
            return True
        # faint: from standing, where there's room to go down
        if not self.can_land(self.pos) or self.state != "standing":
            spot = next((p for p in (self.plan.random_point(self.rng) for _ in range(12)) if self.can_land(p)), None)
            if spot is None or not self.go(spot):
                return False
        self.stand(self._u(1.0, 3.0))
        self.fall("faint", fall.outcome, until)
        return True

    # --- choosing what to do ---

    def free_seat(self, kinds: tuple[str, ...], stay: float) -> Seat | None:
        seats = [s for s in self.plan.seats(kinds) if s != self.seat]  # type: ignore[arg-type]
        seats = [s for s in seats if self.home.seat_free(s, self.t, self.t + 15.0 + stay)]
        return self.rng.choice(seats) if seats else None

    def do_goal(self, goal: Goal, until: float) -> bool:
        u = self._u
        if goal == "sit":
            stay = u(5.0, 20.0) * self.persona.slowness
            seat = self.free_seat(("chair", "couch"), stay)
            if seat is None or not self.go_sit(seat):
                return False
            self.sit(seat, stay, fast=self.rng.random() < 0.25)
            return True
        if goal == "rest":
            stay = u(8.0, 25.0)
            seat = self.free_seat(("couch", "bed", "mat"), stay)
            if seat is None:
                return False
            if seat.kind == "mat":
                if not self.go(self.plan.approach(seat)):
                    return False
            elif not self.go_sit(seat):
                return False
            self.lie(seat, stay)
            return True
        if goal == "pick_up":
            if not self.go(self.plan.random_point(self.rng)):
                return False
            self.bend(u(1.5, 4.0))
            return True
        if goal == "idle":
            self.stand(u(2.0, 6.0), self.rng.choice(["camera", "left", "right"]))
            return True
        if goal == "wander":
            return self.go(self.plan.random_point(self.rng))
        if goal == "leave":
            if until - self.t < 15.0:
                return False
            door = min(self.plan.doors, key=lambda d: distance(self.pos, self.plan.doors[d]))
            return self.leave(door, u(4.0, 12.0), self.rng.choice(list(self.plan.doors)))
        if goal == "browse":
            spots = self.plan.hiding_spots()
            if not spots:
                return False
            spot, top = self.rng.choice(spots)
            return self.browse(spot, top, u(4.0, 10.0))
        raise ValueError(goal)

    def live(self, until: float, fall: FallPlan | None = None, *, pick: Callable[[Actor], Goal] | None = None) -> None:
        """Pursue goals until `until` (scenario seconds), carrying out the planned fall on the way."""
        pending, failures = fall, 0
        for _ in range(200):
            if self.t >= until - 1.0 and pending is None:
                break
            if pending is not None and self.t >= pending.at:
                if self._try(lambda: self.do_fall(pending, until)):
                    if pending.outcome == "long_lie":
                        break
                    pending = None
                    continue
                # Try again a little later. If this room keeps making it impossible (seats taken,
                # no floor in view), fall another way: a trip, then simply going down.
                failures += 1
                kind = pending.kind if failures < 3 else "trip" if failures < 5 else "faint"
                pending = FallPlan(kind, self.t + 2.0, pending.outcome) if failures < 8 else None
            goal = (pick or _pick_goal)(self)
            if not self._try(lambda: self.do_goal(goal, until)):
                self.wait()
        self.hold_clear(until)

    # --- keeping out of each other's way ---

    def _snapshot(self) -> tuple:
        return (list(self.segments), list(self.legs), self.t, self.state, self.seat, self.seated_since, self.pos,
                self.facing, list(self.falls), list(self.fall_events),
                {k: list(v) for k, v in self.home.bookings.items()})  # fmt: skip

    def _restore(self, snap: tuple) -> None:
        (self.segments, self.legs, self.t, self.state, self.seat, self.seated_since, self.pos, self.facing,
         self.falls, self.fall_events, self.home.bookings) = snap  # fmt: skip

    def _try(self, action: Callable[[], bool]) -> bool:
        """Do an action, or undo it if it fails or would walk into someone."""
        snap = self._snapshot()
        if not action():
            self._restore(snap)
            return False
        # Look a little past the end too: finishing in someone's path leaves no time to get out of it.
        if self.home.clash(self, snap[2], self.t + LOOKAHEAD_S, allow=self.allow) is not None:
            self._restore(snap)
            self.clashes += 1
            return False
        return True

    def _stand_still(self) -> bool:
        self.stand(self._u(1.0, 2.0))
        return True

    def wait(self) -> None:
        """Stay put a moment; if someone is coming through here, step somewhere else."""
        if self._try(self._stand_still):
            return
        x, z = self.pos
        for r in (0.8, 1.3, 2.0):  # step aside, a little further each time
            turn = self._u(0.0, math.pi / 4)
            for k in range(8):
                a = turn + k * math.pi / 4
                spot = (x + r * math.cos(a), z + r * math.sin(a))
                if self.plan.free(spot) and self._try(lambda: self.go(spot)):
                    return
        for _ in range(6):
            if self._try(lambda: self.go(self.plan.random_point(self.rng, visible=False))):
                return
        self.unresolved += 1
        self._stand_still()

    def hold_clear(self, until: float) -> None:
        """hold(), but keep out of the way of anyone who comes past in the meantime."""
        snap = self._snapshot()
        self.hold(until)
        if self.home.clash(self, snap[2], self.t, allow=self.allow) is None:
            return
        self._restore(snap)
        for _ in range(100):
            if self.t >= until - 0.5:
                break
            self.wait()
        self.hold(until)

    def respond(self, faller: Actor, until: float) -> bool:
        """Hurry to someone who has fallen and kneel beside them (bending stands in for kneeling)."""
        if not faller.fall_events:
            return False
        _, (fx, fz) = faller.fall_events[0]
        reach = LYING_HALF_LENGTH_M + PERSONAL_SPACE_M
        spots = [(fx, fz + 0.6), (fx - reach, fz), (fx + reach, fz), (fx, fz - 0.6)]
        self.allow.add(faller.resident_id)
        speed, self.speed = self.speed, round(min(self.speed * 1.3, 1.5), 3)
        try:
            for spot in spots:
                if self.plan.free(spot) and self._try(lambda: self._attend(spot, fx, until)):
                    return True
        finally:
            self.speed = speed
        self.allow.discard(faller.resident_id)
        return False

    def _attend(self, spot: Floor, faller_x: float, until: float) -> bool:
        if not self.go(spot):
            return False
        facing: Facing = "left" if faller_x < spot[0] - 0.1 else "right" if faller_x > spot[0] + 0.1 else "camera"
        self.facing = facing
        tags: dict[str, TagValue] = {"responder": True}
        self._seg("bending", self._u(4.0, 8.0), self._u(0.6, 0.9), facing=facing, tags=tags)
        self._seg("standing", self._u(2.0, 4.0), 0.9, tags=tags)
        if until - self.t > 6.0:
            self._seg("bending", self._u(3.0, 6.0), 0.8, tags=tags)
            self._seg("standing", 1.0, 0.9, tags=tags)
        return True

    def hold(self, until: float) -> None:
        """Carry on with the current activity until `until`."""
        extra = round(until - self.t, 3)
        if extra <= 0:
            return
        last = self.segments[-1] if self.segments else None
        if last is None or last.label == "walking":
            self._seg("standing", extra, min(0.4, extra))
            return
        self.segments[-1] = last.model_copy(update={"duration": round(last.duration + extra, 3)})
        self.t = round(self.t + extra, 3)


def _pick_goal(actor: Actor) -> Goal:
    weights = dict(actor.persona.goals)
    if actor.state in ("sitting", "lying"):
        weights["sit"] = weights["rest"] = 0.0  # get up and do something else first
    goals = [g for g, w in weights.items() if w > 0]
    return actor.rng.choices(goals, [weights[g] for g in goals])[0]


def _floor_of(plan: FloorPlan, seg: Segment) -> Floor:
    assert seg.x is not None and seg.depth is not None
    return to_world(plan.scene, seg.x, seg.depth)


def _length(route: list[Floor]) -> float:
    return sum(distance(a, b) for a, b in zip(route, route[1:]))


def _cut(route: list[Floor], length: float) -> list[Floor]:
    """The first `length` metres of a route."""
    out = [route[0]]
    for a, b in zip(route, route[1:]):
        d = distance(a, b)
        if d >= length:
            k = length / d if d else 0.0
            out.append((a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k))
            return out
        out.append(b)
        length -= d
    return out
