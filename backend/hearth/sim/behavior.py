"""Resident behaviour: goals, personas and faults, compiled into ordinary segments.

An Actor is one resident being planned. It keeps where they are, which way they face,
what they're doing and the clock, and each action (walk somewhere, sit, lie down, bend,
fall, leave) appends segments to their script. live() picks goals from the persona's
weights until the scenario's time is used up. All of this happens before the run: the
result is a plain Resident, which the generator plays like any hand-written one.

People behave like people, roughly: they go to places (a chair, the window, beside the
bed; FloorPlan.places()) rather than to random spots, turn round before walking off,
stay a while once they've sat down, walk at an older adult's pace, and keep a little
distance from each other, waiting for someone to pass before going on.

Several residents share a Household: the floor plan, which seats are taken when, and
(see hearth.sim.generate) where everyone planned so far will be.

Randomness comes only from the rng each Actor is given, so the same seed always gives the
same script.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from hearth.sim.floorplan import Floor, FloorPlan, Place, Seat, distance, route_segments, to_frame
from hearth.sim.generator import ResidentTrack, walk_time
from hearth.sim.motion import length as motion_length
from hearth.sim.motion import timeline
from hearth.sim.pose import hips_of
from hearth.sim.scenario import Motion, Resident, Segment, TagValue
from hearth.sim.world import PERSON_HEIGHT_M

State = Literal["standing", "sitting", "lying", "fallen", "absent"]
Goal = Literal["sit", "rest", "pick_up", "idle", "visit", "leave", "browse"]
FallKind = Literal["trip", "sit_miss", "stand_up", "faint"]
FallOutcome = Literal["long_lie", "recover"]

FALL_KINDS: tuple[FallKind, ...] = ("trip", "sit_miss", "stand_up", "faint")
LONG_LIE_MIN_S = 15.0
PERSONAL_SPACE_M = 0.8  # closest two people on their feet come, body centre to body centre
LYING_SPACE_M = 0.5  # closest anyone comes to someone lying on the floor (to the nearest point of them)
LYING_HALF_LENGTH_M = 0.8  # someone lying reaches this far either side of their hips, along their body
CLASH_STEP_S = 0.1
LOOKAHEAD_S = 2.5
ESCAPE_S = 2.0
CONTACT_M = 0.45  # body centres this far apart is a bump
TURN_RATE = math.radians(150)  # how fast someone turns on the spot, radians per second
TURN_BEFORE_WALKING = math.radians(45)  # turns bigger than this happen on the spot, before setting off
DETOUR = 1.8  # a route this much longer than the straight line isn't worth taking for a stroll
CORNER_S = 0.25  # extra seconds a sharp corner costs

Allow = Mapping[str, tuple[float, float]]
"""Residents someone may come close to, and when: resident id -> (from, until) scenario seconds."""


@dataclass(frozen=True)
class Persona:
    name: str
    speed: tuple[float, float]  # walking speed, m/s
    height: tuple[float, float]  # relative to Scene.person_height
    slowness: float  # scales how long turning, getting up and sitting down take
    fall_weight: float  # relative chance of being the one who falls
    goals: dict[Goal, float]


PERSONAS: dict[str, Persona] = {
    "steady": Persona("steady", (0.7, 1.0), (0.95, 1.06), 1.0, 1.0,
                      {"sit": 3, "rest": 1, "pick_up": 1, "idle": 3, "visit": 2, "leave": 0.5, "browse": 1}),
    "frail": Persona("frail", (0.35, 0.6), (0.9, 1.0), 1.6, 2.5,
                     {"sit": 4, "rest": 3, "pick_up": 0.5, "idle": 2, "visit": 0.5, "leave": 0.3, "browse": 0.3}),
    "restless": Persona("restless", (0.8, 1.1), (0.95, 1.08), 0.8, 0.8,
                        {"sit": 1, "rest": 0.3, "pick_up": 1.5, "idle": 2, "visit": 4, "leave": 1.5, "browse": 1.5}),
}  # fmt: skip


@dataclass(frozen=True)
class FallPlan:
    kind: FallKind
    at: float  # scenario seconds; the fall happens at the first chance after this
    outcome: FallOutcome


@dataclass(frozen=True)
class Meeting:
    """Where and when two people walk into each other (see Actor.walk_into)."""

    t: float  # the moment they meet
    at: Floor  # where the first one is then
    travel: float  # the way the first one is walking
    stop: Floor  # where the second one is then, just short of the first
    theirs: float  # the way the second one is walking


@dataclass(frozen=True)
class FallEvent:
    t: float  # when the fall starts
    land: Floor  # where the hips end up
    heading: float  # the body's heading on the floor (its head is along it, or opposite)


@dataclass
class Household:
    """What residents share while they're planned: the room, its seats, and each other.

    Residents are planned one at a time. Everyone planned so far is fixed (`planned`);
    the one being planned must keep out of their way (clash()).
    """

    plan: FloorPlan
    bookings: dict[int, list[tuple[float, float]]] = field(default_factory=dict)  # seat index -> busy intervals
    planned: dict[str, ResidentTrack] = field(default_factory=dict)
    fallen: dict[str, Actor] = field(default_factory=dict)  # people planned to fall and stay down

    def seat_free(self, seat: Seat, t0: float, t1: float) -> bool:
        return all(b1 <= t0 or t1 <= b0 for b0, b1 in self.bookings.get(seat.index, []))

    def book(self, seat: Seat, t0: float, t1: float) -> None:
        self.bookings.setdefault(seat.index, []).append((t0, t1))

    def settle(self, actor: Actor) -> None:
        """Fix an actor's plan; everyone planned after them works around it."""
        if actor.seat is not None:  # still seated at the end: the seat stays taken
            self.book(actor.seat, actor.seated_since, math.inf)
        self.planned[actor.resident_id] = ResidentTrack(actor.resident(), self.plan.scene)
        if actor.state == "fallen" and actor.fall_events:
            self.fallen[actor.resident_id] = actor

    def body(self, track: ResidentTrack, t: float) -> Body | None:
        """Where a body is on the floor at t; None if it is out of everyone's way."""
        i = track.segment_index(t)
        p = track.plans[i]
        seg = p.segment
        if seg.label in ("absent", "sitting") or (seg.label == "lying_intentional" and seg.surface_y is not None):
            return None  # out of the room, or on furniture that walkers already avoid
        lying = seg.label in ("fall", "lying_intentional") and t - p.t_start > 0.5
        return Body(track.root_at(t), LYING_HALF_LENGTH_M if lying else 0.0, p.heading)

    def clash(self, actor: Actor, t0: float, t1: float, *, allow: Allow | None = None) -> float | None:
        """First time in [t0, t1] the actor comes too close to anyone already planned, or None."""
        allow = allow or {}
        others = [(rid, tr) for rid, tr in self.planned.items() if rid != actor.resident_id]
        if not others or t1 <= t0:
            return None
        mine = ResidentTrack(actor.resident(), self.plan.scene)
        steps = max(1, math.ceil((t1 - t0) / CLASH_STEP_S))
        # Someone already too close may move away (that's how they get clear), but not closer,
        # and must be clear within ESCAPE_S.
        closest: dict[str, float] = {}
        for k in range(steps + 1):
            t = t0 + (t1 - t0) * k / steps
            me = self.body(mine, t)
            for rid, other in others:
                window = allow.get(rid)
                if window is not None and window[0] <= t <= window[1]:
                    continue
                them = self.body(other, t) if me is not None else None
                if them is None:
                    closest.pop(rid, None)
                    continue
                g = gap(me, them)  # type: ignore[arg-type]
                if g >= (LYING_SPACE_M if me.half or them.half else PERSONAL_SPACE_M):  # type: ignore[union-attr]
                    closest.pop(rid, None)
                    continue
                if k == 0 or (rid in closest and g >= closest[rid] - 0.02 and t - t0 <= ESCAPE_S):
                    closest[rid] = min(g, closest.get(rid, g))
                    continue
                return t
        return None


    def closest(self, actor: Actor, t0: float, t1: float, *, allow: Allow | None = None) -> float:
        """The closest the actor comes to anyone already planned in [t0, t1] (metres; inf if nobody)."""
        allow = allow or {}
        mine = ResidentTrack(actor.resident(), self.plan.scene)
        best = math.inf
        steps = max(1, math.ceil((t1 - t0) / CLASH_STEP_S))
        for k in range(steps + 1):
            t = t0 + (t1 - t0) * k / steps
            me = self.body(mine, t)
            if me is None:
                continue
            for rid, other in self.planned.items():
                window = allow.get(rid)
                if rid == actor.resident_id or (window is not None and window[0] <= t <= window[1]):
                    continue
                them = self.body(other, t)
                if them is not None:
                    best = min(best, gap(me, them) + (PERSONAL_SPACE_M - LYING_SPACE_M if me.half or them.half else 0.0))
        return best


@dataclass(frozen=True)
class Body:
    """A body on the floor: a point, or a line `half` metres either side of it along `heading`."""

    at: Floor
    half: float
    heading: float

    def ends(self) -> tuple[Floor, Floor]:
        dx, dz = math.sin(self.heading) * self.half, math.cos(self.heading) * self.half
        return (self.at[0] - dx, self.at[1] - dz), (self.at[0] + dx, self.at[1] + dz)


def gap(a: Body, b: Body) -> float:
    """Distance between two bodies (points or lines)."""
    return _segment_distance(*a.ends(), *b.ends())


def _point_segment(p: Floor, a: Floor, b: Floor) -> float:
    dx, dz = b[0] - a[0], b[1] - a[1]
    n = dx * dx + dz * dz
    k = 0.0 if n < 1e-12 else min(max(((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / n, 0.0), 1.0)
    return math.hypot(p[0] - a[0] - k * dx, p[1] - a[1] - k * dz)


def _segment_distance(a0: Floor, a1: Floor, b0: Floor, b1: Floor) -> float:
    def cross(o: Floor, p: Floor, q: Floor) -> float:
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    d1, d2, d3, d4 = cross(b0, b1, a0), cross(b0, b1, a1), cross(a0, a1, b0), cross(a0, a1, b1)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return 0.0  # they cross
    return min(_point_segment(a0, b0, b1), _point_segment(a1, b0, b1),
               _point_segment(b0, a0, a1), _point_segment(b1, a0, a1))  # fmt: skip


def depth_bin(depth: float) -> str:
    return "front" if depth < 1 / 3 else "middle" if depth < 2 / 3 else "back"


def heading_to(a: Floor, b: Floor) -> float:
    return math.atan2(b[0] - a[0], b[1] - a[1])


def turn(a: float, b: float) -> float:
    """The short way round from heading a to heading b."""
    return (b - a + math.pi) % (2.0 * math.pi) - math.pi


def ahead(p: Floor, heading: float, d: float) -> Floor:
    return p[0] + math.sin(heading) * d, p[1] + math.cos(heading) * d


class Actor:
    def __init__(
        self, home: Household, resident_id: str, persona: Persona, rng: random.Random, *,
        height: float | None = None, speed: float | None = None, dwell: float = 1.0,
    ) -> None:  # fmt: skip
        self.home = home
        self.plan = home.plan
        self.resident_id = resident_id
        self.persona = persona
        self.rng = rng
        self.height = round(height if height is not None else rng.uniform(*persona.height), 3)
        self.speed = round(speed if speed is not None else rng.uniform(*persona.speed), 3)
        self.dwell = dwell  # scales how long people stay put (longer scenarios, longer stays)
        self.segments: list[Segment] = []
        self.t = 0.0
        self.state: State = "standing"
        self.seat: Seat | None = None
        self.seated_since = 0.0
        self.pos: Floor = (0.0, 0.0)
        self.heading = 0.0
        self.lying_heading = 0.0  # on the floor: which way the head points
        self.start: tuple[Floor, float] = (self.pos, self.heading)
        self.falls: list[Segment] = []
        self.fall_events: list[FallEvent] = []
        self.allow: dict[str, tuple[float, float]] = {}  # who this one may come close to, and when
        self.clashes = 0  # actions dropped because they'd have walked into someone
        self.unresolved = 0  # clashes it couldn't avoid

    # --- setup and output ---

    def place(self, pos: Floor, *, heading: float = 0.0) -> None:
        self.pos, self.heading = pos, heading
        self.start = (pos, heading)

    def resident(self) -> Resident:
        (x, depth), heading = to_frame(self.plan.scene, self.start[0]), self.start[1]
        return Resident(resident_id=self.resident_id, height=self.height, start_x=round(x, 4),
                        start_depth=round(min(max(depth, 0.0), 1.0), 4), start_heading=round(heading, 4),
                        segments=self.segments)  # fmt: skip

    def _u(self, lo: float, hi: float) -> float:
        return self.rng.uniform(lo, hi)

    def _seg(self, label: str, duration: float, transition: float = 0.0, *, at: Floor | None = None, **kw) -> Segment:
        transition = round(max(transition, 0.0), 3)
        duration = round(max(duration, transition, 0.1), 3)
        if at is not None:
            x, depth = to_frame(self.plan.scene, at)
            kw |= {"x": round(x, 4), "depth": round(min(max(depth, 0.0), 1.0), 4)}
        if "heading" in kw and kw["heading"] is not None:
            kw["heading"] = round(kw["heading"], 4)
        seg = Segment(label=label, duration=duration, transition=transition, **kw)  # type: ignore[arg-type]
        self.segments.append(seg)
        self.t = round(self.t + duration, 3)
        return seg

    def _variant(self) -> int:
        return self.rng.randrange(1_000_000)

    # --- moving ---

    def turn_to(self, heading: float) -> None:
        """Turn on the spot to face `heading`, if it is far enough round to need it."""
        delta = abs(turn(self.heading, heading))
        if delta > TURN_BEFORE_WALKING:
            took = max(0.4, delta / TURN_RATE * self.persona.slowness)
            self._seg("standing", took, took, heading=heading)
        self.heading = heading

    def get_up(self) -> None:
        slow = self.persona.slowness
        if self.state == "absent":
            return
        if self.state == "lying":
            tr = self._u(2.0, 3.0) * slow
            self._seg("sitting", tr + self._u(0.5, 1.5), tr, heading=0.0)
            self.state, self.heading = "sitting", 0.0
        if self.state == "fallen":
            variant = self._variant()
            took = motion_length("get_up", variant) * slow
            # Get up facing the way the head was: rolling over onto hands and knees.
            self._seg("standing", took + self._u(0.5, 1.5), 0.0, motion="get_up", variant=variant,
                      heading=self.lying_heading)  # fmt: skip
            self.state, self.heading = "standing", self.lying_heading
        if self.state == "sitting":
            tr = self._u(1.0, 1.6) * slow
            out = self.plan.approach(self.seat) if self.seat is not None else self.pos
            self._seg("standing", tr + self._u(0.5, 1.2), tr, at=out)  # up, and a step out of the seat
            self.state, self.pos = "standing", out
        if self.seat is not None:
            self.home.book(self.seat, self.seated_since, self.t)
            self.seat = None

    def _route(self, goal: Floor, **kw) -> list[Floor] | None:
        """Route from wherever the actor will stand up (out of a seat first) to goal."""
        start = self.plan.approach(self.seat) if self.seat is not None else self.pos
        return self.plan.find_path(start, goal, **kw)

    def walk(self, route: list[Floor], *, stops: bool = True, quick: bool = False) -> None:
        """Walk a route (route[0] must be where the actor is, once up). quick: a step
        aside, turning while stepping rather than first."""
        self.get_up()
        legs = [(a, b) for a, b in zip(route, route[1:]) if distance(a, b) >= 0.01]
        if not legs:
            return
        if not quick:
            self.turn_to(heading_to(*legs[0]))
        segs = route_segments(self.plan.scene, route, self.speed, first_transition=0.4)
        length = sum(distance(a, b) for a, b in legs)
        sharp = sum(abs(turn(heading_to(*p), heading_to(*q))) > math.radians(60) for p, q in zip(legs, legs[1:]))
        took = walk_time(length, self.speed, stops=stops) + CORNER_S * sharp
        scale = took / sum(s.duration for s in segs)
        for seg in segs:
            duration = round(seg.duration * scale, 3)
            self.segments.append(seg.model_copy(update={"duration": duration, "transition": min(seg.transition, duration)}))
            self.t = round(self.t + duration, 3)
        self.pos = route[-1]
        self.heading = heading_to(*legs[-1])
        self.state = "standing"

    def go(self, goal: Floor, *, detour: float | None = None, stops: bool = True, quick: bool = False) -> bool:
        if self.state == "absent":
            return False
        route = self._route(goal)
        if route is None:
            return False
        if detour is not None and _length(route) > detour * distance(route[0], route[-1]) + 0.5:
            return False  # not worth the trek
        self.walk(route, stops=stops, quick=quick)
        return True

    def go_sit(self, seat: Seat) -> bool:
        if not self.go(self.plan.approach(seat)):
            return False
        self.turn_to(0.0)  # back to the seat
        return True

    # --- activities ---

    def stand(self, duration: float, heading: float | None = None) -> None:
        self.get_up()
        if heading is not None:
            self.turn_to(heading)
        self._seg("standing", duration, min(0.4, duration), heading=self.heading)

    def settle(self) -> None:
        """A moment's pause, as people take before or after doing something."""
        self._seg("standing", self._u(0.5, 1.5) * self.persona.slowness, 0.3)

    def sit(self, seat: Seat, duration: float, *, fast: bool = False) -> None:
        slow = self.persona.slowness
        tr = self._u(0.3, 0.45) if fast else self._u(0.7, 1.1) * slow
        tags: dict[str, TagValue] = {"hard_negative": "fast_sit"} if fast else {}
        self.heading = 0.0
        self._seg("sitting", duration + tr, tr, at=seat.at(self.plan.scene), heading=0.0, tags=tags)
        self.state, self.seat, self.seated_since = "sitting", seat, self.t - duration - tr
        self.pos = seat.at(self.plan.scene)

    def lie(self, seat: Seat, duration: float) -> None:
        """Lie down on a couch or bed (sitting on it first) or on a mat."""
        slow = self.persona.slowness
        since = self.t
        if seat.kind in ("couch", "bed"):
            self.sit(seat, self._u(1.5, 4.0))
            since = self.seated_since
        tr = self._u(2.5, 3.5) * slow
        heading = self.rng.choice([math.pi / 2, -math.pi / 2])
        self._seg("lying_intentional", duration + tr, tr, heading=heading, surface_y=seat.surface_y,
                  tags={"hard_negative": "lie_down", "on": seat.kind})  # fmt: skip
        self.state, self.seat, self.seated_since, self.heading = "lying", seat, since, heading

    def bend(self, duration: float) -> None:
        self.get_up()
        self._seg("bending", duration, self._u(0.8, 1.2) * self.persona.slowness, tags={"hard_negative": "bend"})
        self._seg("standing", self._u(1.0, 2.0), 0.9)

    def leave(self, door: str, away: float, come_back: str | None) -> bool:
        if not self.go(self.plan.doors[door]):
            return False
        back = self.plan.doors[come_back or door]
        self._seg("absent", away, at=back)
        self.pos = back
        if come_back is not None:  # comes back in facing into the room
            self.heading = heading_to(back, (0.0, back[1]))
        self.state = "absent" if come_back is None else "standing"
        return True

    def browse(self, spot: Floor, top: float, duration: float) -> bool:
        """Stand behind a tall piece of furniture, partly hidden from the camera."""
        if not self.go(spot):
            return False
        self.turn_to(0.0)
        self._seg("occluded", duration, 0.4, heading=0.0, occluder_top=top, tags={"hard_negative": "occluded"})
        return True

    def visit(self, place: Place, duration: float) -> bool:
        """Go to a place and spend a moment there, facing what is there."""
        if distance(self.pos, place.at) > 0.3 and not self.go(place.at, detour=DETOUR):
            return False
        self.stand(duration, place.heading)
        return True

    # --- falls ---

    def can_land(self, p: Floor, heading: float, motion: Motion, variant: int) -> bool:
        """Is there floor for this body to end up on, here, in the camera's view?"""
        if not (self.plan.free(p) and self.plan.visible(p, margin=0.1)):
            return False
        end = timeline(motion, variant)[-1].pose
        size = PERSON_HEIGHT_M * self.height
        hips = hips_of(end)
        for joint in ("nose", "left_ankle", "right_ankle", "left_wrist", "right_wrist", "left_knee", "right_knee"):
            f, _, l = end[joint]
            f, l = (f - hips[0]) * size, (l - hips[2]) * size
            q = (p[0] + f * math.sin(heading) + l * math.cos(heading), p[1] + f * math.cos(heading) - l * math.sin(heading))
            # Limbs may rest against furniture, but not inside it or through a wall.
            if not self.plan.free(q, margin=-0.2) or not self.plan.visible(q, margin=-0.1):
                return False
        return True

    def fall(self, kind: Motion, outcome: FallOutcome, until: float, *, land: Floor, heading: float, variant: int,
             fault: str | None = None, tags: dict[str, TagValue] | None = None) -> None:  # fmt: skip
        """Fall, ending up with the hips at `land` and the body along `heading`."""
        x, depth = to_frame(self.plan.scene, land)
        floor_time = max(until - self.t, LONG_LIE_MIN_S) if outcome == "long_lie" else self._u(3.0, 10.0)
        took = motion_length(kind, variant)
        tags = {
            "fault": fault or kind, "outcome": outcome, "depth_bin": depth_bin(depth), "persona": self.persona.name,
            "facing": _facing_name(self.heading), **(tags or {}),
        }  # fmt: skip
        self.fall_events.append(FallEvent(self.t, land, heading))
        seg = self._seg("fall", max(floor_time, took + 1.0), 0.0, at=land, heading=heading, motion=kind,
                        variant=variant, tags=tags)  # fmt: skip
        self.falls.append(seg)
        if self.seat is not None:
            self.home.book(self.seat, self.seated_since, self.t)
        end = timeline(kind, variant)[-1].pose
        head_first = end["nose"][0] >= hips_of(end)[0]
        self.lying_heading = heading if head_first else heading + math.pi
        self.pos, self.state, self.seat, self.heading = land, "fallen", None, heading

    def do_fall(self, fall: FallPlan, until: float) -> bool:
        """Carry out a planned fall. False if this room offers no way to do it right now."""
        u, variant = self._u, self._variant()
        if fall.kind == "trip":
            # On the way somewhere: down mid-stride, sprawling forwards.
            for _ in range(12):
                route = self._route(self._destination())
                if route is None or _length(route) < 1.5:
                    continue
                cut = _cut(route, u(0.35, 0.8) * _length(route))
                travel = heading_to(cut[-2], cut[-1])
                heading = travel + math.radians(u(-25, 25))
                land = ahead(cut[-1], travel, u(0.35, 0.7))
                if not self.can_land(land, heading, "trip", variant):
                    continue
                self.walk(cut, stops=False)
                self.fall("trip", fall.outcome, until, land=land, heading=heading, variant=variant)
                return True
            return False
        if fall.kind == "sit_miss":
            seat = self.free_seat(("chair", "couch"), 10.0)
            if seat is None:
                return False
            front = self.plan.approach(seat)
            land = ahead(front, math.pi, 0.1)  # a little back, towards where the seat should have been
            if not self.can_land(land, 0.0, "sit_miss", variant) or not self.go_sit(seat):
                return False
            self.fall("sit_miss", fall.outcome, until, land=land, heading=0.0, variant=variant)
            return True
        if fall.kind == "stand_up":
            if self.state not in ("sitting", "lying") or self.seat is None:
                seat = self.free_seat(("chair", "couch"), 20.0)
                if seat is None or not self.go_sit(seat):
                    return False
                self.sit(seat, u(4.0, 10.0) * self.dwell)
            assert self.seat is not None
            front = self.plan.approach(self.seat)
            sides = [math.pi / 2, -math.pi / 2, 0.0]
            self.rng.shuffle(sides)
            heading = next((h for h in sides if self.can_land(front, h, "stand_up", variant)), None)
            if heading is None:
                return False
            if self.state == "lying":
                tr = u(2.0, 3.0) * self.persona.slowness
                self._seg("sitting", tr + 0.5, tr, heading=0.0)
            self.heading = 0.0
            self.fall("stand_up", fall.outcome, until, land=front, heading=heading, variant=variant)
            return True
        # faint: wherever they are standing, or at a place they go to; down off to one side
        off = math.radians(u(20, 100)) * self.rng.choice([-1, 1])
        if self.state != "standing" or not self.can_land(self.pos, self.heading + off, "faint", variant):
            spots = [p for p in self._places() if self.can_land(p.at, p.heading + off, "faint", variant)]
            if not spots:
                return False
            spot = self.rng.choice(spots)
            if not self.go(spot.at):
                return False
            self.turn_to(spot.heading)
        self._seg("standing", u(1.0, 3.0), 0.4)
        self.fall("faint", fall.outcome, until, land=self.pos, heading=self.heading + off, variant=variant)
        return True

    # --- bumping into each other ---

    def walk_into(self) -> Meeting | None:
        """Walk to an open spot, arriving at speed: the first person in a collision.

        Picks where they'll meet and the way the other person will come from, so that
        person can be planned to arrive there at the same moment (approach()).
        """
        u = self._u
        spots = [p.at for p in self._places() if p.name == "middle"]
        spots += [self.plan.random_point(self.rng) for _ in range(12)]
        self.rng.shuffle(spots)
        for spot in spots:
            if self.plan.clearance(spot) < 0.25 or not self.plan.visible(spot, margin=0.15):
                continue
            route = self._route(spot)
            if route is None or _length(route) < 1.2:
                continue
            travel = heading_to(route[-2], route[-1])
            for side in (math.radians(u(70, 110)) * self.rng.choice([-1, 1]) for _ in range(3)):
                theirs = travel + math.pi + side  # they come from off to one side, walking across
                stop = ahead(spot, theirs + math.pi, CONTACT_M)
                if self.plan.free(stop) and self.plan.line_free(stop, ahead(stop, theirs + math.pi, 1.2)):
                    self.walk(route, stops=False)
                    return Meeting(self.t, spot, travel, stop, theirs)
        return None

    def approach(self, meeting: Meeting, other: str) -> bool:
        """Be at meeting.stop, walking along meeting.theirs, at exactly meeting.t."""
        heading, stop = meeting.theirs, meeting.stop
        self.allow[other] = (meeting.t - 1.0, meeting.t + 4.0)
        # Set off from a little way back along the way they'll come.
        starts = [(r, ahead(stop, heading + math.pi, r)) for r in (2.2, 1.7, 1.2)]
        starts = [(r, p) for r, p in starts if self.plan.free(p) and self.plan.line_free(p, stop)]
        if not starts:
            return False

        def lead() -> float:
            """Seconds needed from here to the farthest start and on to the meeting."""
            run_up, start = starts[0]
            route = self._route(start)
            if route is None:
                return math.inf
            return walk_time(_length(route), self.speed) + 2.5 * self.persona.slowness + 1.0 +                 walk_time(run_up, self.speed, stops=False)

        for _ in range(60):  # pass the time, keeping out of the way, until it's time to go
            spare = meeting.t - lead() - self.t
            if spare < 1.0:
                break
            if not self._try(lambda: self._stand_for(min(3.0, spare))):
                self.wait()
        for run_up, start in starts:
            depart = meeting.t - walk_time(run_up, self.speed, stops=False)

            def go_there(start: Floor = start, depart: float = depart) -> bool:
                if not self.go(start):
                    return False
                self.turn_to(heading)
                if self.t > depart - 0.2:
                    return False  # too late
                self._seg("standing", depart - self.t, 0.3, heading=heading)
                self.walk([start, stop], stops=False, quick=True)
                return abs(self.t - meeting.t) < 0.05

            if self._try(go_there):
                return True
        return False

    def bumped(self, push: float, falls: bool, outcome: FallOutcome, until: float, *, other: str,
               result: str) -> bool:  # fmt: skip
        """Knocked along heading `push`: stagger and catch it, or go down. True if they fell."""
        variant = self._variant()
        tags: dict[str, TagValue] = {"bumped": other, "collision_outcome": result}
        if falls:
            heading = push + math.radians(self._u(-30, 30))
            for d in (0.5, 0.35, 0.2):
                land = ahead(self.pos, push, d)
                if self.can_land(land, heading, "collision", variant):
                    self.fall("collision", outcome, until, land=land, heading=heading, variant=variant,
                              fault="collision", tags=tags)  # fmt: skip
                    return True
        took = motion_length("stumble", variant) * self.persona.slowness
        to = ahead(self.pos, push, 0.25)
        if not self.plan.free(to):
            to = self.pos
        self._seg("standing", took + self._u(0.8, 2.0), 0.0, at=to, heading=self.heading, motion="stumble",
                  variant=variant, tags={"hard_negative": "bump", **tags})  # fmt: skip
        self.pos = to
        return False

    # --- choosing what to do ---

    def _places(self) -> list[Place]:
        return [p for p in self.plan.places() if not p.name.startswith("on-")]

    def _destination(self) -> Floor:
        places = self._places()
        if places and self.rng.random() < 0.85:
            return self.rng.choice(places).at
        return self.plan.random_point(self.rng)

    def free_seat(self, kinds: tuple[str, ...], stay: float) -> Seat | None:
        seats = [s for s in self.plan.seats(kinds) if s != self.seat]  # type: ignore[arg-type]
        seats = [s for s in seats if self.home.seat_free(s, self.t, self.t + 20.0 + stay)]
        return self.rng.choice(seats) if seats else None

    def do_goal(self, goal: Goal, until: float) -> bool:
        u, left = self._u, until - self.t
        if goal == "sit":
            stay = min(u(15.0, 60.0) * self.dwell * self.persona.slowness, max(left, 5.0))
            seat = self.free_seat(("chair", "couch"), stay)
            if seat is None or not self.go_sit(seat):
                return False
            self.sit(seat, stay, fast=self.rng.random() < 0.2)
            return True
        if goal == "rest":
            stay = min(u(30.0, 120.0) * self.dwell, max(left, 10.0))
            seat = self.free_seat(("couch", "bed", "mat"), stay)
            if seat is None:
                return False
            if seat.kind == "mat":
                if not self.go(self.plan.approach(seat)):
                    return False
                self.settle()
            elif not self.go_sit(seat):
                return False
            self.lie(seat, stay)
            return True
        if goal == "pick_up":
            near = [p for p in self._places() if p.near is not None]
            if not near:
                return False
            place = self.rng.choice(near)
            if not self.visit(place, u(0.5, 1.5)):
                return False
            self.bend(u(1.5, 4.0))
            return True
        if goal == "idle":
            places = self._places()
            if not places:
                return False
            place = min(places, key=lambda p: distance(p.at, self.pos)) if self.rng.random() < 0.5 else self.rng.choice(places)
            return self.visit(place, u(4.0, 12.0))
        if goal == "visit":
            places = [p for p in self._places() if distance(p.at, self.pos) > 1.0]
            if not places:
                return False
            return self.visit(self.rng.choice(places), u(2.0, 6.0))
        if goal == "leave":
            if left < 20.0:
                return False
            door = min(self.plan.doors, key=lambda d: distance(self.pos, self.plan.doors[d]))
            return self.leave(door, u(5.0, 15.0), self.rng.choice(list(self.plan.doors)))
        if goal == "browse":
            spots = self.plan.hiding_spots()
            if not spots:
                return False
            spot, top = self.rng.choice(spots)
            return self.browse(spot, top, u(5.0, 12.0))
        raise ValueError(goal)

    def live(self, until: float, fall: FallPlan | None = None, *, pick: Callable[[Actor], Goal] | None = None) -> None:
        """Pursue goals until `until` (scenario seconds), carrying out the planned fall on the way."""
        pending, failures = fall, 0
        retry: Goal | None = None
        for _ in range(200):
            if self.t >= until - 1.0 and pending is None:
                break
            if pending is not None and self.t >= pending.at:
                if self._try(lambda: self.do_fall(pending, until)):
                    if pending.outcome == "long_lie":
                        break
                    self.get_up()  # after their while on the floor, however little time is left
                    pending = None
                    continue
                # Try again a little later. If this room keeps making it impossible (seats taken,
                # no floor in view), fall another way: a trip, then simply going down.
                failures += 1
                kind = pending.kind if failures < 3 else "trip" if failures < 5 else "faint"
                pending = FallPlan(kind, self.t + 2.0, pending.outcome) if failures < 8 else None
            goal = retry or (pick or _pick_goal)(self)
            clashes = self.clashes
            if self._try(lambda: self.do_goal(goal, until)):
                retry = None
                continue
            # Someone was in the way: let them pass, then try the same thing once more.
            retry = goal if self.clashes > clashes and retry is None else None
            self.wait()
        self.hold_clear(until)

    # --- keeping out of each other's way ---

    def _snapshot(self) -> tuple:
        return (list(self.segments), self.t, self.state, self.seat, self.seated_since, self.pos, self.heading,
                self.lying_heading, list(self.falls), list(self.fall_events),
                {k: list(v) for k, v in self.home.bookings.items()})  # fmt: skip

    def _restore(self, snap: tuple) -> None:
        (self.segments, self.t, self.state, self.seat, self.seated_since, self.pos, self.heading, self.lying_heading,
         self.falls, self.fall_events, self.home.bookings) = snap  # fmt: skip

    def _try(self, action: Callable[[], bool]) -> bool:
        """Do an action, or undo it if it fails or would walk into someone."""
        snap = self._snapshot()
        if not action():
            self._restore(snap)
            return False
        # Look a little past the end too: finishing in someone's path leaves no time to get out of it.
        if self.home.clash(self, snap[1], self.t + LOOKAHEAD_S, allow=self.allow) is not None:
            self._restore(snap)
            self.clashes += 1
            return False
        return True

    def wait(self) -> None:
        """Stay put a moment; if someone is coming through here, step out of their way.

        If nothing keeps clear of everyone (someone planned earlier walks right through
        here), do whatever keeps furthest from them, and count it as unresolved.
        """
        if self._help_nearby():
            return
        duration = self._u(1.0, 3.0)
        options: list[Callable[[], bool]] = [lambda: self._stand_for(duration)]
        x, z = self.pos
        for r in (0.6, 1.0, 1.6):  # step aside, a little further each time
            first = self._u(0.0, math.pi / 4)
            for k in range(8):
                a = first + k * math.pi / 4
                spot = (x + r * math.cos(a), z + r * math.sin(a))
                if self.plan.free(spot):
                    options.append(lambda spot=spot: self.go(spot))
        places = self._places()
        self.rng.shuffle(places)
        options += [lambda p=p: self.go(p.at) for p in places[:6]]
        spots = [self.plan.random_point(self.rng, visible=False) for _ in range(4)]
        options += [lambda p=p: self.go(p) for p in spots]
        for option in options:
            if self._try(option):
                return
        best: tuple[float, Callable[[], bool]] | None = None
        for option in options:
            snap = self._snapshot()
            if option():
                g = self.home.closest(self, snap[1], self.t + LOOKAHEAD_S, allow=self.allow)
                if best is None or g > best[0]:
                    best = (g, option)
            self._restore(snap)
        self.unresolved += 1
        if best is None or not best[1]():
            self._stand_for(duration)

    def _help_nearby(self) -> bool:
        """Someone close by is down on the floor: go and kneel beside them, as anyone would."""
        if any(seg.tags.get("responder") for seg in self.segments) or self.state in ("fallen", "absent"):
            return False
        for rid, faller in self.home.fallen.items():
            ev = faller.fall_events[0]
            if rid not in self.allow and ev.t <= self.t + 1.0 and distance(self.pos, ev.land) < 2.0:
                if self.respond(faller, self.t + 20.0):
                    return True
        return False

    def _stand_for(self, duration: float) -> bool:
        self.stand(duration)
        return True

    def hold_clear(self, until: float) -> None:
        """hold(), but keep out of the way of anyone who comes past in the meantime."""
        if self.state == "fallen":  # still on the floor: they can't
            self.hold(until)
            return
        snap = self._snapshot()
        self.hold(until)
        if self.home.clash(self, snap[1], self.t, allow=self.allow) is None:
            return
        self._restore(snap)
        for _ in range(100):
            if self.t >= until - 0.5:
                break
            self.wait()
        self.hold(until)

    def respond(self, faller: Actor, until: float) -> bool:
        """Hurry to someone who has fallen and kneel beside them."""
        if not faller.fall_events:
            return False
        ev = faller.fall_events[0]
        head = faller.lying_heading  # kneel by their shoulders, either side, or else at their head or feet
        torso = ahead(ev.land, head, 0.3)
        spots = [ahead(torso, head + side, 0.6) for side in (math.pi / 2, -math.pi / 2)]
        spots += [ahead(ev.land, head, 1.2), ahead(ev.land, head + math.pi, 1.2)]
        speed, self.speed = self.speed, round(min(self.speed * 1.3, 1.4), 3)
        try:
            for spot in spots:
                if not self.plan.free(spot):
                    continue
                # Close to them only at the very end: the way there goes round them, not over them.
                snap = self._snapshot()
                arrived = self.go(spot)
                arrival = self.t
                self._restore(snap)
                if not arrived:
                    continue
                self.allow[faller.resident_id] = (arrival - 0.8, math.inf)
                if self._try(lambda: self._attend(spot, torso, until)):
                    return True
                del self.allow[faller.resident_id]
        finally:
            self.speed = speed
        return False

    def _attend(self, spot: Floor, faller: Floor, until: float) -> bool:
        if not self.go(spot):
            return False
        self.turn_to(heading_to(spot, faller))
        tags: dict[str, TagValue] = {"responder": True}
        self._seg("bending", self._u(4.0, 8.0), self._u(0.9, 1.3), pose="kneel", heading=self.heading, tags=tags)
        self._seg("standing", self._u(2.0, 4.0), 1.0, tags=tags)
        if until - self.t > 6.0:
            self._seg("bending", self._u(3.0, 6.0), 1.0, pose="kneel", tags=tags)
            self._seg("standing", 1.0, 1.0, tags=tags)
        return True

    def hold(self, until: float) -> None:
        """Carry on with the current activity until `until`."""
        extra = round(until - self.t, 3)
        if extra <= 0:
            return
        last = self.segments[-1] if self.segments else None
        if last is None or last.label == "walking" or last.motion is not None and last.label != "fall":
            self._seg("standing", extra, min(0.4, extra))
            return
        self.segments[-1] = last.model_copy(update={"duration": round(last.duration + extra, 3)})
        self.t = round(self.t + extra, 3)


def _facing_name(heading: float) -> str:
    names = {"camera": 0.0, "right": math.pi / 2, "left": -math.pi / 2, "away": math.pi}
    return min(names, key=lambda n: abs(turn(names[n], heading)))


def _pick_goal(actor: Actor) -> Goal:
    weights = dict(actor.persona.goals)
    if actor.state in ("sitting", "lying"):
        weights["sit"] = weights["rest"] = 0.0  # get up and do something else first
    goals = [g for g, w in weights.items() if w > 0]
    return actor.rng.choices(goals, [weights[g] for g in goals])[0]


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

