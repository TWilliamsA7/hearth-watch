"""Incidents: the events Hearth is meant to notice, kept apart from ordinary behaviour.

Three ways in, all through trigger():
  manual       someone presses a button (POST /sim/live/incidents)
  scheduled    EngineSpec.incidents, at a scenario time
  procedural   EngineSpec.incident_rates, per actor-hour; none unless configured

Kinds:
  fall         a choreographed fall (hearth.sim.motion) suited to the posture: a trip or
               faint from standing, a fall while getting up from a seat. The person stays
               on the floor for `duration`, then gets up (or stays down until reset).
  inactivity   holds whatever posture they're in, barely moving, for `duration`.
  pacing       walks back and forth between two points, `repeats` legs.
  exit_loiter  goes to a door, lingers there, wanders off, comes back; `repeats` visits.

Each incident leaves an IncidentRecord: who, when, where, posture before and after, when
they reached the floor, when they recovered. Ground truth for falls is the "fall" label
(truth.py); the record carries the details the label can't.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from hearth.sim.nav import Floor, distance

from hearth.sim.live.activities import (
    FALLEN_CLEARANCE_M,
    Activity,
    Do,
    Hold,
    PlayMotion,
    Transition,
    Turn,
    Walk,
    get_up,
    mobility_pace,
    release,
)
from hearth.sim.live.actor import Actor
from hearth.sim.live.animate import heading_to
from hearth.sim.live.room import forward
from hearth.sim.live.spec import IncidentKind, IncidentSpec

if TYPE_CHECKING:
    from hearth.sim.live.engine import World

FALL_TRAVEL_M = {"trip": 0.55, "faint": 0.15, "collapse": 0.12, "sit_miss": 0.25, "stand_up": 0.6}
DEFAULT_FLOOR_S = 30.0
DEFAULT_INACTIVE_S = 120.0
IncidentSource = Literal["manual", "scheduled", "procedural"]


class IncidentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)

    incident_id: str
    kind: IncidentKind
    source: IncidentSource
    person_id: str
    t_start: float
    location: tuple[float, float]
    posture_before: str
    posture_after: str | None = None
    motion: str | None = None
    t_floor: float | None = None
    t_recovered: float | None = None
    t_end: float | None = None


class IncidentError(ValueError):
    """The incident can't happen to that person right now (say why)."""


@dataclass
class _Due:
    spec: IncidentSpec
    fired: bool = False


class IncidentManager:
    def __init__(self, w: World) -> None:
        self.w = w
        self.records: list[IncidentRecord] = []
        self.scheduled = [_Due(s) for s in sorted(w.spec.incidents, key=lambda s: s.t)]
        self.rng = random.Random(f"incidents/{w.seed}")
        self.responders: dict[str, str] = {}  # person down -> resident coming to help

    def active(self) -> list[IncidentRecord]:
        return [r for r in self.records if r.t_end is None]

    def update(self) -> None:
        w = self.w
        for due in self.scheduled:
            if not due.fired and w.t >= due.spec.t:
                due.fired = True
                try:
                    self.trigger(due.spec, source="scheduled")
                except IncidentError:
                    pass  # nobody suitable at that moment; scheduled incidents are best effort
        for kind, rate in w.spec.incident_rates.items():
            if rate <= 0:
                continue
            p = rate * w.dt / 3600.0
            for a in w.actors:
                if self.rng.random() < p and a.present and not a.incident:
                    try:
                        self.trigger(IncidentSpec(kind=kind, actor=a.person_id, in_view=False), source="procedural")
                    except IncidentError:
                        pass

    def trigger(self, spec: IncidentSpec, *, source: IncidentSource = "manual") -> IncidentRecord:
        w = self.w
        a = w.actor(spec.actor) if spec.actor else self._pick(spec.kind, in_view=spec.in_view)
        if not a.present:
            raise IncidentError(f"{a.person_id} is out of the room")
        if a.incident:
            raise IncidentError(f"{a.person_id} already has a {a.incident} in progress")
        rec = IncidentRecord(incident_id=f"i{len(self.records) + 1}", kind=spec.kind, source=source,
                             person_id=a.person_id, t_start=round(w.t, 3), location=_r(a.pos),
                             posture_before=a.posture)  # fmt: skip
        plan = {"fall": self._fall, "inactivity": self._inactivity, "pacing": self._pacing,
                "exit_loiter": self._exit_loiter}[spec.kind](a, spec, rec)  # fmt: skip
        self.records.append(rec)
        a.incident = spec.kind
        w.replace_plan(a, plan + [Do(lambda a, w, rec=rec: self._finish(a, rec))], spec.kind)
        return rec

    def reset(self, person_id: str) -> None:
        """Back to normal: end any incident; anyone on the floor gets up."""
        w = self.w
        a = w.actor(person_id)
        for rec in self.active():
            if rec.person_id == person_id:
                rec.t_end = round(w.t, 3)
                rec.posture_after = rec.posture_after or a.posture
        a.incident = None
        w.replace_plan(a, get_up(a, w) if a.posture == "floor" else [], "reset")

    def _finish(self, a: Actor, rec: IncidentRecord) -> None:
        if rec.t_end is None:
            rec.t_end = round(self.w.t, 3)
        if rec.posture_after is None:
            rec.posture_after = a.posture
        a.incident = None

    # --- helping someone who is down ---

    def respond(self, target: Actor) -> Actor | None:
        """The nearest free resident (standing ones first; someone seated gets up) walks over and
        kneels beside `target` until they're up. For simulated falls and for a live person the
        monitor says is down."""
        w = self.w
        if target.person_id in self.responders or not target.present:
            return None
        free = [a for a in w.actors if a is not target and a.present and a.posture in ("standing", "sitting")
                and not a.incident and a.intent != "help" and a.person_id not in self.responders.values()]
        if not free:
            return None
        helper = min(free, key=lambda a: (a.posture != "standing", distance(a.pos, target.pos)))
        side = heading_to(target.pos, helper.pos)  # kneel on the side they arrive from
        # Just outside the space everyone keeps round someone on the floor (FALLEN_CLEARANCE_M).
        reach = FALLEN_CLEARANCE_M + 0.1
        goal = (target.pos[0] + reach * math.sin(side), target.pos[1] + reach * math.cos(side))
        down = lambda: not target.present or target.posture != "floor"  # noqa: E731

        def done(a: Actor, w: World) -> None:
            self.responders.pop(target.person_id, None)

        plan: list[Activity] = [
            *get_up(helper, w),  # out of the chair first, if seated
            Walk(goal, purpose="help", stop_short_ok=True, arrive=0.3),
            Turn(heading_to(goal, target.pos)),
            Transition("kneel", 1.0, name="kneeling to help", label="bending", posture="standing"),
            Hold(math.inf, "helping", pose="kneel", label="bending", sway=0.3, until=down),
            Transition("standing", 1.0, name="standing up", label="standing", posture="standing"),
            Do(done),
        ]
        self.responders[target.person_id] = helper.person_id
        w.replace_plan(helper, plan, "help")
        return helper

    def _pick(self, kind: IncidentKind, *, in_view: bool = False) -> Actor:
        ok = [a for a in self.w.actors if a.present and not a.incident and (kind != "fall" or a.posture != "lying")]
        if not ok:
            raise IncidentError("nobody suitable is in the room")
        if in_view and kind == "fall":  # someone the camera can see well
            seen = self.w.observer.visible
            best = max(seen.get(a.person_id, 0.0) for a in ok)
            ok = [a for a in ok if seen.get(a.person_id, 0.0) >= best - 0.1]
        return self.rng.choice(ok)

    # --- kinds ---

    def _fall(self, a: Actor, spec: IncidentSpec, rec: IncidentRecord) -> list[Activity]:
        w = self.w
        if a.posture == "lying":
            raise IncidentError(f"{a.person_id} is lying down; there's nothing to fall from")
        if a.posture == "floor":
            raise IncidentError(f"{a.person_id} is already on the floor")
        start = a.pos
        if a.posture == "sitting":
            motion = spec.motion if spec.motion in ("stand_up",) else "stand_up"
            seat = next((w.room.slot(s) for s in a.slots if w.room.slot(s).kind == "sit"), None)
            if seat is not None:
                start = seat.stand_point(a.size)
        else:
            motion = spec.motion or ("trip" if a.speed > 0.2 else a.rng.choice(("faint", "collapse", "trip")))
            if motion == "stand_up":
                motion = "collapse"
        heading, end = self._landing(a, start, motion, in_view=spec.in_view)
        rec.motion = motion
        pace = 1.0 if a.profile.mobility == "normal" else 1.15

        def on_floor(a: Actor, w: World) -> None:
            rec.t_floor = round(w.t, 3)
            rec.posture_after = "floor"
            release(a, w)  # out of the chair, onto the floor
            self.respond(a)

        plan: list[Activity] = [
            PlayMotion(motion, name="falling", label="fall", posture="floor", end=end, heading=heading, pace=pace),
            Do(on_floor),
            Hold(spec.duration or DEFAULT_FLOOR_S, "on the floor", label="fall", sway=0.0),
        ]
        if spec.recover:

            def recovered(a: Actor, w: World) -> None:
                rec.t_recovered = round(w.t, 3)
                rec.posture_after = "standing"

            plan += [PlayMotion("get_up", name="getting up", label="standing", posture="standing",
                                pace=mobility_pace(a)), Do(recovered)]  # fmt: skip
        else:
            plan[-1] = Hold(math.inf, "on the floor", label="fall", sway=0.0)
        return plan

    def _landing(self, a: Actor, start: Floor, motion: str, *, in_view: bool) -> tuple[float, Floor]:
        """A heading to fall towards with room for a body on the floor, and where the root lands.

        in_view also weighs how well the camera would see the body there: in the picture,
        not behind furniture, and lying across the view rather than along it.
        """
        travel = FALL_TRAVEL_M[motion]
        room, cam = self.w.room, self.w.camera
        offsets = [0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.4, -2.4, math.pi]
        if motion == "stand_up":  # falling out of a chair goes forward-ish
            offsets = [0.0, 0.4, -0.4, 0.8, -0.8]
        best: tuple[float, float, Floor] | None = None
        for off in offsets:
            h = a.heading + off
            f = forward(h)
            end = (start[0] + f[0] * travel, start[1] + f[1] * travel)
            body = [(end[0] + f[0] * k * a.size, end[1] + f[1] * k * a.size)
                    for k in (-0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6)]  # fmt: skip
            clear = sum(room.raw_free(p, margin=0.08) for p in body) / len(body)
            score = clear * 10
            if in_view:
                seen = low = 0
                for p in body:
                    q = cam.project((p[0], 0.15, p[1]))
                    ok = bool(q and 0.03 <= q[0] <= 0.97 and 0.03 <= q[1] <= 0.97 and not cam.blocked((p[0], 0.15, p[1])))
                    seen += ok
                    low += bool(ok and q and q[1] >= 0.5)  # near the camera: low in the picture
                to_cam = math.atan2(cam.position[0] - end[0], cam.position[2] - end[1])
                across = abs(math.sin(h - to_cam))
                score += 8 * seen / len(body) + 3 * low / len(body) + 4 * across
            elif clear == 1.0:
                return h, _r(end)
            if best is None or score > best[0]:
                best = (score, h, end)
        assert best is not None
        return best[1], _r(best[2])

    def _inactivity(self, a: Actor, spec: IncidentSpec, rec: IncidentRecord) -> list[Activity]:
        if a.posture == "floor":
            raise IncidentError(f"{a.person_id} is on the floor")
        hold = Hold(spec.duration or DEFAULT_INACTIVE_S, "inactive", sway=0.1)
        return [hold]

    def _pacing(self, a: Actor, spec: IncidentSpec, rec: IncidentRecord) -> list[Activity]:
        w = self.w
        nav = w.room.nav
        here = a.pos if a.posture == "standing" else None
        plan = get_up(a, w)
        if here is None:  # they'll get up where they are: pace from somewhere open nearby
            here = min((p.at for p in w.room.places if p.name == "middle"), key=lambda p: distance(p, a.pos),
                       default=a.pos)  # fmt: skip
        best: Floor | None = None
        for _ in range(40):
            ang = a.rng.uniform(-math.pi, math.pi)
            length = a.rng.uniform(2.5, 4.0)
            f = forward(ang)
            p = (here[0] + f[0] * length, here[1] + f[1] * length)
            if nav.free(p) and nav.line_free(here, p, margin=0.1):
                best = p
                break
        if best is None:
            raise IncidentError("no room to pace here")
        legs: list[Activity] = []
        for i in range(spec.repeats):
            target = _r(best) if i % 2 == 0 else _r(here)
            legs.append(Walk(target, purpose="pacing", pace=1.1))
            legs.append(Hold(a.rng.uniform(0.2, 0.8), "pacing", sway=0.2))
        rec.location = _r(here)
        return plan + [Walk(_r(here), purpose="pacing")] + legs

    def _exit_loiter(self, a: Actor, spec: IncidentSpec, rec: IncidentRecord) -> list[Activity]:
        w = self.w
        if not w.room.doors:
            raise IncidentError("the room has no door")
        door = min(w.room.doors, key=lambda d: distance(d.inside, a.pos))
        mids = [p.at for p in w.room.places if p.name == "middle"] or [a.pos]
        plan = get_up(a, w)
        for i in range(spec.repeats):
            spot = (door.inside[0] + a.rng.uniform(-0.2, 0.2), door.inside[1] + a.rng.uniform(-0.2, 0.2))
            if not w.room.nav.free(spot):
                spot = door.inside
            plan += [Walk(_r(spot), purpose="to the door"), Turn(door.heading_in + math.pi),
                     Hold(a.rng.uniform(6.0, 14.0), "lingering by the door", look=door.outside)]  # fmt: skip
            if i < spec.repeats - 1:
                plan += [Walk(a.rng.choice(mids), purpose="away from the door"), Hold(a.rng.uniform(3.0, 8.0), "pausing")]
        rec.location = door.inside
        return plan


def _r(p: Floor) -> tuple[float, float]:
    return (round(p[0], 3), round(p[1], 3))
