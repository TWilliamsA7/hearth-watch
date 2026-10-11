"""The actor state machine: primitive activities, steering, and the Brain that plans them.

An actor always carries out one Activity and holds a plan (a queue) of the ones after it.
Activities are small and composable: Hold a posture, Walk somewhere, Step a short way,
Turn on the spot, Transition between postures (sit down, stand up, lie down), Play a
choreographed motion (falls, getting up), leave the room and come back. When the plan
runs out, the Brain picks what to do next and turns it into a plan.

Posture changes are never skipped: get_up() prepends standing up (or sitting up, or
getting up off the floor) to any plan that needs the person on their feet, so nobody
goes from sitting to walking in one step.

New activities are new Activity subclasses; new things to do are new entries in
PLANNERS. Neither needs changes to the engine.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from hearth.sim.generator import Stance
from hearth.sim.nav import Floor, distance
from hearth.sim.pose import KEY_POSES, BodyPose, blend_body, smoothstep, tucked

from hearth.sim.live.actor import Actor, Posture
from hearth.sim.live.animate import Blend, Play, apply, gait, heading_to, idle, stance, wrap
from hearth.sim.live.room import Piece, Slot, forward

if TYPE_CHECKING:
    from hearth.sim.live.engine import World

POSTURE_LABEL: dict[str, str] = {"standing": "standing", "sitting": "sitting", "lying": "lying_intentional",
                                 "floor": "fall", "absent": "absent"}  # fmt: skip

PERSONAL_SPACE_M = 0.75  # walkers start steering away inside this
MIN_SEPARATION_M = 0.52  # bodies never get closer than this (centre to centre)
FALLEN_CLEARANCE_M = 0.95  # ...or this, from someone on the floor
LOOKAHEAD_S = 2.5
ACCEL = 1.0  # m/s^2
DECEL = 2.2
TURN_RATE = 3.2  # rad/s while walking
TURN_ON_SPOT_RATE = 2.0
STALL_S = 1.2  # no progress for this long: replan around people
MAX_REPLANS = 5


class Activity:
    name = "idle"
    label = "standing"

    def __init__(self) -> None:
        self.tau = 0.0
        self.started = False
        self.failed = False

    def label_for(self, a: Actor) -> str:
        return self.label

    def update(self, a: Actor, w: World, dt: float) -> bool:
        """Advance by dt; True when finished."""
        if not self.started:
            self.started = True
            self.begin(a, w)
        self.tau += dt
        return self.tick(a, w, dt)

    def begin(self, a: Actor, w: World) -> None:
        pass

    def tick(self, a: Actor, w: World, dt: float) -> bool:
        return True

    def end(self, a: Actor, w: World) -> None:
        """Called once, finished or cancelled."""


class Hold(Activity):
    """Stay in the current posture (or a given pose) for a while, never perfectly still."""

    def __init__(self, duration: float, name: str = "idle", *, pose: str | None = None, label: str | None = None,
                 sway: float = 1.0, look: Floor | None = None, talking: bool = False,
                 until: Callable[[], bool] | None = None) -> None:  # fmt: skip
        super().__init__()
        self.duration, self.name, self.pose, self.sway = duration, name, pose, sway
        self.look, self.talking, self.until = look, talking, until
        self._label = label

    def label_for(self, a: Actor) -> str:
        return self._label or POSTURE_LABEL[a.posture]

    def begin(self, a: Actor, w: World) -> None:
        self.start = a.body
        if self.pose is not None:
            self.target = KEY_POSES[self.pose]
        elif a.posture == "standing":
            self.target = KEY_POSES["standing"]
        elif a.posture == "sitting":
            self.target = a.rest_pose or KEY_POSES["sitting"]
        else:
            self.target = a.body
        if self.look is not None:
            a.look_at = self.look
        a.velocity = (0.0, 0.0)

    def tick(self, a: Actor, w: World, dt: float) -> bool:
        base = blend_body(self.start, self.target, smoothstep(self.tau / 0.6)) if self.tau < 0.6 else self.target
        a.body = idle(a, base, w.t, dt, sway=self.sway, talking=self.talking)
        if self.until is not None and self.until():
            return True
        return self.tau >= self.duration

    def end(self, a: Actor, w: World) -> None:
        if self.look is not None and a.look_at == self.look:
            a.look_at = None


class Walk(Activity):
    """Walk to a point along a planned route, steering around people on the way."""

    name = "walking"
    label = "walking"

    def __init__(self, goal: Floor, *, purpose: str = "", ignore: frozenset[int] = frozenset(),
                 pace: float = 1.0, arrive: float = 0.08, stop_short_ok: bool = True) -> None:  # fmt: skip
        super().__init__()
        self.goal, self.purpose, self.extra_ignore, self.pace, self.arrive = goal, purpose, ignore, pace, arrive
        self.stop_short_ok = stop_short_ok
        self.path: list[Floor] = []
        self.i = 1
        self.stall = 0.0
        self.best = math.inf
        self.replans = 0

    def begin(self, a: Actor, w: World) -> None:
        nav = w.room.nav
        self.ignore = set(self.extra_ignore) | nav.blockers(a.pos) | nav.blockers(self.goal)
        self._plan(a, w, avoid=False)
        self.limit = 12.0 + 4.0 * _length([a.pos, *self.path]) / max(a.profile.speed * self.pace, 0.3)

    def _plan(self, a: Actor, w: World, *, avoid: bool) -> None:
        discs = []
        if avoid:
            for o in w.bodies:
                if o is a or not o.present or o.slots_in_use or distance(o.pos, a.pos) > 3.5:
                    continue
                r = FALLEN_CLEARANCE_M if o.posture == "floor" else 0.6
                # Keep the disc of someone right next to us (a disc covering the start is ignored).
                discs.append((o.pos[0], o.pos[1], min(r, distance(o.pos, a.pos) - 0.03)))
        path = w.room.nav.find_path(a.pos, self.goal, ignore=self.ignore, avoid=discs)
        if path is None and not avoid:
            self.failed = True
            return
        if path is not None:
            self.path, self.i = path, 1
            a.path, a.destination = path, self.goal

    def tick(self, a: Actor, w: World, dt: float) -> bool:
        if self.failed or len(self.path) < 2:
            a.velocity = (0.0, 0.0)
            return True
        while self.i < len(self.path) - 1 and distance(a.pos, self.path[self.i]) < 0.3:
            self.i += 1
        remaining = distance(a.pos, self.path[self.i]) + _length(self.path[self.i :])
        if remaining <= self.arrive:
            a.velocity = (0.0, 0.0)
            return True
        if remaining < 1.2 and self.stop_short_ok and any(
            o is not a and o.present and not o.slots_in_use and distance(o.pos, self.goal) < 0.6 for o in w.bodies
        ):  # someone is standing on the spot: stop short of it
            a.velocity = (0.0, 0.0)
            return True
        v_pref = a.profile.speed * self.pace * min(1.0, max(remaining / 0.7, 0.2))
        steer(a, w, dt, self.path[self.i], v_pref, self.ignore)
        a.path = [a.pos, *self.path[self.i :]]

        if remaining < self.best - 0.08:
            self.best, self.stall = remaining, 0.0
        else:
            self.stall += dt
        if self.stall > STALL_S:
            self.stall = 0.0
            self.replans += 1
            make_way(a, w, self.path[self.i])
            if self.replans > MAX_REPLANS:
                self.failed = True
                return True
            self._plan(a, w, avoid=True)
        if self.tau > self.limit:
            self.failed = True
            return True
        return False

    def end(self, a: Actor, w: World) -> None:
        a.path, a.destination = [], None


IDLE_HOLDS = frozenset({"pausing", "standing about", "looking out", "browsing", "waiting", "idle"})


def make_way(a: Actor, w: World, towards: Floor) -> None:
    """Someone idling in a's way steps aside, as people do when someone needs to get past."""
    for o in w.actors:
        if o is a or not o.present or o.posture != "standing" or o.conversation or o.incident:
            continue
        if distance(o.pos, a.pos) > 0.9 or not isinstance(o.activity, Hold) or o.activity.name not in IDLE_HOLDS:
            continue
        ux, uz = towards[0] - a.pos[0], towards[1] - a.pos[1]
        n = math.hypot(ux, uz) or 1e-9
        ux, uz = ux / n, uz / n
        for side in (1.0, -1.0):  # out of the line a is walking along
            p = (o.pos[0] - uz * side * 0.8 + ux * 0.2, o.pos[1] + ux * side * 0.8 + uz * 0.2)
            if w.room.nav.free(p) and all(distance(p, q.pos) > 0.7 for q in w.bodies if q is not o and q.present):
                rest = max(o.activity.duration - o.activity.tau, 1.0)
                w.replace_plan(o, [Walk(p, purpose="out of the way", pace=0.8),
                                   Hold(rest, o.activity.name, look=a.pos)], o.intent)  # fmt: skip
                return


def steer(a: Actor, w: World, dt: float, target: Floor, v_pref: float, ignore: set[int]) -> None:
    """One tick of local movement towards target: separation, look-ahead avoidance (pass
    on the right), slowing behind someone in the way, and hard limits so nobody walks
    into furniture or another person."""
    px, pz = a.pos
    dx, dz = target[0] - px, target[1] - pz
    d = math.hypot(dx, dz) or 1e-9
    ux, uz = dx / d, dz / d
    ax = az = 0.0
    slow = 1.0
    for o in w.bodies:
        if o is a or not o.present or o.slots_in_use:
            continue
        rx, rz = o.pos[0] - px, o.pos[1] - pz
        dist = math.hypot(rx, rz)
        if dist > 3.0 or dist < 1e-6:
            continue
        space = FALLEN_CLEARANCE_M + 0.3 if o.posture == "floor" else PERSONAL_SPACE_M
        if dist < space:  # too close: move apart
            k = (space - dist) / space * 1.6
            ax -= rx / dist * k
            az -= rz / dist * k
        # Look ahead: where will we be closest, if both carry on?
        vx, vz = ux * v_pref - o.velocity[0], uz * v_pref - o.velocity[1]
        vv = vx * vx + vz * vz
        if vv > 1e-6:
            tca = max(0.0, min((rx * vx + rz * vz) / vv, LOOKAHEAD_S))
            cx, cz = rx - vx * tca, rz - vz * tca
            closest = math.hypot(cx, cz)
            if closest < space and tca > 0:
                k = (space - closest) / space * (1.0 - tca / LOOKAHEAD_S) * 1.4
                # Sidestep away from where they'll be; head-on, pass on the right.
                sx, sz = -cx, -cz
                if math.hypot(sx, sz) < 0.05:
                    sx, sz = -uz, ux  # the walker's right... in x-right/z-front axes
                n = math.hypot(sx, sz)
                ax += sx / n * k
                az += sz / n * k
        ahead = (rx * ux + rz * uz) / dist
        if ahead > 0.8 and dist < 1.1:  # someone right in front: ease off
            slow = min(slow, max(0.15, (dist - MIN_SEPARATION_M) / 0.6))
    wx, wz = ux + ax, uz + az
    n = math.hypot(wx, wz) or 1e-9
    want = (wx / n * v_pref * slow, wz / n * v_pref * slow)
    # Turn before striding off in a very different direction.
    if a.speed < 0.25:
        diff = abs(wrap(math.atan2(want[0], want[1]) - a.heading))
        if diff > math.radians(50):
            want = (want[0] * 0.15, want[1] * 0.15)
    vx, vz = a.velocity
    ddx, ddz = want[0] - vx, want[1] - vz
    dd = math.hypot(ddx, ddz)
    lim = (DECEL if math.hypot(*want) < a.speed else ACCEL) * dt
    if dd > lim:
        ddx, ddz = ddx / dd * lim, ddz / dd * lim
    vx, vz = vx + ddx, vz + ddz
    new = (px + vx * dt, pz + vz * dt)
    nav = w.room.nav
    if not nav.free(new, ignore):
        if nav.free((new[0], pz), ignore):
            new, vz = (new[0], pz), 0.0
        elif nav.free((px, new[1]), ignore):
            new, vx = (px, new[1]), 0.0
        else:
            new, vx, vz = (px, pz), 0.0, 0.0
    # Nobody walks into anybody: take away the part of the step that closes in on them,
    # so a walker slides round someone rather than stopping dead against them.
    mx, mz = new[0] - px, new[1] - pz
    for o in w.bodies:
        if o is a or not o.present or o.slots_in_use:
            continue
        limit = FALLEN_CLEARANCE_M if o.posture == "floor" else MIN_SEPARATION_M
        before, after = distance(a.pos, o.pos), distance((px + mx, pz + mz), o.pos)
        if after < limit and after < before:
            ox, oz = px - o.pos[0], pz - o.pos[1]
            n = math.hypot(ox, oz) or 1e-9
            ox, oz = ox / n, oz / n
            inward = mx * ox + mz * oz
            if inward < 0:
                mx, mz = mx - ox * inward, mz - oz * inward
    cand = (px + mx, pz + mz)
    if (mx or mz) and nav.free(cand, ignore) and all(
        distance(cand, o.pos) >= min(distance(a.pos, o.pos), FALLEN_CLEARANCE_M if o.posture == "floor" else MIN_SEPARATION_M) - 1e-6
        for o in w.bodies if o is not a and o.present and not o.slots_in_use
    ):  # fmt: skip
        new = cand
        vx, vz = mx / dt, mz / dt
    else:
        new, vx, vz = (px, pz), 0.0, 0.0
    a.velocity = (vx, vz)
    a.pos = new
    sp = math.hypot(vx, vz)
    if sp > 0.05 or math.hypot(*want) > 0.05:
        face = math.atan2(vx, vz) if sp > 0.05 else math.atan2(want[0], want[1])
        turn = wrap(face - a.heading)
        step = TURN_RATE * dt
        a.heading = wrap(a.heading + max(-step, min(step, turn)))
    a.body = gait(a, sp, dt, stepping=0.4 if sp < 0.1 and math.hypot(*want) > 0.01 else 0.0)


def _length(points: list[Floor]) -> float:
    return sum(distance(p, q) for p, q in zip(points, points[1:]))


class Step(Activity):
    """A short, straight, unhurried step (back to a seat, through a door), ignoring furniture.
    face=False keeps the heading, stepping backwards if the step goes behind."""

    name = "stepping"
    label = "walking"

    def __init__(self, to: Floor, *, speed: float = 0.45, name: str = "stepping", face: bool = True,
                 stride: float = 1.0) -> None:  # fmt: skip
        super().__init__()
        self.to, self.speed, self.name, self.face, self.stride = to, speed, name, face, stride
        self.waited = 0.0

    def tick(self, a: Actor, w: World, dt: float) -> bool:
        dx, dz = self.to[0] - a.pos[0], self.to[1] - a.pos[1]
        d = math.hypot(dx, dz)
        if d < 0.01:
            a.velocity = (0.0, 0.0)
            a.body = gait(a, 0.0, dt)
            return True
        sp = min(self.speed, d / dt, 0.15 + d * 1.5)
        nxt = (a.pos[0] + dx / d * sp * dt, a.pos[1] + dz / d * sp * dt)
        if any(o is not a and o.present and not o.slots_in_use
               and distance(nxt, o.pos) < MIN_SEPARATION_M + 0.05 and distance(nxt, o.pos) < distance(a.pos, o.pos)
               for o in w.bodies):  # fmt: skip
            # Someone's in the way: wait for them, and give up if they stay.
            a.velocity = (0.0, 0.0)
            a.body = gait(a, 0.0, dt)
            self.waited += dt
            if self.waited > 6.0:
                self.failed = True
                return True
            return False
        a.velocity = (dx / d * sp, dz / d * sp)
        a.pos = nxt
        if self.face:
            turn = wrap(math.atan2(dx, dz) - a.heading)
            step = TURN_RATE * dt
            a.heading = wrap(a.heading + max(-step, min(step, turn)))
        backwards = dx * math.sin(a.heading) + dz * math.cos(a.heading) < 0
        a.body = gait(a, sp, dt, backwards=backwards, stride=self.stride)
        return self.tau > 8.0


class Turn(Activity):
    """Turn on the spot, stepping round."""

    name = "turning"
    label = "standing"

    def __init__(self, heading: float) -> None:
        super().__init__()
        self.heading = heading

    def tick(self, a: Actor, w: World, dt: float) -> bool:
        turn = wrap(self.heading - a.heading)
        step = TURN_ON_SPOT_RATE * dt * (0.7 if a.profile.mobility == "slow" else 1.0)
        a.velocity = (0.0, 0.0)
        if abs(turn) <= step:
            a.heading = self.heading
            a.body = gait(a, 0.0, dt)
            return True
        a.heading = wrap(a.heading + math.copysign(step, turn))
        a.body = gait(a, 0.0, dt, stepping=1.0)
        return False


class Transition(Activity):
    """From the current body to a key pose, moving the root, turning and rising as it goes."""

    def __init__(self, pose: str | BodyPose, duration: float, *, name: str, label: str, posture: Posture,
                 root: Floor | None = None, heading: float | None = None, base: float | None = None,
                 on_end: Callable[[Actor, World], None] | None = None, lag: float = 1.0) -> None:  # fmt: skip
        """lag > 1: the root moves later than the body changes shape (lowering onto a seat
        before settling back into it); lag < 1: sooner (weight forward over the feet before
        rising)."""
        super().__init__()
        self.pose = KEY_POSES[pose] if isinstance(pose, str) else pose
        self.duration, self.name, self.label, self.posture = duration, name, label, posture
        self.root, self.heading, self.base, self.on_end, self.lag = root, heading, base, on_end, lag

    def begin(self, a: Actor, w: World) -> None:
        st = stance(a)
        end = Stance(self.pose, st.heading if self.heading is None else self.heading,
                     *(self.root if self.root is not None else (st.x, st.z)),
                     st.base if self.base is None else self.base)  # fmt: skip
        self.blend = Blend(st, end, self.duration)
        a.velocity = (0.0, 0.0)

    def tick(self, a: Actor, w: World, dt: float) -> bool:
        st = self.blend.at(self.tau)
        if self.lag != 1.0 and not self.blend.done(self.tau):
            s, e = self.blend.start, self.blend.end
            k = self.blend.ease((self.tau / self.duration) ** self.lag)
            st = Stance(st.body, st.heading, s.x + (e.x - s.x) * k, s.z + (e.z - s.z) * k, st.base)
        apply(a, st)
        if self.blend.done(self.tau):
            a.posture = self.posture
            return True
        return False

    def end(self, a: Actor, w: World) -> None:
        if self.started and self.blend.done(self.tau):
            a.posture = self.posture
            a.rest_pose = self.pose if self.posture in ("sitting", "lying") else None
            if self.on_end:
                self.on_end(a, w)


class PlayMotion(Activity):
    """A choreographed motion: a fall, a stumble, getting up off the floor."""

    def __init__(self, motion: str, *, name: str, label: str, posture: Posture, end: Floor | None = None,
                 heading: float | None = None, base: float = 0.0, pace: float = 1.0) -> None:  # fmt: skip
        super().__init__()
        self.motion, self.name, self.label, self.posture = motion, name, label, posture
        self.end_at, self.heading, self.base, self.pace = end, heading, base, pace

    def begin(self, a: Actor, w: World) -> None:
        self.play = Play(stance(a), self.motion, a.rng.randrange(1 << 30), self.end_at or a.pos,
                         a.heading if self.heading is None else self.heading, self.base, self.pace)  # fmt: skip
        a.velocity = (0.0, 0.0)

    def tick(self, a: Actor, w: World, dt: float) -> bool:
        apply(a, self.play.at(self.tau))
        if self.play.done(self.tau):
            a.posture = self.posture
            return True
        return False


class Do(Activity):
    """Run a function (bookkeeping between activities); takes no time."""

    name = "idle"

    def __init__(self, fn: Callable[[Actor, World], None]) -> None:
        super().__init__()
        self.fn = fn

    def label_for(self, a: Actor) -> str:
        return POSTURE_LABEL[a.posture]

    def tick(self, a: Actor, w: World, dt: float) -> bool:
        self.fn(a, w)
        return True


class Away(Activity):
    """Out of the room for a while."""

    name = "away"
    label = "absent"

    def __init__(self, duration: float) -> None:
        super().__init__()
        self.duration = duration

    def begin(self, a: Actor, w: World) -> None:
        a.posture = "absent"
        a.velocity = (0.0, 0.0)

    def tick(self, a: Actor, w: World, dt: float) -> bool:
        return self.tau >= self.duration


class Enter(Activity):
    """Come in through a door: appear just outside it and step in."""

    name = "entering"
    label = "walking"

    def __init__(self, door_id: str) -> None:
        super().__init__()
        self.door_id = door_id

    def begin(self, a: Actor, w: World) -> None:
        door = w.room.door(self.door_id)
        a.pos, a.heading, a.posture, a.base = door.outside, door.heading_in, "standing", 0.0
        a.body = dict(KEY_POSES["standing"])
        self.step = Step(door.inside, speed=a.profile.speed * 0.8, name="entering")

    def tick(self, a: Actor, w: World, dt: float) -> bool:
        return self.step.update(a, w, dt)


class Leave(Activity):
    """Step out through a door (from just inside it)."""

    name = "leaving"
    label = "walking"

    def __init__(self, door_id: str) -> None:
        super().__init__()
        self.door_id = door_id

    def begin(self, a: Actor, w: World) -> None:
        self.step = Step(w.room.door(self.door_id).outside, speed=a.profile.speed * 0.8, name="leaving")

    def tick(self, a: Actor, w: World, dt: float) -> bool:
        return self.step.update(a, w, dt)


# --- slots ---


def reserve(a: Actor, w: World, *slots: Slot) -> None:
    for s in slots:
        s.occupant = a.person_id
        if s.slot_id not in a.slots:
            a.slots.append(s.slot_id)


def release(a: Actor, w: World, only: list[str] | None = None) -> None:
    for sid in list(a.slots if only is None else only):
        s = w.room.slot(sid)
        if s.occupant == a.person_id:
            s.occupant = None
        if sid in a.slots:
            a.slots.remove(sid)


def _releaser(a: Actor) -> Callable[[Actor, World], None]:
    """Frees the slots held now (not any reserved later, for the next seat)."""
    held = list(a.slots)
    return lambda a, w: release(a, w, held)


def _slot_of(a: Actor, w: World, kind: str) -> Slot | None:
    for sid in a.slots:
        s = w.room.slot(sid)
        if s.kind == kind:
            return s
    return None


def mobility_pace(a: Actor) -> float:
    return 1.45 if a.profile.mobility == "slow" else 1.0


def get_up(a: Actor, w: World) -> list[Activity]:
    """Whatever it takes to be standing on open floor again."""
    pace = mobility_pace(a)
    if a.posture == "sitting":
        seat = _slot_of(a, w, "sit")
        root = seat.stand_point(a.size) if seat else None
        return [Transition("standing", 1.5 * pace, name="standing up", label="standing", posture="standing",
                           root=root, on_end=_releaser(a), lag=0.67)]  # fmt: skip
    if a.posture == "lying":
        lie = _slot_of(a, w, "lie")
        # Off the side they got on from (the seat they hold), else the lie slot's usual one.
        entry = _slot_of(a, w, "sit")
        if entry is None and lie and lie.entry and lie.entry in w.room._slot_index:
            entry = w.room.slot(lie.entry)
        stances = lie_stances(lie, w.room.pieces[lie.piece], a.size) if lie else None
        out: list[Activity] = []
        if stances is not None:  # sit up where they lie, then swing the legs down off the edge
            out.append(_to(stances[1], 1.6 * pace, name="sitting up", label="sitting", posture="sitting"))
        if entry is not None:
            edge = seat_stance(entry, a.size)
            if stances is not None:
                out.append(_to(dataclasses.replace(edge, body=tucked(edge.body)), 1.5 * pace, name="swinging legs down",
                               label="sitting", posture="sitting"))  # fmt: skip
            out.append(_to(edge, 0.9 * pace, name="swinging legs down", label="sitting", posture="sitting"))
        else:
            out.append(Transition("sitting", 2.3 * pace, name="sitting up", label="sitting", posture="sitting", base=0.0))
        if entry is not None and entry.slot_id not in a.slots:
            a.slots.append(entry.slot_id)
        out.append(Transition("standing", 1.5 * pace, name="standing up", label="standing", posture="standing",
                              root=entry.stand_point(a.size) if entry else None, on_end=_releaser(a), lag=0.67))  # fmt: skip
        return out
    if a.posture == "floor":
        return [PlayMotion("get_up", name="getting up", label="standing", posture="standing", pace=pace)]
    if a.posture == "absent":
        return [Enter(a.rng.choice(w.room.doors).door_id)]
    return []


def seat_stance(seat: Slot, size: float) -> Stance:
    """How someone `size` metres tall ends up sitting in `seat`."""
    return Stance(seat.pose(size), seat.heading, seat.at[0], seat.at[1], 0.0)


def lie_stances(lie: Slot, piece: Piece, size: float) -> tuple[Stance, Stance] | None:
    """How someone `size` metres tall lies on `lie` (fit.lie_fit: centred, knees drawn up or
    curled if the piece is short for them), and sits up on the same spot on the way:
    (lying, sitting up). None if they don't fit."""
    from hearth.sim.live.fit import lie_fit

    fit = lie_fit(piece.kind, lie.heading - piece.rotation, size)
    if fit is None:
        return None
    x, z = piece.local(*fit.root)
    sx, sz = piece.local(*fit.sit_root)
    return (Stance(fit.pose, lie.heading, x, z, fit.base), Stance(fit.sit_pose, lie.heading, sx, sz, fit.sit_base))


def lie_stance(lie: Slot, piece: Piece, size: float) -> Stance | None:
    """How someone `size` metres tall ends up lying on `lie`; None if they don't fit."""
    st = lie_stances(lie, piece, size)
    return st[0] if st else None


def _to(st: Stance, duration: float, **kw) -> Transition:
    """A Transition to a whole stance."""
    return Transition(st.body, duration, root=(st.x, st.z), heading=st.heading, base=st.base, **kw)


def sit_plan(a: Actor, w: World, seat: Slot) -> list[Activity] | None:
    """Walk up to a seat, turn round on open floor, step back to it and sit down (slot already
    reserved). Turning at the seat itself would sweep the feet through its front."""
    approach = w.room.approach(seat, a.size)
    if approach is None:
        return None
    pace = mobility_pace(a)
    piece = w.room.pieces[seat.piece]
    ignore = frozenset({piece.index})
    return [
        Walk(approach, purpose="to a seat", ignore=ignore, stop_short_ok=False),
        Turn(seat.heading),
        Step(seat.stand_point(a.size), speed=0.3, name="stepping back", face=False, stride=0.25),
        _to(seat_stance(seat, a.size), 1.4 * pace, name="sitting down", label="sitting", posture="sitting", lag=1.5),
    ]


def lie_plan(a: Actor, w: World, lie: Slot, entry: Slot) -> list[Activity] | None:
    """Sit on `entry`, swing the legs up onto the piece, lie back and rest (both already
    reserved). Going straight from sitting on the edge to lying would sweep the body through
    the backrest or the mattress."""
    stances = lie_stances(lie, w.room.pieces[lie.piece], a.size)
    sit = sit_plan(a, w, entry)
    if sit is None or stances is None:
        return None
    lying, sitting_up = stances
    pace = mobility_pace(a)
    edge = seat_stance(entry, a.size)
    return sit + [
        Hold(a.rng.uniform(1.0, 3.0), "seated"),
        _to(dataclasses.replace(edge, body=tucked(edge.body)), 0.9 * pace, name="lifting legs up", label="sitting",
            posture="sitting"),  # fmt: skip
        _to(sitting_up, 1.5 * pace, name="lifting legs up", label="sitting", posture="sitting"),
        _to(lying, 1.6 * pace, name="lying down", label="lying_intentional", posture="lying"),
        Hold(dwell(a, w, 50.0, 160.0), "resting", sway=0.0),
    ]


# --- the Brain ---

BASE_WEIGHTS: dict[str, float] = {
    "sit": 3.0, "rest": 0.7, "window": 0.9, "browse": 0.8, "wander": 1.0, "pick_up": 0.35, "chat": 1.6,
    "leave": 0.3, "pause": 0.8,
}  # fmt: skip
COOLDOWN_S: dict[str, float] = {"chat": 45.0, "leave": 150.0, "rest": 180.0, "pick_up": 60.0, "browse": 30.0,
                                "window": 30.0}  # fmt: skip


def dwell(a: Actor, w: World, lo: float, hi: float) -> float:
    return a.rng.uniform(lo, hi) * a.profile.dwell / w.spec.activity_frequency


def choose(a: Actor, w: World) -> tuple[str, list[Activity]]:
    """What to do next: a weighted pick among what's possible right now."""
    from hearth.sim.live.social import plan_chat

    weights: dict[str, float] = {}
    recent = [i for i, _ in list(a.history)[-3:]]
    for intent, base in BASE_WEIGHTS.items():
        wgt = base * a.profile.prefs.get(intent, 1.0)
        if intent == "chat":
            wgt *= a.profile.sociability * w.spec.social_frequency * (1.0 if len(w.present()) > 1 else 0.0)
        if w.t < a.cooldowns.get(intent, -1.0):
            continue
        if intent in recent:
            wgt *= 0.25 if recent[-1] == intent else 0.6
        if a.posture == "sitting":
            wgt *= {"sit": 0.15, "pause": 0.2, "pick_up": 0.3}.get(intent, 1.0)
        if a.posture == "lying":
            wgt *= {"rest": 0.0, "sit": 0.6}.get(intent, 1.0)
        # After a long time on their feet, people want to sit down.
        if a.posture == "standing" and intent in ("sit", "rest"):
            wgt *= 1.0 + min(w.t - a.intent_since, 120.0) / 60.0
        if wgt > 0:
            weights[intent] = wgt
    planners = dict(PLANNERS, chat=plan_chat)
    while weights:
        intent = a.rng.choices(list(weights), weights=list(weights.values()))[0]
        del weights[intent]
        plan = planners[intent](a, w)
        if plan:
            return intent, plan
    return "pause", get_up(a, w) + [Hold(dwell(a, w, 2.0, 6.0), "pausing")]


def _place_plan(kind: str, lo: float, hi: float):
    def plan(a: Actor, w: World) -> list[Activity] | None:
        places = [p for p in w.room.places if p.name == kind and distance(p.at, a.pos) > 0.8
                  and not any(o is not a and o.present and distance(o.pos, p.at) < 0.9 for o in w.bodies)]  # fmt: skip
        if not places:
            return None
        place = a.rng.choice(places)
        at = (place.at[0] + a.rng.uniform(-0.15, 0.15), place.at[1] + a.rng.uniform(-0.15, 0.15))
        if not w.room.nav.free(at):
            at = place.at
        look = (at[0] + forward(place.heading)[0] * 3, at[1] + forward(place.heading)[1] * 3)
        name = {"window": "looking out", "browse": "browsing", "middle": "standing about"}[kind]
        return get_up(a, w) + [Walk(at, purpose=kind), Turn(place.heading),
                               Hold(dwell(a, w, lo, hi), name, look=look if kind != "middle" else None)]  # fmt: skip

    return plan


def plan_sit(a: Actor, w: World) -> list[Activity] | None:
    seats = w.room.free_slots("sit")
    if not seats:
        return None
    pref = {"armchair": 1.3, "couch": 1.0, "loveseat": 1.0, "chair": 0.9, "bed": 0.25}
    scored = []
    for s in seats:
        kind = w.room.pieces[s.piece].kind.name
        if a.slots and s.slot_id in a.slots:
            continue
        scored.append((s, pref.get(kind, 0.5) / (1.0 + distance(s.at, a.pos) / 3.0)))
    if not scored:
        return None
    seat = a.rng.choices([s for s, _ in scored], weights=[v for _, v in scored])[0]
    if w.room.approach(seat, a.size) is None:
        return None
    first = get_up(a, w)
    reserve(a, w, seat)
    return first + sit_plan(a, w, seat) + [Hold(dwell(a, w, 25.0, 90.0), "seated")]  # type: ignore[operator]


def plan_rest(a: Actor, w: World) -> list[Activity] | None:
    beds = w.room.free_slots("lie", a.person_id)
    a.rng.shuffle(beds)
    for lie in beds:
        entry = w.room._slot_index.get(lie.entry or "")
        if entry is None or entry.occupant not in (None, a.person_id):
            entry = next((s for s in w.room.slots if s.group == lie.group and s.kind == "sit" and s.occupant is None), None)
        if entry is None or w.room.approach(entry, a.size) is None:
            continue
        if lie_stances(lie, w.room.pieces[lie.piece], a.size) is None:  # too short for them, even curled up
            continue
        first = get_up(a, w)
        reserve(a, w, lie, entry)
        rest = lie_plan(a, w, lie, entry)
        assert rest is not None
        return first + rest
    return None


def plan_pick_up(a: Actor, w: World) -> list[Activity] | None:
    """Bend down to the floor for something and straighten up: an ordinary movement that looks like trouble."""
    for _ in range(10):
        p = (a.pos[0] + a.rng.uniform(-2.5, 2.5), a.pos[1] + a.rng.uniform(-2.5, 2.5))
        if w.room.nav.free(p) and w.room.nav.clearance(p) > 0.3:
            break
    else:
        return None
    return get_up(a, w) + [
        Walk(p, purpose="to pick something up"),
        Transition("bending", 0.9, name="bending down", label="bending", posture="standing"),
        Hold(a.rng.uniform(1.5, 3.5), "picking up", pose="bending", label="bending", sway=0.3),
        Transition("standing", 1.0, name="straightening up", label="standing", posture="standing"),
    ]


def plan_leave(a: Actor, w: World) -> list[Activity] | None:
    if len(w.room.doors) == 0 or a.posture == "absent":
        return None
    out = a.rng.choice(w.room.doors)
    back = a.rng.choice(w.room.doors)
    return get_up(a, w) + [Walk(out.inside, purpose="to the door", stop_short_ok=False), Leave(out.door_id),
                           Away(dwell(a, w, 30.0, 120.0)), Enter(back.door_id)]  # fmt: skip


def plan_pause(a: Actor, w: World) -> list[Activity] | None:
    return get_up(a, w) + [Hold(dwell(a, w, 3.0, 9.0), "pausing")]


PLANNERS: dict[str, Callable[[Actor, World], list[Activity] | None]] = {
    "sit": plan_sit,
    "rest": plan_rest,
    "window": _place_plan("window", 8.0, 25.0),
    "browse": _place_plan("browse", 6.0, 18.0),
    "wander": _place_plan("middle", 4.0, 12.0),
    "pick_up": plan_pick_up,
    "leave": plan_leave,
    "pause": plan_pause,
}
"""intent -> planner. Add an entry (and a BASE_WEIGHTS weight) to teach people something new."""


def abandon(a: Actor, w: World) -> None:
    """Drop the rest of the plan (a walk failed, an incident began). Reserved seats the
    person isn't actually in are freed."""
    while a.plan:
        a.plan.popleft().end(a, w)
    if a.posture not in ("sitting", "lying"):
        release(a, w)
