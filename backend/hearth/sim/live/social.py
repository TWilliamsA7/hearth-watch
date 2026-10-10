"""Conversations: two or more people gathering, facing each other and talking a while.

One person decides to talk to someone who isn't busy (standing about, or sitting). The
partner is marked as engaged so they stay put; the initiator walks to a spot about a
metre from them (in front of them, if they're sitting, so a seated and a standing person
can talk), turns to face them, and the conversation begins: both look at each other and
gesture now and then. A third or fourth person may join a conversation already going,
taking the widest gap on a circle around its centre. When the conversation's time is up,
everyone goes back to their own business, and won't start another one for a while.

No speech is simulated: the point is the bodies, which is what the camera sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hearth.sim.nav import Floor, distance

from hearth.sim.live.activities import COOLDOWN_S, Activity, Hold, Turn, Walk, get_up
from hearth.sim.live.actor import Actor
from hearth.sim.live.animate import gait, heading_to, idle, wrap
from hearth.sim.live.room import forward

if TYPE_CHECKING:
    from hearth.sim.live.engine import World

TALK_DISTANCE_M = 1.05
GROUP_RADIUS_M = 0.85
MAX_GROUP = 4
ENGAGE_TIMEOUT_S = 25.0
AVAILABLE = frozenset({"seated", "pausing", "standing about", "looking out", "browsing", "idle", "waiting"})
"""Activities someone can be drawn out of to talk."""


@dataclass
class Conversation:
    conv_id: str
    members: list[str]
    started: float
    ends: float
    centre: Floor = (0.0, 0.0)


@dataclass
class Social:
    conversations: dict[str, Conversation] = field(default_factory=dict)
    engaged_since: dict[str, float] = field(default_factory=dict)  # person engaged -> since when
    counter: int = 0

    def of(self, a: Actor) -> Conversation | None:
        return self.conversations.get(a.conversation or "")

    def start(self, w: World, a: Actor, b: Actor) -> Conversation:
        self.counter += 1
        mean_social = (a.profile.sociability + b.profile.sociability) / 2
        length = a.rng.uniform(15.0, 45.0) * (0.6 + mean_social)
        conv = Conversation(f"c{self.counter}", [b.person_id, a.person_id], w.t, w.t + length)
        self.conversations[conv.conv_id] = conv
        for m in (a, b):
            m.conversation = conv.conv_id
        self._centre(w, conv)
        return conv

    def join(self, w: World, conv: Conversation, a: Actor) -> None:
        conv.members.append(a.person_id)
        conv.ends = max(conv.ends, w.t + a.rng.uniform(10.0, 25.0))
        a.conversation = conv.conv_id
        self._centre(w, conv)

    def leave(self, w: World, a: Actor) -> None:
        conv = self.of(a)
        a.conversation = None
        if conv and a.person_id in conv.members:
            conv.members.remove(a.person_id)
            self._centre(w, conv)
        self.release_engaged(w, a)

    def release_engaged(self, w: World, a: Actor) -> None:
        """a was walking over to talk to someone and won't now."""
        for o in w.actors:
            if o.engaged_by == a.person_id:
                o.engaged_by = None
                self.engaged_since.pop(o.person_id, None)
        if a.engaged_by is not None:
            a.engaged_by = None
            self.engaged_since.pop(a.person_id, None)

    def _centre(self, w: World, conv: Conversation) -> None:
        pts = [w.actor(m).pos for m in conv.members]
        if pts:
            conv.centre = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    def update(self, w: World) -> None:
        for pid, since in list(self.engaged_since.items()):
            if w.t - since > ENGAGE_TIMEOUT_S:
                w.actor(pid).engaged_by = None
                del self.engaged_since[pid]
        for conv in list(self.conversations.values()):
            if w.t >= conv.ends or len(conv.members) < 2:
                for m in list(conv.members):
                    member = w.actor(m)
                    member.conversation = None
                    cool = COOLDOWN_S["chat"] / max(w.spec.social_frequency, 0.1)
                    member.cooldowns["chat"] = w.t + cool * member.rng.uniform(0.7, 1.4)
                del self.conversations[conv.conv_id]


class Converse(Activity):
    """Talking: face the others (turning on the spot if standing), look at them, gesture."""

    name = "talking"

    def label_for(self, a: Actor) -> str:
        return "sitting" if a.posture == "sitting" else "standing"

    def begin(self, a: Actor, w: World) -> None:
        self.next_look = 0.0

    def tick(self, a: Actor, w: World, dt: float) -> bool:
        conv = w.social.of(a)
        if conv is None:
            a.look_at = None
            return True
        others = [w.actor(m) for m in conv.members if m != a.person_id]
        if not others:
            a.look_at = None
            return True
        if self.tau >= self.next_look and others:  # look from one person to another now and then
            a.look_at = a.rng.choice(others).pos
            self.next_look = self.tau + a.rng.uniform(2.0, 6.0)
        a.velocity = (0.0, 0.0)
        if a.posture == "standing":
            face = heading_to(a.pos, conv.centre) if len(others) > 1 else heading_to(a.pos, others[0].pos)
            turn = wrap(face - a.heading)
            if abs(turn) > math.radians(12):
                step = 1.8 * dt
                a.heading = wrap(a.heading + max(-step, min(step, turn)))
                a.body = gait(a, 0.0, dt, stepping=1.0)
                return False
            from hearth.sim.pose import KEY_POSES

            a.body = idle(a, KEY_POSES["standing"], w.t, dt, sway=0.8, talking=True)
        elif a.posture == "sitting":
            from hearth.sim.pose import KEY_POSES

            a.body = idle(a, KEY_POSES["sitting"], w.t, dt, sway=0.6, talking=True)
        return False

    def end(self, a: Actor, w: World) -> None:
        a.look_at = None


class Approach(Activity):
    """The moment of arrival: start the conversation, or join it."""

    name = "greeting"

    def __init__(self, target: str) -> None:
        super().__init__()
        self.target = target

    def label_for(self, a: Actor) -> str:
        return "standing"

    def tick(self, a: Actor, w: World, dt: float) -> bool:
        b = w.actor(self.target)
        conv = w.social.of(b)
        if not b.present or b.incident or b.posture not in ("standing", "sitting"):
            self.failed = True
            return True
        if conv is not None:
            if len(conv.members) >= MAX_GROUP:
                self.failed = True
                return True
            w.social.join(w, conv, a)
        else:
            if b.engaged_by != a.person_id:
                self.failed = True
                return True
            w.social.start(w, a, b)
            b.engaged_by = None
            w.social.engaged_since.pop(b.person_id, None)
            w.replace_plan(b, [Converse()], "chat", keep_social=True)
        a.plan.appendleft(Converse())
        return True


def _spot(a: Actor, w: World, b: Actor) -> Floor | None:
    """Where to stand to talk to b."""
    conv = w.social.of(b)
    candidates: list[Floor] = []
    if conv is not None:
        c = conv.centre
        taken = [heading_to(c, w.actor(m).pos) for m in conv.members]
        for k in range(16):
            ang = -math.pi + k * math.pi / 8
            gap = min(abs(wrap(ang - t)) for t in taken)
            if gap > math.radians(50):
                f = forward(ang)
                candidates.append((c[0] + f[0] * GROUP_RADIUS_M, c[1] + f[1] * GROUP_RADIUS_M))
        candidates.sort(key=lambda p: distance(p, a.pos))
    elif b.posture == "sitting":
        for off in (0.0, 0.35, -0.35, 0.7, -0.7):
            f = forward(b.heading + off)
            candidates.append((b.pos[0] + f[0] * (TALK_DISTANCE_M + 0.25), b.pos[1] + f[1] * (TALK_DISTANCE_M + 0.25)))
    else:
        base = heading_to(b.pos, a.pos)
        for off in (0.0, 0.4, -0.4, 0.9, -0.9, 1.6, -1.6, math.pi):
            f = forward(base + off)
            candidates.append((b.pos[0] + f[0] * TALK_DISTANCE_M, b.pos[1] + f[1] * TALK_DISTANCE_M))
    # Stand clear of furniture (where others step up to sit), unless talking to someone seated.
    margin = 0.25 if b.posture == "sitting" and conv is None else 0.5
    for p in candidates:
        if not (w.room.nav.free(p) and w.room.raw_free(p, margin=margin)):
            continue
        if any(distance(p, s.stand_point(1.7)) < 0.7 for s in w.room.slots if s.piece != _seat_piece(w, b)):
            continue
        if any(o is not a and o.present and distance(o.pos, p) < 0.7 for o in w.actors):
            continue
        return (round(p[0], 3), round(p[1], 3))
    return None


def _seat_piece(w: World, b: Actor) -> int | None:
    for sid in b.slots:
        return w.room.slot(sid).piece
    return None


def plan_chat(a: Actor, w: World) -> list[Activity] | None:
    if a.conversation or a.incident:
        return None
    options: list[tuple[Actor, float]] = []
    for b in w.actors:
        if b is a or not b.present or b.incident or b.engaged_by or b.posture not in ("standing", "sitting"):
            continue
        conv = w.social.of(b)
        if conv is not None:
            if len(conv.members) >= MAX_GROUP:
                continue
        elif b.activity is None or b.activity.name not in AVAILABLE or w.t < b.cooldowns.get("chat", -1) - 20:
            continue
        d = distance(a.pos, b.pos)
        if d > 7.0:
            continue
        options.append((b, (0.3 + b.profile.sociability) / (1.0 + d / 2.5) * (0.7 if conv else 1.0)))
    while options:
        b = a.rng.choices([o for o, _ in options], weights=[v for _, v in options])[0]
        options = [o for o in options if o[0] is not b]
        spot = _spot(a, w, b)
        if spot is None:
            continue
        if w.social.of(b) is None:
            b.engaged_by = a.person_id
            w.social.engaged_since[b.person_id] = w.t
        face = heading_to(spot, b.pos)
        return get_up(a, w) + [Walk(spot, purpose=f"to talk to {b.person_id.upper()}", arrive=0.1), Turn(face),
                               Approach(b.person_id)]  # fmt: skip
    return None


def waiting(a: Actor, w: World) -> list[Activity]:
    """Someone is coming over to talk: stay put and watch them come."""
    who = w.actor(a.engaged_by) if a.engaged_by else None
    return [Hold(1.0, "waiting", look=who.pos if who else None, sway=0.8 if a.posture == "standing" else 0.5)]
