"""Procedural scenarios: a spec and a seed in, an ordinary Scenario out.

    scenario = generate(GenSpec(people=3, fall_rate=1.0), seed=7)

Residents are planned one after another on a shared floor plan (hearth.sim.behavior),
each pursuing goals from their persona and keeping out of the way of everyone planned
before them. Faults are decided up front from the seed: whether someone falls, who,
roughly when and how, whether someone else comes to help, and whether the camera drops
out for a while. The faller is planned first, so a helper can head for where they landed.
Everything that drives a result is written into tags (Scenario.tags for the run,
Segment.tags for each fall, hard negative and outage) so evaluation can break results down
by them.

The output is a plain Scenario with no seed variation of its own: the generator already
is the variation. Its scenario_id carries the room, the number of people, a short hash of
the spec and the seed, so a run name says how to make it again.
"""

from __future__ import annotations

import hashlib
import random
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from hearth.sim.behavior import FALL_KINDS, PERSONAL_SPACE_M, PERSONAS, Actor, FallKind, FallPlan, Household, gap
from hearth.sim.floorplan import FloorPlan
from hearth.sim.generator import ResidentTrack
from hearth.sim.scenario import Scenario, Scene, SceneObject, Segment, TagValue, Variation

GENERATOR_VERSION = "1"

Unit = Annotated[float, Field(ge=0.0, le=1.0)]
RoomName = Literal["day-room", "bedroom", "lounge"]


def _o(kind: str, x: float, depth: float, w: float, h: float, scene: Scene = Scene()) -> SceneObject:
    """An object standing at `depth`, given its frame width and height."""
    bottom = scene.floor_at(depth)
    return SceneObject(kind=kind, x=x, y=round(bottom - h, 4), w=w, h=h)  # type: ignore[arg-type]


ROOMS: dict[str, Scene] = {
    # As the hand-written day-room scenarios.
    "day-room": Scene(objects=[
        SceneObject(kind="couch", x=0.62, y=0.64, w=0.30, h=0.182),
        SceneObject(kind="chair", x=0.10, y=0.66, w=0.12, h=0.218),
    ]),
    "bedroom": Scene(objects=[
        _o("bed", 0.56, 0.75, 0.36, 0.16),
        _o("chair", 0.30, 0.55, 0.11, 0.20),
        _o("occluder", 0.06, 0.35, 0.18, 0.40),
    ]),
    "lounge": Scene(objects=[
        _o("couch", 0.08, 0.8, 0.30, 0.17),
        _o("chair", 0.74, 0.45, 0.12, 0.21),
        _o("mat", 0.40, 0.05, 0.30, 0.05),
    ]),
}  # fmt: skip


class GenSpec(BaseModel):
    """What to generate. Every field has a sensible default."""

    model_config = ConfigDict(extra="forbid")

    people: int = Field(default=1, ge=1, le=6)
    room: RoomName = "day-room"
    duration: float = Field(default=45.0, ge=20.0, le=600.0, description="Target seconds; a long lie may run over.")
    fall_rate: Unit = Field(default=0.5, description="Chance that someone falls.")
    fall_kinds: dict[FallKind, float] = Field(default_factory=lambda: {k: 1.0 for k in FALL_KINDS})
    recover_rate: Unit = Field(default=0.3, description="Chance a faller gets back up (else they stay down).")
    responder_rate: Unit = Field(default=0.5, description="Chance someone comes to help a faller who stays down.")
    outage_rate: Unit = Field(default=0.1, description="Chance the camera drops out for a few seconds.")
    personas: dict[str, float] = Field(default_factory=lambda: {"steady": 2.0, "frail": 1.0, "restless": 1.0})
    start_inside: Unit = Field(default=0.7, description="Chance each resident is in the room at the start.")
    fps: float = Field(default=12.0, gt=0.0, le=30.0)

    def key(self) -> str:
        """Short, stable hash of the spec, for scenario ids."""
        return hashlib.sha1(self.model_dump_json().encode()).hexdigest()[:5]


def scenario_id(spec: GenSpec, seed: int) -> str:
    return f"gen-{spec.room}-p{spec.people}-{spec.key()}-{seed}"


def _weighted(rng: random.Random, weights: dict) -> str:
    keys = [k for k, w in weights.items() if w > 0]
    return rng.choices(keys, [weights[k] for k in keys])[0]


def generate(spec: GenSpec, seed: int) -> Scenario:
    rng = random.Random(f"gen/{GENERATOR_VERSION}/{seed}")
    scene = ROOMS[spec.room]
    home = Household(FloorPlan(scene))
    ids = [f"p{i + 1}" for i in range(spec.people)]
    personas = {rid: PERSONAS[_weighted(rng, spec.personas)] for rid in ids}

    # Faults, decided before anyone moves.
    faller = None
    fall: FallPlan | None = None
    if rng.random() < spec.fall_rate:
        faller = rng.choices(ids, [personas[r].fall_weight for r in ids])[0]
        fall = FallPlan(_weighted(rng, spec.fall_kinds), round(rng.uniform(0.15, 0.55) * spec.duration, 3),  # type: ignore[arg-type]
                        "recover" if rng.random() < spec.recover_rate else "long_lie")  # fmt: skip
    outage = rng.random() < spec.outage_rate
    responder, reaction = None, 0.0
    if fall is not None and fall.outcome == "long_lie" and spec.people > 1 and rng.random() < spec.responder_rate:
        responder = rng.choice([r for r in ids if r != faller])
        reaction = rng.uniform(2.0, 8.0)

    planned: dict[str, Actor] = {}
    total = spec.duration
    for rid in sorted(ids, key=lambda r: r != faller):  # the faller first
        actor = _arrive(home, rid, personas[rid], random.Random(f"gen/{GENERATOR_VERSION}/{seed}/{rid}"), spec)
        fallen = planned[faller] if faller in planned else None
        if rid == responder and fallen is not None and fallen.fall_events:
            actor.live(fallen.fall_events[0][0] + reaction)
            if actor.respond(fallen, total):
                actor.hold_clear(total)
            else:
                actor.live(total)
        else:
            actor.live(total, fall if rid == faller else None)
        if rid == faller:
            total = max(total, actor.t)
        home.settle(actor)
        planned[rid] = actor
    # Someone's last action may run past the end; everyone else carries on until then too,
    # still keeping out of the way.
    total = max(a.t for a in planned.values())
    for actor in planned.values():
        actor.hold_clear(total)
        home.settle(actor)
    actors = [planned[rid] for rid in ids]
    if outage:
        _insert_outage(actors, rng)

    falls = [s for a in actors for s in a.falls]
    tags: dict[str, TagValue] = {
        "generator": GENERATOR_VERSION, "spec": spec.key(), "room": spec.room, "people": spec.people,
        "falls": len(falls), "outage": any(s.label == "camera_outage" for a in actors for s in a.segments),
    }  # fmt: skip
    if falls:
        tags |= {"fall_kind": str(falls[0].tags["fault"]), "fall_outcome": str(falls[0].tags["outcome"])}
    helpers = [a for a in actors if any(s.tags.get("responder") for s in a.segments)]
    tags["responder"] = bool(helpers)
    if helpers and faller is not None:
        tags["response_s"] = round(_arrival(helpers[0]) - planned[faller].fall_events[0][0], 2)
    unresolved = sum(a.unresolved for a in actors)
    if unresolved:
        tags["clashes"] = unresolved
    return Scenario(
        scenario_id=scenario_id(spec, seed),
        title=_title(spec, actors),
        description=_description(actors),
        seed=seed,
        fps=spec.fps,
        scene=scene,
        variation=Variation.none(),
        tags=tags,
        residents=[a.resident() for a in actors],
    )


def _timed(segments: list[Segment]) -> list[tuple[float, Segment]]:
    """(start time, segment) pairs."""
    out, t = [], 0.0
    for s in segments:
        out.append((t, s))
        t += s.duration
    return out


def _arrival(helper: Actor) -> float:
    """When a responder reaches the faller: the start of their first kneel."""
    return next(t for t, s in _timed(helper.segments) if s.tags.get("responder"))


def _arrive(home: Household, rid: str, persona, rng: random.Random, spec: GenSpec) -> Actor:
    """A resident and their opening moments, placed out of the way of everyone planned so far."""
    for _ in range(8):
        actor = Actor(home, rid, persona, rng)
        _start(actor, spec)
        if home.clash(actor, 0.0, actor.t) is None:
            return actor
    actor.unresolved += 1
    return actor


def _start(actor: Actor, spec: GenSpec) -> None:
    """Begin in the room (standing, or already seated), or outside, coming in later."""
    plan, rng = actor.plan, actor.rng
    if rng.random() >= spec.start_inside:
        door = rng.choice(list(plan.doors))
        actor.place(plan.doors[door])
        actor.state = "absent"
        actor._seg("absent", rng.uniform(1.0, 0.3 * spec.duration))
        actor.state = "standing"
        actor.go(plan.random_point(rng))
        return
    seat = actor.free_seat(("chair", "couch"), 10.0) if rng.random() < 0.4 else None
    if seat is not None:
        actor.place(seat.at(plan.scene))
        actor.sit(seat, rng.uniform(3.0, 12.0))
        actor.segments[0] = actor.segments[0].model_copy(update={"transition": 0.0})
        return
    actor.place(plan.random_point(rng), facing=rng.choice(["camera", "left", "right"]))
    actor._seg("standing", rng.uniform(1.0, 3.0))


STATIC = frozenset({"standing", "sitting", "lying_intentional", "bending"})


def _insert_outage(actors: list[Actor], rng: random.Random) -> None:
    """Black the camera out for a few seconds, during one resident's still moment.

    The outage is a segment in one resident's script (it silences the whole camera). It
    replaces the middle of a still activity, so nobody's motion changes. Falls are never
    split: one fall must stay one expected alert.
    """
    candidates = []
    for a in actors:
        t = 0.0
        for i, seg in enumerate(a.segments):
            settled = t + seg.transition + 0.5
            if seg.label in STATIC and t + seg.duration - settled >= 5.0:
                candidates.append((a, i, settled, t + seg.duration))
            t += seg.duration
    if not candidates:
        return
    a, i, t0, t1 = rng.choice(candidates)
    length = round(rng.uniform(4.0, min(10.0, t1 - t0 - 0.5)), 3)
    begin = round(rng.uniform(t0, t1 - length - 0.5), 3)
    seg = a.segments[i]
    seg_start = sum(s.duration for s in a.segments[:i])
    before = round(begin - seg_start, 3)
    after = round(seg.duration - before - length, 3)
    a.segments[i : i + 1] = [
        seg.model_copy(update={"duration": before}),
        Segment(label="camera_outage", duration=length, transition=0.0, tags={"fault": "outage"}),
        seg.model_copy(update={"duration": after, "transition": 0.0, "x": None, "depth": None}),
    ]


def _title(spec: GenSpec, actors: list[Actor]) -> str:
    who = "1 person" if spec.people == 1 else f"{spec.people} people"
    falls = [(a.resident_id, f) for a in actors for f in a.falls]
    if not falls:
        return f"Generated: {who} in the {spec.room.replace('-', ' ')}, no falls"
    rid, f = falls[0]
    return f"Generated: {who} in the {spec.room.replace('-', ' ')}, {rid} falls ({f.tags['fault']}, {f.tags['outcome']})"


def _description(actors: list[Actor]) -> str:
    lines = []
    for a in actors:
        labels: list[str] = []
        for s in a.segments:
            if not labels or labels[-1] != s.label:
                labels.append(s.label)
        lines.append(f"{a.resident_id} ({a.persona.name}, {a.speed} m/s): " + " > ".join(labels))
    return "\n".join(lines)


def close_calls(scenario: Scenario, *, step: float = 0.1, tolerance: float = 0.05) -> list[tuple[float, str, str, float]]:
    """(t, person, person, gap in metres) wherever two residents come closer than personal space.

    A responder kneeling by the person they came to help doesn't count. For checking
    generated scenarios; hand-written ones may put people as close as they like.
    """
    home = Household(FloorPlan(scenario.scene))
    tracks = {r.resident_id or "": ResidentTrack(r, scenario.scene) for r in scenario.cast()}
    helpers = {r.resident_id for r in scenario.cast() if any(s.tags.get("responder") for s in r.segments)}
    fallers = {r.resident_id for r in scenario.cast() if any(s.label == "fall" for s in r.segments)}
    ids = list(tracks)
    out = []
    for k in range(int(scenario.duration / step) + 1):
        t = k * step
        bodies = {rid: home.body(tracks[rid], t) for rid in ids}
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if {a, b} & helpers and {a, b} & fallers:
                    continue
                ba, bb = bodies[a], bodies[b]
                if ba and bb and (g := gap(ba, bb)) < PERSONAL_SPACE_M - tolerance:
                    out.append((round(t, 3), a, b, round(g, 3)))
    return out
