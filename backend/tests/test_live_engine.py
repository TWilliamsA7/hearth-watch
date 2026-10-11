"""The live engine: procedural rooms, autonomous people, conversations, incidents, camera, truth."""

from __future__ import annotations

import math
from collections import Counter

import pytest
from hearth_schema.models import Observation

from hearth.eval.real_monitor import create_monitor
from hearth.sim.live.engine import DT, World
from hearth.sim.live.incidents import IncidentError
from hearth.sim.live.room import generate_room
from hearth.sim.live.spec import EngineSpec, IncidentSpec, RoomSpec

LEGAL = {
    ("standing", "sitting"): {"sitting down"},
    ("sitting", "standing"): {"standing up"},
    ("sitting", "lying"): {"lying down"},
    ("lying", "sitting"): {"sitting up"},
    ("standing", "floor"): {"falling"},
    ("sitting", "floor"): {"falling"},
    ("floor", "standing"): {"getting up"},
    ("standing", "absent"): {"away"},
    ("absent", "standing"): {"entering"},
}
"""(posture before, after) -> activities allowed to make that change."""


@pytest.fixture(scope="module")
def long_run() -> dict:
    """Eight people, four minutes, with everything worth checking recorded per tick."""
    w = World(EngineSpec(actors=8), 3)
    last = {a.person_id: (a.pos, a.posture) for a in w.actors}
    out = {"close": Counter(), "jumps": [], "transitions": [], "double": 0, "convs": set(), "intents": {},
           "facing": [], "stalled": Counter()}  # fmt: skip
    while w.t < 240:
        doing = {a.person_id: a.activity.name if a.activity else "idle" for a in w.actors}
        w.step()
        for a in w.actors:
            pos, posture = last[a.person_id]
            # The activity that changed the posture may finish on the same tick: take it from either side.
            act = a.activity.name if a.activity and posture == a.posture else doing[a.person_id]
            if posture != a.posture and act in ("idle", "turning") and a.activity:
                act = a.activity.name
            if a.present and posture != "absent" and act != "entering" and math.dist(pos, a.pos) > 0.1:
                out["jumps"].append((a.person_id, w.t, act, math.dist(pos, a.pos)))
            if posture != a.posture:
                out["transitions"].append((posture, a.posture, act))
            if a.present and act == "walking" and a.speed < 0.02:
                out["stalled"][a.person_id] += 1
            last[a.person_id] = (a.pos, a.posture)
            if a.conversation:
                out["convs"].add(a.conversation)
        floor = [a for a in w.actors if a.present and a.on_floor_space]
        for i, a in enumerate(floor):
            for b in floor[i + 1 :]:
                if math.dist(a.pos, b.pos) < 0.45:
                    out["close"][(a.person_id, b.person_id)] += 1
        out["double"] += sum(1 for a in w.actors for sid in a.slots if w.room.slot(sid).occupant != a.person_id)
        for conv in w.social.conversations.values():
            people = [w.actor(m) for m in conv.members]
            if len(people) == 2 and all(p.posture == "standing" and p.activity and p.activity.name == "talking"
                                        and p.activity.tau > 3 for p in people):  # fmt: skip
                a, b = people
                rel = abs((math.atan2(b.pos[0] - a.pos[0], b.pos[1] - a.pos[1]) - a.heading + math.pi) % (2 * math.pi) - math.pi)
                out["facing"].append((math.dist(a.pos, b.pos), math.degrees(rel)))
        for lie in (s for s in w.room.slots if s.kind == "lie" and s.occupant):  # nobody sits on someone lying
            assert {s.occupant for s in w.room.slots if s.group == lie.group and s.occupant} == {lie.occupant}
    out["intents"] = {a.person_id: [i for i, _ in a.history] for a in w.actors}
    out["world"] = w
    return out


def test_rooms_are_reproducible_and_varied() -> None:
    a, b = generate_room(RoomSpec(), 11), generate_room(RoomSpec(), 11)
    assert [(p.kind.name, p.at, p.rotation) for p in a.pieces] == [(p.kind.name, p.at, p.rotation) for p in b.pieces]
    layouts = {tuple(p.kind.name for p in generate_room(RoomSpec(), s).pieces) for s in range(8)}
    assert len(layouts) > 4


@pytest.mark.parametrize("seed", range(6))
def test_every_slot_and_door_can_be_reached(seed: int) -> None:
    room = generate_room(RoomSpec(furniture_density=1.0), seed)
    reach = room.nav.reachable(room.doors[0].inside)
    assert room.slots, "somewhere to sit"
    for slot in room.slots:
        approach = room.approach(slot, 1.7)
        assert approach is not None and room.nav._cell(approach) in reach, slot.slot_id
    for door in room.doors:
        assert room.nav._cell(door.inside) in reach


@pytest.mark.parametrize("seed", range(6))
def test_furniture_stays_inside_the_walls_and_apart(seed: int) -> None:
    room = generate_room(RoomSpec(furniture_density=1.0), seed)
    x0, z0, x1, z1 = room.bounds
    solid = [p for p in room.pieces if not p.kind.walkable]
    for p in solid:
        f = p.footprint()
        assert x0 - 1e-6 <= f[0] and f[2] <= x1 + 1e-6 and z0 - 1e-6 <= f[1] and f[3] <= z1 + 1e-6
    for i, p in enumerate(solid):
        for q in solid[i + 1 :]:
            fp, fq = p.footprint(), q.footprint()
            assert not (fp[0] < fq[2] and fp[2] > fq[0] and fp[1] < fq[3] and fp[3] > fq[1]), (p.kind.name, q.kind.name)
    for d in room.doors:
        assert d.at[0] in (x0, x1) or d.at[1] in (z0, z1)


def test_same_spec_and_seed_same_run() -> None:
    def run() -> list[str]:
        w = World(EngineSpec(actors=4), 5)
        return [r.model_dump_json() for r in w.advance(20)]

    assert run() == run()


def test_nobody_walks_into_anybody(long_run: dict) -> None:
    assert not long_run["close"], long_run["close"]


def test_nobody_teleports(long_run: dict) -> None:
    assert not long_run["jumps"], long_run["jumps"][:5]


def test_posture_changes_go_through_a_transition(long_run: dict) -> None:
    assert long_run["transitions"]
    for before, after, act in long_run["transitions"]:
        assert act in LEGAL[(before, after)], (before, after, act)


def test_nobody_is_stuck_for_long(long_run: dict) -> None:
    # A few seconds of waiting for someone to pass is fine; standing in a walk for long isn't.
    for pid, ticks in long_run["stalled"].items():
        assert ticks * DT < 20.0, (pid, ticks * DT)


def test_seats_are_never_double_booked(long_run: dict) -> None:
    assert long_run["double"] == 0


def test_people_do_varied_things(long_run: dict) -> None:
    every = Counter(i for intents in long_run["intents"].values() for i in intents)
    assert len(every) >= 6, every
    for pid, intents in long_run["intents"].items():
        assert len(set(intents)) >= 2, (pid, intents)
        loops = sum(1 for k in range(len(intents) - 3) if intents[k] == intents[k + 2] and intents[k + 1] == intents[k + 3]
                    and intents[k] != intents[k + 1])  # fmt: skip
        assert loops <= 2, (pid, intents)


def test_conversations_happen_face_to_face(long_run: dict) -> None:
    assert len(long_run["convs"]) >= 2
    for dist, rel in long_run["facing"]:
        assert 0.6 <= dist <= 1.8, dist
        assert rel <= 40, rel


def test_ground_truth_is_contiguous_per_person(long_run: dict) -> None:
    w = long_run["world"]
    truth = w.ground_truth()
    for a in w.actors:
        segs = [g for g in truth if g.person_id == a.person_id]
        assert segs[0].t_start == 0.0
        for g, h in zip(segs, segs[1:]):
            assert g.t_end == h.t_start
        assert segs[-1].t_end == pytest.approx(w.t)
    assert not any(g.expects_alert for g in truth)  # no incidents unless configured


def test_camera_records_are_contract_records(long_run: dict) -> None:
    w = World(EngineSpec(actors=3), 2)
    recs = w.advance(10)
    obs = [r for r in recs if isinstance(r, Observation)]
    assert obs and all(o.source_type == "simulation" and o.scenario_id == w.scenario_id for o in obs)
    seen = [o for o in obs if o.person_detected]
    assert seen and all(o.person_id and len(o.keypoints or []) >= 4 for o in seen)
    assert sum(1 for r in recs if not isinstance(r, Observation)) in (10, 11)  # a heartbeat a second, from t = 0


def test_procedural_incidents_are_off_by_default() -> None:
    w = World(EngineSpec(actors=6), 4)
    w.advance(120)
    assert w.incidents.records == []


def test_procedural_incidents_when_configured() -> None:
    w = World(EngineSpec(actors=6, incident_rates={"inactivity": 600.0}), 4)
    w.advance(60)
    assert any(r.source == "procedural" for r in w.incidents.records)


def test_a_manual_fall_is_ground_truth_and_the_monitor_sees_it() -> None:
    # A seed whose fall the rule-based monitor catches (it misses some camera angles).
    w = World(EngineSpec(actors=3), 2)
    monitor = create_monitor()
    for r in w.advance(12):
        monitor.ingest(r)
    rec = w.incidents.trigger(IncidentSpec(kind="fall", duration=20))
    events = []
    for r in w.advance(30):
        events.extend(monitor.ingest(r))
    assert any(e.event_type == "possible_fall" and e.person_id == rec.person_id for e in events)
    falls = [g for g in w.ground_truth() if g.truth_label == "fall"]
    assert len(falls) == 1 and falls[0].expects_alert and falls[0].person_id == rec.person_id
    assert falls[0].t_start == pytest.approx(rec.t_start, abs=0.1)
    assert rec.t_floor is not None and rec.t_recovered is not None and rec.posture_after == "standing"


def test_a_fall_that_doesnt_recover_waits_for_reset() -> None:
    w = World(EngineSpec(actors=2), 1)
    w.advance(5)
    rec = w.incidents.trigger(IncidentSpec(kind="fall", actor="p1", recover=False))
    w.advance(90)
    assert w.actor("p1").posture == "floor" and rec.t_end is None
    w.incidents.reset("p1")
    w.advance(10)
    assert w.actor("p1").posture == "standing" and rec.t_end is not None


@pytest.mark.parametrize("kind", ["inactivity", "pacing", "exit_loiter"])
def test_other_incidents_run_and_finish(kind: str) -> None:
    w = World(EngineSpec(actors=3), 6)
    w.advance(5)
    pid = next(a.person_id for a in w.actors if a.present)
    rec = w.incidents.trigger(IncidentSpec(kind=kind, actor=pid, duration=8, repeats=2))
    assert w.actor(pid).incident == kind
    w.advance(90)
    assert rec.t_end is not None and w.actor(pid).incident is None


def test_incidents_that_cant_happen_say_why() -> None:
    w = World(EngineSpec(actors=1), 1)
    w.advance(1)
    w.incidents.trigger(IncidentSpec(kind="fall", actor="p1", recover=False))
    with pytest.raises(IncidentError):
        w.incidents.trigger(IncidentSpec(kind="fall", actor="p1"))


def test_scheduled_incidents_fire_on_time() -> None:
    w = World(EngineSpec(actors=3, incidents=[IncidentSpec(kind="inactivity", t=10.0, duration=5)]), 2)
    w.advance(9.5)
    assert not w.incidents.records
    w.advance(1.0)
    assert [r.source for r in w.incidents.records] == ["scheduled"]


def test_the_live_world_lists_its_slots_places_and_walkable_floor() -> None:
    w = World(EngineSpec(actors=3), 4)
    lw = w.live_world("run")
    assert {s.slot_id for s in lw.slots} == {s.slot_id for s in w.room.slots}
    for s in lw.slots:
        assert 0 <= s.piece < len(lw.room.objects)
        obj = lw.room.objects[s.piece]
        assert s.height == (obj.seat_height if s.kind == "sit" else obj.surface_height)
        if s.kind == "sit":
            assert s.stand is not None and s.approach is not None
    assert {p.name for p in lw.places} >= {"middle", "door"}
    nav = lw.nav
    assert nav is not None and len(nav.rows) > 10 and all(len(r) == nav.cols for r in nav.rows)
    blocked = sum(r.count("1") for r in nav.rows)
    assert 0 < blocked < nav.cols * len(nav.rows)
    # Every place is on walkable floor.
    for p in lw.places:
        i, j = round((p.at[0] - nav.origin[0]) / nav.cell), round((p.at[1] - nav.origin[1]) / nav.cell)
        assert nav.rows[j][i] == "0", p


def test_frames_say_who_holds_which_slot() -> None:
    w = World(EngineSpec(actors=4), 4)
    w.advance(1)
    taken = w.frame().slots_taken
    assert taken == {s.slot_id: s.occupant for s in w.room.slots if s.occupant}
    assert set(taken.values()) <= {a.person_id for a in w.actors}
