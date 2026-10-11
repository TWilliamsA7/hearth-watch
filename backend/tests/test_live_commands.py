"""Telling a simulated person what to do: commands replace their plan and play out as their
own choices would, and refuse what they can't do."""

from __future__ import annotations

import math

import pytest
from test_live_api import make_client

from hearth.sim.live.animate import world_pose
from hearth.sim.live.commands import ActorCommand, CommandError, apply_command
from hearth.sim.live.engine import World
from hearth.sim.live.fit import penetration
from hearth.sim.live.spec import EngineSpec, IncidentSpec


def _free(w: World, kind: str, slot_kind: str = "sit") -> list:
    return [s for s in w.room.slots if w.room.pieces[s.piece].kind.name == kind and s.kind == slot_kind
            and s.occupant is None and not w.room.group_busy(s.group)]  # fmt: skip


def _world(kind: str, slot_kind: str = "sit") -> World:
    """A room with a free `kind` to use, and room for someone else to use one too."""
    for seed in range(60):
        w = World(EngineSpec(actors=2), seed)
        if len(_free(w, kind, slot_kind)) >= 1:
            return w
    raise AssertionError(f"no room with a free {kind}")


def _free_slot(w: World, kind: str, slot_kind: str = "sit"):
    return _free(w, kind, slot_kind)[0]


def _until(w: World, done, limit: float = 60.0) -> None:
    t0 = w.t
    while not done():
        assert w.t - t0 < limit, "took too long"
        w.step()


@pytest.mark.parametrize("kind", ["armchair", "couch", "bed"])
def test_sit_where_told(kind: str) -> None:
    w = _world(kind)
    a = next(x for x in w.actors if x.present)
    seat = _free_slot(w, kind)
    apply_command(w, a.person_id, ActorCommand(action="sit", slot_id=seat.slot_id))
    assert a.intent == "command" and seat.occupant == a.person_id
    _until(w, lambda: a.posture == "sitting" and a.pos == seat.at and a.activity is not None and a.activity.name == "seated")
    assert seat.slot_id in a.slots
    assert penetration(world_pose(a), w.room.pieces[seat.piece], a.size).depth <= 0.03


def test_lie_where_told_and_get_up_again() -> None:
    w = _world("bed", "lie")
    a = next(x for x in w.actors if x.present)
    lie = _free_slot(w, "bed", "lie")
    apply_command(w, a.person_id, ActorCommand(action="lie", slot_id=lie.slot_id))
    _until(w, lambda: a.posture == "lying" and a.activity is not None and a.activity.name == "resting")
    assert lie.slot_id in a.slots
    apply_command(w, a.person_id, ActorCommand(action="go", at=w.room.places[0].at))
    _until(w, lambda: a.posture == "standing" and not a.slots)


def test_kneel_and_pick_up_at_a_point_then_carry_on() -> None:
    w = World(EngineSpec(actors=2), 3)
    a = next(x for x in w.actors if x.present)
    spot = next(p.at for p in w.room.places if p.name == "middle")
    apply_command(w, a.person_id, ActorCommand(action="kneel", at=spot))
    _until(w, lambda: a.activity is not None and a.activity.name == "kneeling")
    assert math.dist(a.pos, spot) < 0.15 and a.label == "bending"
    _until(w, lambda: a.intent != "command")  # done: the Brain has them again
    assert a.posture in ("standing", "sitting")


def test_fall_where_told() -> None:
    w = World(EngineSpec(actors=2), 3)
    a = next(x for x in w.actors if x.present)
    spot = next(p.at for p in w.room.places if p.name == "middle")
    apply_command(w, a.person_id, ActorCommand(action="fall", at=spot))
    _until(w, lambda: a.posture == "floor")
    rec = w.incidents.records[-1]
    assert rec.person_id == a.person_id and rec.source == "manual" and rec.kind == "fall"
    assert math.dist(a.pos, spot) < 1.0


def test_commands_refuse_what_cant_be_done() -> None:
    w = _world("couch")
    a, b = [x for x in w.actors if x.present][:2]
    seat = _free_slot(w, "couch")
    apply_command(w, a.person_id, ActorCommand(action="sit", slot_id=seat.slot_id))
    with pytest.raises(CommandError, match="has"):
        apply_command(w, b.person_id, ActorCommand(action="sit", slot_id=seat.slot_id))
    blocked = next((x, z) for x in (p.at[0] for p in w.room.pieces) for z in (p.at[1] for p in w.room.pieces)
                   if not w.room.nav.free((x, z)))  # fmt: skip
    with pytest.raises(CommandError, match="open floor"):
        apply_command(w, b.person_id, ActorCommand(action="go", at=blocked))
    with pytest.raises(CommandError, match="needs a slot_id"):
        apply_command(w, b.person_id, ActorCommand(action="sit"))
    with pytest.raises(KeyError):
        apply_command(w, b.person_id, ActorCommand(action="sit", slot_id="nope"))
    w.incidents.trigger(IncidentSpec(kind="fall", actor=b.person_id))
    with pytest.raises(CommandError, match="in progress"):
        apply_command(w, b.person_id, ActorCommand(action="go", at=w.room.places[0].at))


def test_the_command_route() -> None:
    client, _, _, live = make_client()
    with client:
        assert client.post("/sim/live/actors/p1/command", json={"action": "go", "at": [0, -2]}).status_code == 409  # no room yet
        world = client.post("/sim/live/new", json={"seed": 3, "spec": {"actors": 2}}).json()
        seat = next(s for s in world["slots"] if s["kind"] == "sit")
        pid = world["actors"][0]["person_id"]
        r = client.post(f"/sim/live/actors/{pid}/command", json={"action": "sit", "slot_id": seat["slot_id"]})
        assert r.status_code == 200, r.text
        assert r.json()["now"]["intent"] == "command"
        assert client.post("/sim/live/actors/nobody/command", json={"action": "go", "at": [0, -2]}).status_code == 404
        assert client.post(f"/sim/live/actors/{pid}/command", json={"action": "sit", "slot_id": "nope"}).status_code == 404
        r = client.post(f"/sim/live/actors/{pid}/command", json={"action": "go", "at": [99, 99]})
        assert r.status_code == 409 and "open floor" in r.json()["detail"]
