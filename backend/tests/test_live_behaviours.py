"""Everyday behaviours the Brain can choose: eating at a table, watching TV, and the rest."""

from __future__ import annotations

import dataclasses
import math

import pytest

from hearth.sim.live.activities import PLANNERS, Hold, facing_tv, get_up, reserve
from hearth.sim.live.animate import gait, heading_to, idle, world_pose, wrap
from hearth.sim.live.engine import World
from hearth.sim.live.incidents import IncidentError
from hearth.sim.live.spec import EngineSpec, IncidentSpec
from hearth.sim.pose import KEY_POSES


def _world_where(ok, actors: int = 2) -> World:
    for seed in range(80):
        w = World(EngineSpec(actors=actors), seed)
        if ok(w):
            return w
    raise AssertionError("no such room")


def _do(w: World, intent: str):
    """Have someone standing carry out a planner's plan; returns them once it's holding."""
    a = next(x for x in w.actors if x.present)
    plan = PLANNERS[intent](a, w)
    assert plan, f"{intent}: nothing to do here"
    held = [w.room.slot(s) for s in a.slots]
    w.replace_plan(a, plan, intent)
    reserve(a, w, *held)  # replace_plan frees seats not yet sat in; the Brain's own plans keep them
    t0 = w.t
    while not (isinstance(a.activity, Hold) and a.activity.name not in ("seated", "pausing")):
        assert w.t - t0 < 60, f"stuck in {a.activity.name if a.activity else None}"
        w.step()
    return a


def test_eating_happens_at_a_table() -> None:
    w = _world_where(lambda w: any(p.kind.name == "table" for p in w.room.pieces))
    a = _do(w, "table")
    assert a.activity.name == "eating" and a.posture == "sitting" and a.label == "sitting"
    chair = w.room.pieces[w.room.slot(a.slots[0]).piece]
    assert any(p.kind.name == "table" and p.group == chair.group for p in w.room.pieces)


def test_eating_brings_a_hand_to_the_mouth_now_and_then() -> None:
    w = World(EngineSpec(actors=1), 0)
    a = w.actors[0]
    a.posture, a.look_at, a.idle_phase = "sitting", None, 0.5
    rest = KEY_POSES["sitting"]
    heights = [idle(a, rest, k * 0.1, 0.1, sway=0.0, eating=True)["right_wrist"][1] for k in range(80)]
    assert max(heights) - rest["right_wrist"][1] > 0.2  # up by the mouth
    assert min(heights) == pytest.approx(rest["right_wrist"][1], abs=0.01)  # and back in the lap
    nose = rest["nose"]
    top = idle(a, rest, heights.index(max(heights)) * 0.1, 0.1, sway=0.0, eating=True)["right_wrist"]
    assert math.dist(top, nose) < 0.2


def test_tv_is_watched_from_a_seat_facing_it() -> None:
    w = _world_where(lambda w: any(facing_tv(w, s) for s in w.room.slots if s.kind == "sit"))
    a = _do(w, "tv")
    assert a.activity.name == "watching TV" and a.posture == "sitting"
    seat = w.room.slot(a.slots[0])
    tv = facing_tv(w, seat)
    assert tv is not None and a.look_at == tv
    assert abs(wrap(heading_to(seat.at, tv) - seat.heading)) <= math.radians(35)


def test_no_tv_no_watching() -> None:
    w = _world_where(lambda w: not any(p.kind.name == "cabinet" for p in w.room.pieces))
    a = next(x for x in w.actors if x.present)
    assert PLANNERS["tv"](a, w) is None


def test_tying_a_shoe_is_bending_then_standing_again() -> None:
    w = World(EngineSpec(actors=1), 2)
    a = _do(w, "tie_shoe")
    assert a.activity.name == "tying a shoe" and a.label == "bending" and a.posture == "standing"
    while a.intent == "tie_shoe":
        w.step()
    assert a.posture == "standing"


def test_reaching_low_happens_in_front_of_a_cabinet_or_shelf() -> None:
    w = _world_where(lambda w: any(p.name == "browse" for p in w.room.places))
    a = _do(w, "reach_low")
    assert a.activity.name == "reaching low" and a.label == "bending"
    place = min((p for p in w.room.places if p.name == "browse"), key=lambda p: math.dist(p.at, a.pos))
    assert math.dist(place.at, a.pos) < 0.2


def test_sitting_on_the_floor_is_sitting_not_a_fall() -> None:
    w = World(EngineSpec(actors=1), 2)
    a = _do(w, "floor_sit")
    assert a.posture == "floor_sit" and a.label == "sitting" and a.activity.name == "sitting on the floor"
    assert max(j[1] for j in world_pose(a).values()) < 1.0  # down on the floor
    with pytest.raises(IncidentError, match="already on the floor"):
        w.incidents.trigger(IncidentSpec(kind="fall", actor=a.person_id))
    w.replace_plan(a, get_up(a, w), "test")
    while a.posture != "standing":
        w.step()
    assert not [g for g in w.ground_truth() if g.truth_label == "fall"]


def test_someone_frail_shuffles() -> None:

    w = World(EngineSpec(actors=2), 0)
    steady, frail = w.actors[0], w.actors[1]
    frail.profile = dataclasses.replace(frail.profile, mobility="slow")
    steady.profile = dataclasses.replace(steady.profile, mobility="normal", height=frail.profile.height)
    for a in (steady, frail):
        a.gait_phase = 0.0
        for _ in range(5):  # a quarter of a metre at 1 m/s: less than one stride
            gait(a, 1.0, 0.05)
    assert frail.gait_phase == pytest.approx(steady.gait_phase / 0.75, rel=1e-6)  # more, shorter steps
