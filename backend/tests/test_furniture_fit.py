"""Bodies and furniture: people sit on seats and lie on beds and couches without going into them.

The furniture is room.collision_boxes() (what the 3D view draws, from the same numbers);
a body is fit.spheres() round its keypoints. A body may touch furniture, and sink into it
by up to TOL (cushions give), but no more. Checked three ways: the final sitting and lying
stances on their own, every frame of sitting down, lying down and getting up again, and
whole runs of the live engine.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from hearth.sim.live.activities import Hold, Walk, get_up, lie_plan, lie_stance, release, reserve, seat_stance, sit_plan
from hearth.sim.live.animate import idle, world_pose
from hearth.sim.live.engine import World
from hearth.sim.live.fit import Overlap, lowest, penetration, seat_gap
from hearth.sim.live.room import KINDS, Piece, Slot, _slots, generate_room
from hearth.sim.live.spec import EngineSpec, RoomSpec
from hearth.sim.pose import KEY_POSES
from hearth.sim.world import body

TOL = 0.03  # metres a body may sink into furniture (cushions give a little)
HEIGHTS = (1.52, 1.69, 1.86)  # the range make_profile draws from
BUILDS = (1.0, 1.15)  # Appearance.build: the 3D view widens the body by this
SEATED = ("chair", "armchair", "couch", "loveseat", "bed")
LYING = ("couch", "loveseat", "bed")


def _top(slot: Slot, piece: Piece) -> float:
    return piece.kind.seat if slot.kind == "sit" else piece.kind.surface


def _stance_case(kind: str, slot_kind: str, rotation: float = 0.0) -> list[tuple[Piece, Slot]]:
    piece = Piece(0, KINDS[kind], (0.4, -2.0), rotation)
    return [(piece, s) for s in _slots([piece]) if s.kind == slot_kind]


# --- the final stances ---


@pytest.mark.parametrize("build", BUILDS)
@pytest.mark.parametrize("size", HEIGHTS)
@pytest.mark.parametrize("kind", SEATED)
def test_a_seated_body_rests_on_the_seat(kind: str, size: float, build: float) -> None:
    for piece, slot in _stance_case(kind, "sit"):
        st = seat_stance(slot, size)
        joints = body(st.body, st.heading, st.x, st.z, st.base, size)
        o = penetration(joints, piece, size, build)
        assert o.depth <= TOL, f"{slot.slot_id}: {o}"
        gap = seat_gap(joints, size, _top(slot, piece))
        assert -TOL <= gap <= TOL, f"{slot.slot_id}: hips {gap:+.3f} m from the seat"


@pytest.mark.parametrize("build", BUILDS)
@pytest.mark.parametrize("size", HEIGHTS)
@pytest.mark.parametrize("kind", LYING)
def test_a_lying_body_lies_on_the_surface_and_inside_the_arms(kind: str, size: float, build: float) -> None:
    for piece, slot in _stance_case(kind, "lie"):
        st = lie_stance(slot, piece, size)
        if st is None:
            assert kind == "loveseat" and size > 1.5, f"{kind}: nowhere to lie for someone {size} m tall"
            continue  # they rest somewhere longer
        joints = body(st.body, st.heading, st.x, st.z, st.base, size)
        o = penetration(joints, piece, size, build)
        assert o.depth <= TOL, f"{slot.slot_id}: {o}"
        # Resting on it: the underside of the body on the surface. (The hips themselves sit
        # higher when lying on one side: the width of the pelvis is under them.)
        rest = lowest(joints, size) - _top(slot, piece)
        assert -TOL <= rest <= TOL, f"{slot.slot_id}: underside {rest:+.3f} m from the surface"
        gap = seat_gap(joints, size, _top(slot, piece))
        assert -TOL <= gap <= 0.08, f"{slot.slot_id}: hips {gap:+.3f} m from the surface"


@pytest.mark.parametrize("rotation", [math.pi / 2, math.pi, -math.pi / 2])
def test_fit_follows_the_piece_round(rotation: float) -> None:
    for kind in ("armchair", "bed"):
        for piece, slot in _stance_case(kind, "sit", rotation):
            st = seat_stance(slot, 1.69)
            joints = body(st.body, st.heading, st.x, st.z, st.base, 1.69)
            assert penetration(joints, piece, 1.69).depth <= TOL, f"{kind} turned {rotation:.2f}: {slot.slot_id}"


@pytest.mark.parametrize("size", HEIGHTS)
@pytest.mark.parametrize("kind", ("armchair", "couch", "loveseat"))
def test_talking_and_swaying_while_seated_keep_clear_of_the_arms(kind: str, size: float) -> None:
    w, a = _stage(kind, size)
    for piece, slot in _stance_case(kind, "sit"):
        st = seat_stance(slot, size)
        a.pos, a.heading, a.base, a.posture, a.rest_pose = (st.x, st.z), st.heading, st.base, "sitting", st.body
        a.slots = [slot.slot_id]
        a.look_at = None
        for phase in (0.5, 1.5):  # one arm, then the other
            a.idle_phase, a.gesture = phase, 1.0
            for k in range(200):
                t = k * 0.1
                a.body = idle(a, st.body, t, 0.1, sway=1.0, talking=True)
                a.gesture = 1.0
                o = penetration(world_pose(a), piece, size, 1.15)
                assert o.depth <= TOL, f"{slot.slot_id} at t={t:.1f}: {o}"


def test_the_measure_sees_a_body_inside_furniture() -> None:
    piece = Piece(0, KINDS["bed"], (0.0, -2.0), 0.0)
    joints = body(KEY_POSES["standing"], 0.0, piece.at[0], piece.at[1], 0.0, 1.69)  # standing in the bed
    o = penetration(joints, piece, 1.69)
    assert o.depth > 0.3 and o.part == "mattress"


# --- every frame of using a piece ---


def _room_with(kind: str) -> int:
    for seed in range(60):
        room = generate_room(RoomSpec(), seed)
        if any(room.pieces[s.piece].kind.name == kind for s in room.slots):
            return seed
    raise AssertionError(f"no room with a usable {kind}")


def _stage(kind: str, size: float) -> tuple[World, object]:
    w = World(EngineSpec(actors=1), _room_with(kind))
    a = w.actors[0]
    a.profile = dataclasses.replace(a.profile, height=size)
    release(a, w)
    w.replace_plan(a, [], "test")
    a.posture, a.base, a.body = "standing", 0.0, dict(KEY_POSES["standing"])
    return w, a


def _run(w: World, a, plan: list, pieces: list[Piece], limit: float = 40.0) -> Overlap:
    """Carry out a plan; the worst overlap with `pieces` over every frame not spent walking."""
    held = [w.room.slot(s) for s in a.slots]
    w.replace_plan(a, plan + [Hold(math.inf, "done")], "test")
    reserve(a, w, *held)  # replace_plan frees seats the person isn't in yet; the planners' callers don't
    worst = Overlap(-1e9, "", "")
    t0 = w.t
    while not (isinstance(a.activity, Hold) and a.activity.name == "done"):
        assert w.t - t0 < limit, f"stuck in {a.activity.name if a.activity else None}"
        w.step()
        if a.activity is None or isinstance(a.activity, Walk):
            continue
        for piece in pieces:
            o = penetration(world_pose(a), piece, a.size)
            if o.depth > worst.depth:
                worst = dataclasses.replace(o, where=f"{o.where} ({a.activity.name})")
    return worst


@pytest.mark.parametrize("size", HEIGHTS)
@pytest.mark.parametrize("kind", SEATED)
def test_sitting_down_and_standing_up_stay_out_of_the_furniture(kind: str, size: float) -> None:
    w, a = _stage(kind, size)
    seat = next(s for s in w.room.slots if w.room.pieces[s.piece].kind.name == kind and s.kind == "sit")
    piece = w.room.pieces[seat.piece]
    a.pos = w.room.approach(seat, size)
    reserve(a, w, seat)
    o = _run(w, a, sit_plan(a, w, seat), [piece])
    assert o.depth <= TOL, f"sitting down: {o}"
    assert a.posture == "sitting"
    gap = seat_gap(world_pose(a), size, piece.kind.seat)
    assert -TOL <= gap <= TOL, f"hips {gap:+.3f} m from the seat"
    o = _run(w, a, get_up(a, w), [piece])
    assert o.depth <= TOL, f"standing up: {o}"
    assert a.posture == "standing"


@pytest.mark.parametrize("size", HEIGHTS)
@pytest.mark.parametrize("kind", LYING)
def test_lying_down_and_getting_up_stay_out_of_the_furniture(kind: str, size: float) -> None:
    w, a = _stage(kind, size)
    lie = next(s for s in w.room.slots if w.room.pieces[s.piece].kind.name == kind and s.kind == "lie")
    entry = w.room.slot(lie.entry) if lie.entry else None
    assert entry is not None
    piece = w.room.pieces[lie.piece]
    a.pos = w.room.approach(entry, size)
    reserve(a, w, lie, entry)
    plan = lie_plan(a, w, lie, entry)
    if plan is None:
        assert kind == "loveseat" and size > 1.5, f"{kind}: nowhere to lie for someone {size} m tall"
        return  # they rest somewhere longer
    plan = [x for x in plan if not (isinstance(x, Hold) and x.name == "resting")]  # don't wait out the rest
    o = _run(w, a, plan, [piece])
    assert o.depth <= TOL, f"lying down: {o}"
    assert a.posture == "lying"
    o = _run(w, a, get_up(a, w), [piece])
    assert o.depth <= TOL, f"getting up: {o}"
    assert a.posture == "standing"


# --- whole runs ---


@pytest.mark.parametrize("seed", range(6))
def test_nobody_goes_into_the_furniture_they_use_in_a_run(seed: int) -> None:
    w = World(EngineSpec(actors=6), seed)
    worst = Overlap(-1e9, "", "")
    n = 0
    while w.t < 240:
        w.step()
        n += 1
        if n % 4:
            continue
        for a in w.actors:
            if not a.present or not a.slots or isinstance(a.activity, Walk):
                continue
            for piece in {w.room.slot(s).piece for s in a.slots}:
                o = penetration(world_pose(a), w.room.pieces[piece], a.size)
                if o.depth > worst.depth:
                    worst = dataclasses.replace(o, where=f"{a.person_id} {o.where} at t={w.t:.1f} "
                                                         f"({a.activity.name if a.activity else None})")
    assert worst.depth <= TOL, str(worst)


@pytest.mark.parametrize("kind", ("chair", "armchair", "couch", "bed"))
def test_someone_frail_pushes_up_out_of_a_seat_without_going_into_it(kind: str) -> None:
    w, a = _stage(kind, 1.62)
    a.profile = dataclasses.replace(a.profile, mobility="slow")
    seat = next(s for s in w.room.slots if w.room.pieces[s.piece].kind.name == kind and s.kind == "sit")
    piece = w.room.pieces[seat.piece]
    a.pos = w.room.approach(seat, a.size)
    reserve(a, w, seat)
    _run(w, a, sit_plan(a, w, seat), [piece])
    plan = get_up(a, w)
    assert [x.name for x in plan][-2:] == ["pushing up", "standing up"]
    o = _run(w, a, plan, [piece])
    assert o.depth <= TOL, f"standing up: {o}"
    assert a.posture == "standing"
