import math
import random

import pytest
from hearth.sim.floorplan import FloorPlan, distance, route_segments, to_frame, to_world
from hearth.sim.generate import ROOMS
from hearth.sim.scenario import Scene, SceneObject, load_all
from hearth.sim.world import room_geometry

DAY_ROOM = load_all()["day-room-02"].scene


@pytest.mark.parametrize(("x", "depth"), [(0.0, 0.0), (0.3, 0.6), (1.15, 0.1), (0.9, 1.0)])
def test_frame_and_world_round_trip(x, depth):
    assert to_frame(DAY_ROOM, to_world(DAY_ROOM, x, depth)) == pytest.approx((x, depth))


def test_doors_are_off_camera_and_the_middle_is_not():
    plan = FloorPlan(DAY_ROOM)
    assert not plan.visible(plan.doors["left"]) and not plan.visible(plan.doors["right"])
    assert plan.visible(to_world(DAY_ROOM, 0.5, 0.5))


def _assert_walkable(plan: FloorPlan, route, ignore=()):
    for a, b in zip(route, route[1:]):
        assert plan.line_free(a, b, ignore), (a, b)


def test_routes_go_around_furniture():
    plan = FloorPlan(DAY_ROOM)
    rng = random.Random(3)
    for _ in range(30):
        a, b = plan.random_point(rng, visible=False), plan.random_point(rng, visible=False)
        route = plan.find_path(a, b)
        assert route is not None and route[0] == a and route[-1] == b
        _assert_walkable(plan, route)
        assert sum(distance(p, q) for p, q in zip(route, route[1:])) >= distance(a, b) - 1e-9


def test_routes_to_a_seat_approach_from_the_front():
    plan = FloorPlan(DAY_ROOM)
    for seat in plan.seats():
        route = plan.route_to_seat(plan.doors["left"], seat)
        assert route is not None and route[-1] == seat.at(DAY_ROOM)
        _assert_walkable(plan, route[:-1])
        assert route[-2] == plan.approach(seat)
        assert route[-2][0] == pytest.approx(route[-1][0])  # the last step is straight back into the seat


def test_no_route_through_a_wall_of_furniture():
    # A dresser across the whole room near the back wall.
    wall = SceneObject(kind="occluder", x=0.0, y=0.40, w=1.0, h=0.92 - 0.8 * 0.14 - 0.40)
    plan = FloorPlan(Scene(objects=[wall]))
    assert plan.find_path(to_world(plan.scene, 0.5, 0.2), to_world(plan.scene, 0.5, 0.93)) is None


def test_route_segments_walk_each_leg_at_the_given_speed():
    plan = FloorPlan(DAY_ROOM)
    route = plan.route_to_seat(plan.doors["right"], plan.seats(["chair"])[0])
    segments = route_segments(DAY_ROOM, route, speed=0.8)
    assert all(s.label == "walking" for s in segments)
    total = sum(distance(a, b) for a, b in zip(route, route[1:]))
    assert sum(s.duration for s in segments) == pytest.approx(total / 0.8, abs=0.01)
    assert (segments[-1].x, segments[-1].depth) == pytest.approx(to_frame(DAY_ROOM, route[-1]), abs=1e-3)


def test_places_are_on_open_floor_in_view_and_face_something():
    for scene in (DAY_ROOM, *ROOMS.values()):
        plan = FloorPlan(scene)
        places = plan.places()
        names = {p.name for p in places}
        assert "middle" in names and "window" in names
        for p in places:
            assert plan.visible(p.at, margin=0.05), p
            assert p.name.startswith("on-") or plan.free(p.at), p
            if p.near is not None and not p.name.startswith("on-"):
                # Facing the furniture it is beside.
                o = room_geometry(scene).objects[p.near]
                to = math.atan2(o.center[0] - p.at[0], o.center[2] - p.at[1])
                assert abs(math.remainder(to - p.heading, 2 * math.pi)) < math.radians(60), p
        if any(o.kind != "mat" for o in scene.objects):
            assert any(p.near is not None for p in places)


def test_routes_keep_off_furniture_where_there_is_room():
    plan = FloorPlan(DAY_ROOM)
    rng = random.Random(5)
    clear = []
    for _ in range(30):
        a, b = plan.random_point(rng), plan.random_point(rng)
        route = plan.find_path(a, b)
        assert route is not None
        _assert_walkable(plan, route)
        clear += [plan.clearance(p) for p in route[1:-1]]
    assert sum(clear) / max(len(clear), 1) > 0.3  # turning points sit out in the open, not on a corner


def test_corners_are_rounded():
    plan = FloorPlan(DAY_ROOM)
    route = plan.find_path(to_world(DAY_ROOM, 0.05, 0.9), to_world(DAY_ROOM, 0.95, 0.05))
    assert route is not None
    turns = []
    for p, q, r in zip(route, route[1:], route[2:]):
        a, b = math.atan2(q[0] - p[0], q[1] - p[1]), math.atan2(r[0] - q[0], r[1] - q[1])
        turns.append(abs(math.remainder(b - a, 2 * math.pi)))
    assert max(turns, default=0.0) < math.radians(50)
