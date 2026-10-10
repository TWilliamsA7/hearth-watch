"""Lifting live 2D keypoints back into the simulated room (hearth.sim.live.lift)."""

import json
import math
from pathlib import Path

import pytest
from hearth.sim.live.camera import Camera
from hearth.sim.live.lift import lift
from hearth.sim.live.room import generate_room
from hearth.sim.live.spec import CameraSpec, RoomSpec
from hearth.sim.pose import KEY_POSES
from hearth.sim.world import body

ROOM = generate_room(RoomSpec(width=6.5, depth=5.5), 1)
HEIGHT = 1.7


@pytest.fixture(params=["jetson", "corner_left", "front"])
def camera(request):
    return Camera(ROOM, CameraSpec(preset=request.param))


def seen(camera, pose):
    pts = []
    for name, p in pose.items():
        q = camera.project(p)
        if q is not None and 0 <= q[0] <= 1 and 0 <= q[1] <= 1:
            pts.append((name, q[0], q[1]))
    return pts


def mid_hip(joints):
    return ((joints["left_hip"][0] + joints["right_hip"][0]) / 2, (joints["left_hip"][2] + joints["right_hip"][2]) / 2)


def test_ray_inverts_project(camera):
    p = (0.4, 1.1, -3.0)
    u, v, depth = camera.project(p)
    d = camera.ray(u, v)
    o = camera.position
    back = (o[0] + d[0] * depth, o[1] + d[1] * depth, o[2] + d[2] * depth)
    assert back == pytest.approx(p, abs=1e-9)


@pytest.mark.parametrize("x,z,heading", [(0.0, -3.0, 0.0), (1.2, -2.5, 0.0), (-1.0, -4.0, math.pi / 2)])
def test_standing_body_lands_where_it_stood(camera, x, z, heading):
    truth = body(KEY_POSES["standing"], heading, x, z, 0.0, HEIGHT)
    got = lift(seen(camera, truth), camera, ROOM, height=HEIGHT, posture="standing")
    assert got is not None
    assert got.root == pytest.approx(mid_hip(truth), abs=0.12)
    assert got.joints["nose"][1] == pytest.approx(truth["nose"][1], abs=0.12)  # head at head height


def test_body_on_the_floor_lies_on_the_floor(camera):
    truth = body(KEY_POSES["side_lying"], 0.4, 0.5, -3.0, 0.0, HEIGHT)
    got = lift(seen(camera, truth), camera, ROOM, height=HEIGHT, posture="floor")
    assert got is not None
    assert all(p[1] < 0.25 for p in got.joints.values())
    # One camera can't tell how high off the floor each joint is (a side-lying hip is ~0.4 m up),
    # so from a low, grazing camera the body lands up to ~0.3 m off. Fine for an avatar.
    assert got.root == pytest.approx(mid_hip(truth), abs=0.3)


def test_heading_faces_the_camera_when_the_person_does():
    camera = Camera(ROOM, CameraSpec(preset="jetson"))
    truth = body(KEY_POSES["standing"], 0.0, 0.0, -3.0, 0.0, HEIGHT)
    got = lift(seen(camera, truth), camera, ROOM, height=HEIGHT)
    assert abs(math.remainder(got.heading, 2 * math.pi)) < 0.3


def test_bad_calibration_keeps_the_body_in_the_room():
    camera = Camera(ROOM, CameraSpec(preset="jetson", tilt=1.0))  # nearly level: feet rays reach far
    truth = body(KEY_POSES["standing"], 0.0, 0.0, -5.3, 0.0, HEIGHT)
    got = lift(seen(Camera(ROOM, CameraSpec(preset="jetson")), truth), camera, ROOM, height=HEIGHT)
    assert got is None or -ROOM.depth < got.root[1] < 0


def test_lifts_the_real_edge_wire_fixture():
    wire = json.loads((Path(__file__).parent / "fixtures/gaston_edge.json").read_text())
    camera = Camera(ROOM, CameraSpec(preset="jetson"))
    obs = next(o for o in wire["observations"] if o.get("keypoints"))
    got = lift([(k["name"], k["x"], k["y"]) for k in obs["keypoints"]], camera, ROOM)
    assert got is not None and -ROOM.depth < got.root[1] < 0


def test_nothing_usable_gives_none(camera):
    assert lift([], camera, ROOM) is None
    assert lift([("left_eye", 0.5, 0.01)], camera, ROOM) is None
