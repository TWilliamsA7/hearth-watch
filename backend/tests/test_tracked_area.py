"""The tracked area: a corner camera above it, calibrated in two steps, people placed inside it."""

import math
import random

import pytest
from hearth.sim.live.avatar import LiveAvatar
from hearth.sim.live.calibrate import C270_FOV, frame_from, solve, solve_pan
from hearth.sim.live.camera import Camera
from hearth.sim.live.room import generate_room
from hearth.sim.live.spec import CameraSpec, RoomSpec
from hearth.sim.pose import KEY_POSES
from hearth.sim.world import body

AREA = generate_room(RoomSpec(width=5.0, depth=4.0, furniture_density=0.0), 1)  # the taped-out area
H = 1.7


def corner_camera(height=2.4, tilt=32.0, pan=0.0, corner="left"):
    return Camera(AREA, CameraSpec(preset="jetson_corner", height=height, tilt=tilt, pan=pan, corner=corner,
                                   fov=C270_FOV))  # fmt: skip


def keypoints(cam, x, z, *, noise=0.003, rng=None, feet=True, heading=0.0):
    rng = rng or random.Random(0)
    out = []
    for name, p in body(KEY_POSES["standing"], heading, x, z, 0.0, H).items():
        q = cam.project(p)
        if q is None or not (0 <= q[0] <= 1 and 0 <= q[1] <= 1) or (not feet and "ankle" in name):
            continue
        out.append((name, q[0] + rng.gauss(0, noise), q[1] + rng.gauss(0, noise)))
    return out


@pytest.mark.parametrize("corner", ["left", "right"])
def test_corner_camera_looks_at_the_area_centre(corner):
    cam = corner_camera(corner=corner)
    side = -1 if corner == "left" else 1
    assert cam.position[0] == pytest.approx(side * 2.45) and cam.position[1] == pytest.approx(2.4)
    centre = cam.project((0.0, 0.0, -2.0))
    assert centre is not None and abs(centre[0] - 0.5) < 0.01  # straight ahead across the floor
    u, v, depth = cam.project((0.7, 1.1, -2.8))
    d, o = cam.ray(u, v), cam.position
    assert (o[0] + d[0] * depth, o[1] + d[1] * depth, o[2] + d[2] * depth) == pytest.approx((0.7, 1.1, -2.8), abs=1e-9)


def test_pan_turns_the_view_right():
    straight, turned = corner_camera(), corner_camera(pan=10.0)
    p = (0.0, 0.0, -2.0)
    assert turned.project(p)[0] < straight.project(p)[0]  # turned right: the centre slides left in the image


def test_two_step_calibration_recovers_a_corner_camera_with_the_feet_cut_off():
    real = corner_camera(height=2.3, tilt=30.0, pan=7.0)
    rng = random.Random(3)
    walk = [frame_from(keypoints(real, rng.uniform(-1.8, 1.8), rng.uniform(-3.6, -0.8), rng=rng, feet=False))
            for _ in range(60)]  # fmt: skip
    walk = [f for f in walk if f]
    spec = CameraSpec(preset="jetson_corner", corner="left")
    got = solve(walk, H, spec=spec, width=AREA.width, depth=AREA.depth)
    assert got.ok, got.reason
    assert got.height == pytest.approx(2.3, abs=0.06) and got.tilt == pytest.approx(30.0, abs=1.5)

    stood = [frame_from(keypoints(real, 0.0, -2.0, rng=rng, feet=False)) for _ in range(20)]
    calibrated = spec.model_copy(update={"height": got.height, "tilt": got.tilt})
    pan = solve_pan([f for f in stood if f], H, calibrated, AREA.width, AREA.depth)
    assert pan.ok, pan.reason
    assert pan.pan == pytest.approx(7.0, abs=2.0)
    assert pan.off_by < 0.25


def test_apparent_size_places_someone_when_hips_and_feet_are_out_of_view():
    cam = corner_camera()
    for x, z in [(0.0, -2.0), (1.0, -3.2), (-1.2, -1.2)]:
        a = LiveAvatar(H)
        upper = [k for k in keypoints(cam, x, z, noise=0.0) if not k[0].endswith(("hip", "ankle"))]
        t = 0.0
        for _ in range(15):
            a.update(upper, "standing", cam, AREA, t)
            t += 0.1
        assert math.dist(a.root, (x, z)) < 0.15, (x, z, a.root)


def test_leaving_the_area_is_noticed():
    cam = corner_camera()
    a = LiveAvatar(H)
    t = 0.0
    for _ in range(10):
        a.update(keypoints(cam, 0.0, -2.0), "standing", cam, AREA, t)
        t += 0.1
    assert not a.outside
    for step in range(100):  # walk out, then allow the speed-limited avatar to reach the boundary
        a.update(keypoints(cam, min(3.5, 0.2 * step), -2.0), "standing", cam, AREA, t)
        t += 0.1
    assert a.outside
