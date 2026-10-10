"""Calibrating the simulated camera against the real one from a person standing in view."""

import random
import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from hearth.api.app import create_app
from hearth.monitor.config import MonitorConfig
from hearth.monitor.service import MonitorService
from hearth.sim.live.calibrate import C270_FOV, MIN_FRAMES, frame_from, solve
from hearth.sim.live.camera import Camera
from hearth.sim.live.room import generate_room
from hearth.sim.live.spec import CameraSpec, RoomSpec
from hearth.sim.pose import KEY_POSES
from hearth.sim.world import body

ROOM = generate_room(RoomSpec(width=7.0, depth=6.0), 1)
PERSON = 1.7


def seen_frames(cam_height, tilt, n=30, noise=0.004, seed=0):
    """Keypoints a real camera (height, tilt) would report for someone standing about the room."""
    rng = random.Random(seed)
    cam = Camera(ROOM, CameraSpec(preset="jetson", height=cam_height, tilt=tilt, fov=C270_FOV))
    out = []
    for _ in range(n):
        x, z = rng.uniform(-1.0, 1.0), rng.uniform(-4.2, -2.6)
        kps = []
        for name, p in body(KEY_POSES["standing"], rng.uniform(-0.5, 0.5), x, z, 0.0, PERSON).items():
            q = cam.project(p)
            if q and 0 <= q[0] <= 1 and 0 <= q[1] <= 1:
                kps.append((name, q[0] + rng.gauss(0, noise), q[1] + rng.gauss(0, noise)))
        out.append(kps)
    return out


@pytest.mark.parametrize("cam_height,tilt", [(1.6, 18.0), (2.2, 25.0), (1.3, 8.0)])
def test_recovers_the_cameras_height_and_tilt(cam_height, tilt):
    frames = [f for f in (frame_from(k) for k in seen_frames(cam_height, tilt)) if f]
    got = solve(frames, PERSON)
    assert got.ok, got.reason
    assert got.height == pytest.approx(cam_height, abs=0.06)
    assert got.tilt == pytest.approx(tilt, abs=1.2)


def test_too_few_frames_fails_with_a_reason():
    frames = [f for f in (frame_from(k) for k in seen_frames(1.6, 18.0, n=MIN_FRAMES - 1)) if f]
    got = solve(frames, PERSON)
    assert not got.ok and "frames" in got.reason


def test_standing_in_one_place_is_refused():
    """From one spot, height and tilt trade off: a confident wrong answer is worse than none."""
    frames = [f for f in (frame_from(k) for k in seen_frames(1.9, 29.0)) if f]
    one_spot = [type(f)(frames[0].low, f.low_frac, frames[0].high, f.high_frac) for f in frames]
    got = solve(one_spot, PERSON)
    assert not got.ok and "walk" in got.reason


def test_feet_are_not_needed():
    """The Jetson's camera cut off the feet in every real frame: knees or hips with the head do."""
    assert frame_from([("nose", 0.5, 0.2), ("left_hip", 0.5, 0.6), ("right_hip", 0.52, 0.6)]) is not None
    assert frame_from([("nose", 0.5, 0.2), ("left_shoulder", 0.45, 0.3), ("right_shoulder", 0.55, 0.3)]) is None


def test_recovers_the_camera_with_the_feet_out_of_view():
    frames = [f for f in (frame_from([k for k in kps if "ankle" not in k[0]])
                          for kps in seen_frames(1.75, 12.0)) if f]  # fmt: skip
    got = solve(frames, PERSON)
    assert got.ok, got.reason
    assert got.height == pytest.approx(1.75, abs=0.08) and got.tilt == pytest.approx(12.0, abs=1.5)


def test_calibrate_endpoint_applies_the_camera():
    """Through the real app: frames in at /ingest, POST /sim/live/link/calibrate, camera updated."""
    app = create_app(service=MonitorService(MonitorConfig.load()))
    sim = app.state.simulation
    with TestClient(app) as c:
        c.post("/sim/live/new", json={"spec": {"actors": 0, "camera": {"preset": "jetson"},
                                               "room": {"width": 7.0, "depth": 6.0}}, "seed": 1})
        c.post("/sim/live/link", json={"enabled": True, "source": "replay-01"})
        frames = seen_frames(1.6, 18.0, n=40, noise=0.002)

        # Feed frames while the calibration listens (it reads the newest one every 50 ms).
        import threading

        def feed():
            for i, kps in enumerate(frames):
                obs = {"obs_id": f"cal-{i}", "timestamp": datetime.now(timezone.utc).isoformat(),
                       "source": "replay-01", "source_type": "replay", "room_id": "room-01", "run_id": "cal",
                       "person_detected": True, "keypoints": [{"name": n, "x": min(max(u, 0), 1), "y": min(max(v, 0), 1)}
                                                              for n, u, v in kps]}  # fmt: skip
                c.post("/ingest/observations", json={"observations": [obs]})
                time.sleep(0.07)

        feeder = threading.Thread(target=feed)
        feeder.start()
        res = c.post("/sim/live/link/calibrate", json={"seconds": 3.2, "height": PERSON}).json()
        feeder.join()
        assert res["ok"], res
        assert res["height"] == pytest.approx(1.6, abs=0.08) and res["tilt"] == pytest.approx(18.0, abs=1.5)
        assert sim.live.world.camera.spec.tilt == res["tilt"]
        assert c.get("/sim/live/link").json()["camera"]["tilt"] == res["tilt"]
