"""The live camera link, end to end through the real app: camera observations in at /ingest,
the real monitor decides the posture, the person appears in the live room, residents respond."""

import time
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from hearth.api.app import create_app
from hearth.monitor.config import MonitorConfig
from hearth.monitor.service import MonitorService
from hearth.sim.nav import distance
from hearth.sim.pose import KEY_POSES
from hearth.sim.world import body

SPEC = {"actors": 2, "noise": False, "camera": {"preset": "jetson"},
        "room": {"width": 6.5, "depth": 5.5, "furniture_density": 0.0}}
SOURCE = "replay-01"  # what edge/tools/replay.py sends; jetson-01 works the same way
T0 = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)


def camera_frames(world, pose, x, z, start, seconds, hz=10):
    """Observations shaped like the edge's (no keypoint confidence) for a body at (x, z)."""
    truth = body(KEY_POSES[pose], 0.0, x, z, 0.0, 1.7)
    kps = []
    for name, p in truth.items():
        q = world.camera.project(p)
        if q is not None and 0 <= q[0] <= 1 and 0 <= q[1] <= 1:
            kps.append({"name": name, "x": round(q[0], 4), "y": round(q[1], 4)})
    xs, ys = [k["x"] for k in kps], [k["y"] for k in kps]
    aspect = (max(xs) - min(xs)) * 4 / 3 / max(1e-6, max(ys) - min(ys))
    lying = pose != "standing"
    feats = ({"torso_angle_deg": 85.0, "head_y": 0.6, "hip_y": 0.6, "bbox_aspect": max(aspect, 1.5)} if lying
             else {"torso_angle_deg": 3.0, "head_y": 0.2, "hip_y": 0.5, "bbox_aspect": 0.4})
    return [{"obs_id": f"{pose}-{start + i / hz:.1f}", "timestamp": (T0 + timedelta(seconds=start + i / hz)).isoformat(),
             "source": SOURCE, "source_type": "replay", "room_id": "room-01", "run_id": "r1",
             "person_detected": True, "keypoints": kps, "features": feats, "face_present": None}
            for i in range(int(seconds * hz))]


def tick(sim, seconds=0.8):
    """The live controller calls update() every 50 ms; posture changes take ~0.4 s to settle."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        sim.link.update()
        time.sleep(0.05)


def test_a_live_person_plays_out_in_the_live_room():
    app = create_app(service=MonitorService(MonitorConfig.load()))
    sim = app.state.simulation
    with TestClient(app) as c:
        assert c.post("/sim/live/new", json={"spec": SPEC, "seed": 1}).status_code == 200
        assert c.post("/sim/live/link", json={"enabled": True, "source": SOURCE}).json()["state"] == "waiting"
        world = sim.live.world

        c.post("/ingest/observations", json={"observations": camera_frames(world, "standing", 0.4, -3.0, 0, 2)})
        tick(sim)
        status = c.get("/sim/live/link").json()
        assert status["state"] == "live" and status["posture"] == "standing"
        person = next(p for p in world.frame().people if p.kind == "live")
        assert person.person_id == f"live-{SOURCE}"
        live = world.externals[person.person_id]
        assert distance(live.pos, (0.4, -3.0)) < 0.25

        # On the floor long enough for the real monitor's 3 s hold: a possible fall, and help comes.
        c.post("/ingest/observations", json={"observations": camera_frames(world, "side_lying", 0.4, -3.0, 2, 4.5)})
        tick(sim)
        status = c.get("/sim/live/link").json()
        assert status["posture"] == "floor" and status["alert_open"]
        assert any(a["kind"] == "possible_fall" and a["source"] == SOURCE for a in c.get("/alerts").json())
        assert sim.live.world.incidents.responders.get(f"live-{SOURCE}")

        assert c.post("/sim/live/link", json={"enabled": False, "source": SOURCE}).json()["state"] == "off"
        assert all(p.kind != "live" for p in world.frame().people)


def test_link_needs_a_room():
    with TestClient(create_app(service=MonitorService(MonitorConfig.load()))) as c:
        assert c.post("/sim/live/link", json={"enabled": True}).status_code == 409
