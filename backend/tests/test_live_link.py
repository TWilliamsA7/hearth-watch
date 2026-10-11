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


def people_frames(world, people, start, seconds, hz=10):
    """One camera reporting several people: per frame, one observation per (person_id, pose, x, z),
    with the edge's posture-model call. People in one frame share its timestamp."""
    out = []
    for person_id, pose, x, z in people:
        frames = camera_frames(world, pose, x, z, start, seconds, hz)
        label = "standing" if pose == "standing" else "fallen"
        for f in frames:
            xs, ys = [k["x"] for k in f["keypoints"]], [k["y"] for k in f["keypoints"]]
            f.update(obs_id=f"{person_id}-{f['obs_id']}", person_id=person_id,
                     posture_prediction={"model_id": "hearth/posture-mlp", "model_revision": "test",
                                         "input_processing": "original", "label": label, "confidence": 0.93,
                                         "bbox": {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)},
                                         "detection_count": 1, "uncertainty_reason": None})
        out += frames
    return sorted(out, key=lambda o: o["timestamp"])


def test_each_person_from_one_camera_gets_their_own_avatar_and_alert():
    app = create_app(service=MonitorService(MonitorConfig.load()))
    sim = app.state.simulation
    with TestClient(app) as c:
        c.post("/sim/live/new", json={"spec": SPEC, "seed": 1})
        c.post("/sim/live/link", json={"enabled": True, "source": SOURCE})
        world = sim.live.world

        both_up = [("p1", "standing", -1.2, -3.0), ("p2", "standing", 1.4, -3.5)]
        c.post("/ingest/observations", json={"observations": people_frames(world, both_up, 0, 2)})
        tick(sim)
        status = c.get("/sim/live/link").json()
        assert status["state"] == "live"
        assert [(p["person_id"], p["track_id"], p["posture"]) for p in status["people"]] == [
            (f"live-{SOURCE}-p1", "p1", "standing"), (f"live-{SOURCE}-p2", "p2", "standing")]
        assert status["people"][0]["model"] == {"model_id": "hearth/posture-mlp", "label": "standing",
                                                "confidence": 0.93}
        live = {p.person_id: p for p in world.frame().people if p.kind == "live"}
        assert set(live) == {f"live-{SOURCE}-p1", f"live-{SOURCE}-p2"}
        assert live[f"live-{SOURCE}-p2"].model is not None and live[f"live-{SOURCE}-p2"].model.label == "standing"
        assert distance(world.externals[f"live-{SOURCE}-p1"].pos, (-1.2, -3.0)) < 0.3
        assert distance(world.externals[f"live-{SOURCE}-p2"].pos, (1.4, -3.5)) < 0.3

        # p2 goes down for longer than the monitor's hold: only p2's alert, and help goes to p2.
        p2_down = [("p1", "standing", -1.2, -3.0), ("p2", "side_lying", 1.4, -3.5)]
        c.post("/ingest/observations", json={"observations": people_frames(world, p2_down, 2, 4.5)})
        tick(sim)
        status = c.get("/sim/live/link").json()
        by_id = {p["track_id"]: p for p in status["people"]}
        assert by_id["p2"]["posture"] == "floor" and by_id["p2"]["alert_open"]
        assert by_id["p2"]["model"]["label"] == "fallen"
        assert by_id["p1"]["posture"] == "standing" and not by_id["p1"]["alert_open"]
        assert status["alert_open"]
        alerts = [a for a in c.get("/alerts").json() if a["kind"] == "possible_fall" and a["source"] == SOURCE]
        assert [a["metadata"]["person_id"] for a in alerts] == ["p2"]
        assert set(world.incidents.responders) == {f"live-{SOURCE}-p2"}

        assert c.post("/sim/live/link", json={"enabled": False, "source": SOURCE}).json()["people"] == []
        assert all(p.kind != "live" for p in world.frame().people)


def test_someone_who_leaves_is_taken_out_and_the_others_stay():
    now = [1000.0]
    app = create_app(service=MonitorService(MonitorConfig.load(), clock=lambda: now[0]))
    sim = app.state.simulation
    with TestClient(app) as c:
        c.post("/sim/live/new", json={"spec": SPEC, "seed": 1})
        c.post("/sim/live/link", json={"enabled": True, "source": SOURCE})
        world = sim.live.world
        c.post("/ingest/observations", json={"observations": people_frames(
            world, [("p1", "standing", -1.2, -3.0), ("p2", "standing", 1.4, -3.5)], 0, 1)})
        for _ in range(12):
            now[0] += 0.05
            sim.link.update()
        assert {p.person_id for p in world.frame().people if p.kind == "live"} == {
            f"live-{SOURCE}-p1", f"live-{SOURCE}-p2"}
        # p1 walks out; p2 is still seen 5 s later. (Also the "nobody" record is never a person.)
        now[0] += 5.0
        nobody = {**people_frames(world, [("p2", "standing", 1.4, -3.5)], 5, 0.1)[0],
                  "obs_id": "nobody", "person_detected": False, "keypoints": None, "features": None,
                  "posture_prediction": None}
        nobody.pop("person_id")
        c.post("/ingest/observations", json={"observations": [nobody] + people_frames(
            world, [("p2", "standing", 1.4, -3.5)], 5.1, 1)})
        for _ in range(12):
            now[0] += 0.05
            sim.link.update()
        assert {p.person_id for p in world.frame().people if p.kind == "live"} == {f"live-{SOURCE}-p2"}
        assert [p["track_id"] for p in c.get("/sim/live/link").json()["people"]] == ["p2"]
