"""Enroll on the laptop, walk in front of the Jetson: you show up in the live room by name.

End to end through the real app: enrollment, the Jetson's observations and face templates at
the edge routes, the live camera link placing the person in the simulated room, and the name
the link reports (only with the person's consent to be named)."""
import functools
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("cryptography")
from cryptography.fernet import Fernet

import hearth.api.app as app_module
from hearth.api.face_demo import create_face_router
from hearth.monitor.config import MonitorConfig
from hearth.monitor.face_demo import FaceDemo
from hearth.monitor.service import MonitorService
from hearth.sim.pose import KEY_POSES
from hearth.sim.world import body
from test_live_link import SPEC, tick
from test_patient_camera import Extractor

TOKEN = "edge-secret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
SOURCE = "jetson-01"
ALEX, BO = [1.] + [0.] * 127, [0., 1.] + [0.] * 126  # what Extractor gives their enrollment frames


def jetson_frames(world, person_id, x, z, end, count=3, hz=10):
    """What run_edge.py sends for one standing person: live timestamps, no run_id, a track id."""
    truth = body(KEY_POSES["standing"], 0.0, x, z, 0.0, 1.7)
    kps = [{"name": n, "x": round(q[0], 4), "y": round(q[1], 4)} for n, p in truth.items()
           if (q := world.camera.project(p)) is not None and 0 <= q[0] <= 1 and 0 <= q[1] <= 1]
    return [{"obs_id": f"{person_id}-{end.timestamp() - i / hz:.2f}", "timestamp": (end - timedelta(seconds=i / hz)).isoformat(),
             "source": SOURCE, "source_type": "edge", "room_id": "room-01", "person_id": person_id,
             "person_detected": True, "keypoints": kps, "face_present": True,
             "features": {"torso_angle_deg": 3.0, "head_y": 0.2, "hip_y": 0.5, "bbox_aspect": 0.4}}
            for i in reversed(range(count))]


def template(person_id, embedding, at):
    return {"detector": "res10+sface", "source": SOURCE, "room_id": "room-01", "person_id": person_id,
            "obs_id": f"{person_id}-{at.timestamp():.2f}", "timestamp": at.isoformat(),
            "face_box": {"x1": .4, "y1": .1, "x2": .5, "y2": .2}, "embedding": embedding}


@pytest.fixture
def hearth(tmp_path, monkeypatch):
    faces = FaceDemo(tmp_path / "faces.sqlite", Fernet.generate_key(), Extractor(), clock=time.time)
    faces.enroll("Alex", ["a", "b", "c"], True, identify_alerts=True)   # agreed to be named
    faces.enroll("Bo", ["other-a", "other-b", "other-c"], True)         # recognized, not named
    monkeypatch.setattr(app_module, "create_face_router",
                        functools.partial(create_face_router, demo=faces, edge_token=TOKEN))
    app = app_module.create_app(service=MonitorService(MonitorConfig.load()))
    with TestClient(app, client=("192.168.55.1", 40000)) as client:  # the Jetson, over USB
        yield client, app.state.simulation
    faces.db.close()


def test_a_recognized_live_participant_is_named_in_the_live_room(hearth):
    client, sim = hearth
    client.post("/sim/live/new", json={"spec": SPEC, "seed": 1})
    client.post("/sim/live/link", json={"enabled": True, "source": SOURCE})
    world = sim.live.world
    results = []
    for _ in range(4):  # three matching face checks confirm a name
        now = datetime.now(timezone.utc)
        batch = sorted(jetson_frames(world, "p1", -1.2, -3.0, now) + jetson_frames(world, "p2", 1.4, -3.5, now),
                       key=lambda o: o["timestamp"])  # in time order, as the Jetson sends them
        counts = client.post("/ingest/observations", json={"observations": batch}).json()
        assert counts["accepted"] == len(batch), counts
        results.append([client.post("/face-demo/edge-embeddings", headers=AUTH, json=template(p, e, now)).json()["status"]
                        for p, e in (("p1", ALEX), ("p2", BO))])
        tick(sim, 0.3)
    assert results[2] == ["recognized", "recognized"]

    people = {p["track_id"]: p for p in client.get("/sim/live/link").json()["people"]}
    assert people["p1"]["state"] == "live"
    assert (people["p1"]["recognized"], people["p1"]["name"]) == (True, "Alex")
    assert (people["p2"]["recognized"], people["p2"]["name"]) == (True, None)  # no consent to be named
    assert {p.person_id for p in world.frame().people if p.kind == "live"} == {f"live-{SOURCE}-p1", f"live-{SOURCE}-p2"}


def test_unenrolled_people_stay_anonymous(hearth):
    client, sim = hearth
    client.post("/sim/live/new", json={"spec": SPEC, "seed": 1})
    client.post("/sim/live/link", json={"enabled": True, "source": SOURCE})
    stranger = [0., 0., 1.] + [0.] * 125
    for _ in range(4):
        now = datetime.now(timezone.utc)
        client.post("/ingest/observations", json={"observations": jetson_frames(sim.live.world, "p1", -1.2, -3.0, now)})
        assert client.post("/face-demo/edge-embeddings", headers=AUTH, json=template("p1", stranger, now)).json()["status"] == "unknown"
        tick(sim, 0.3)
    person = client.get("/sim/live/link").json()["people"][0]
    assert (person["recognized"], person["name"]) == (False, None)
