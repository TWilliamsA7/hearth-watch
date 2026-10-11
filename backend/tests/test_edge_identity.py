"""Jetson face templates name one tracked person's incidents, never anyone else's."""
from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("cryptography")
from hearth.api.face_demo import create_face_router
from hearth.monitor.appointments import Appointments, AppointmentInput
from conftest import BASE, FLOOR, UPRIGHT, observation
from test_patient_camera import faces  # noqa: F401 (fixture: "Alex" enrolled with named alerts)

TOKEN = "edge-secret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
ALEX = [1.] + [0.] * 127
OTHER = [0., 1.] + [0.] * 126


def person(t, person_id, floor=False, **changes):
    data = dict(features=FLOOR if floor else UPRIGHT, obs_id=f"{person_id}-{t:g}", source="jetson-01",
                source_type="edge", scenario_id=None, person_id=person_id)
    data.update(changes)
    return observation(t, **data)


def template(t, person_id, embedding=ALEX, **changes):
    data = dict(detector="res10+sface", source="jetson-01", room_id="room-01", run_id="run-1",
                person_id=person_id, obs_id=f"{person_id}-{t:g}",
                timestamp=(BASE + timedelta(seconds=t)).isoformat(),
                face_box=dict(x1=.4, y1=.1, x2=.5, y2=.2), embedding=embedding)
    data.update(changes)
    return data


@pytest.fixture
def jetson(faces, service):
    app = FastAPI()
    appointments = Appointments(service, ":memory:")
    app.include_router(create_face_router(appointments, service, faces, edge_token=TOKEN))
    # The Jetson reaches the laptop over the USB network, not loopback.
    with TestClient(app, client=("192.168.55.1", 40000)) as client:
        client.appointments = appointments
        yield client
    appointments.db.close()


def step(client, service, clock, t, people, faces_seen=()):
    """One frame: each (person_id, floor) observed, then face templates for faces_seen."""
    clock.seconds = t
    assert service.ingest([person(t, p, floor) for p, floor in people])["accepted"] == len(people)
    return [client.post("/face-demo/edge-embeddings", headers=AUTH, json=template(t, p, e)).json()
            for p, e in faces_seen]


def run_fall(client, service, clock, faces_for):
    """p1 and p2 stand, then both lie down; faces_for(t) gives the face templates sent at t."""
    results = []
    for i in range(14):
        t = i * .5
        results.append(step(client, service, clock, t, [("p1", t >= 1.5), ("p2", t >= 1.5)], faces_for(t)))
    return results


def test_named_only_on_the_recognized_persons_fall(jetson, service, clock):
    results = run_fall(jetson, service, clock, lambda t: [("p1", ALEX), ("p2", OTHER)] if t < 1.5 else [])
    assert [r[0]["status"] for r in results[:3]] == ["confirming", "confirming", "recognized"]
    assert results[0][1]["status"] == "unknown"
    assert "profile" not in results[2][0]  # the edge never learns who it saw
    falls = {a.metadata["person_id"]: a for a in service.alerts.values() if a.kind == "possible_fall"}
    assert falls["p1"].metadata["resident_label"] == "Alex"
    assert falls["p1"].metadata["identity_method"] == "edge_face_track"
    assert "resident_label" not in falls["p2"].metadata


def test_two_matches_are_not_enough(jetson, service, clock):
    run_fall(jetson, service, clock, lambda t: [("p1", ALEX)] if t < 1 else [])
    fall = next(a for a in service.alerts.values() if a.metadata["person_id"] == "p1")
    assert "resident_label" not in fall.metadata


def test_name_ends_with_the_track(jetson, service, clock):
    for i in range(3):
        step(jetson, service, clock, i * .5, [("p1", False)], [("p1", ALEX)])
    clock.seconds = 1.5
    service.ingest([person(1.5, "p1", person_detected=False, features=None)])
    for i in range(4, 14):
        step(jetson, service, clock, i * .5, [("p1", True)])
    fall = next(a for a in service.alerts.values() if a.kind == "possible_fall")
    assert "resident_label" not in fall.metadata


def test_name_expires_without_a_fresh_match(jetson, service, clock):
    for i in range(3):
        step(jetson, service, clock, i * .5, [("p1", False)], [("p1", ALEX)])
    for i in range(3, 24):  # stands for 10 s more with the face never checked, then lies down
        step(jetson, service, clock, i * .5, [("p1", i >= 20)])
    for i in range(24, 32):
        step(jetson, service, clock, i * .5, [("p1", True)])
    fall = next(a for a in service.alerts.values() if a.kind == "possible_fall")
    assert "resident_label" not in fall.metadata


def test_one_face_on_two_tracks_names_neither(jetson, service, clock):
    results = run_fall(jetson, service, clock, lambda t: [("p1", ALEX), ("p2", ALEX)] if t < 1.5 else [])
    assert results[2][0]["status"] == "recognized" and results[2][1]["status"] == "ambiguous_track"
    assert all("resident_label" not in a.metadata for a in service.alerts.values())


def test_unknown_face_clears_the_name(jetson, service, clock):
    run_fall(jetson, service, clock, lambda t: [("p1", ALEX)] if t < 1.5 else [("p1", OTHER)] if t == 1.5 else [])
    assert all("resident_label" not in a.metadata for a in service.alerts.values())


def test_withdrawal_clears_the_name(jetson, faces, service, clock):
    for i in range(3):
        step(jetson, service, clock, i * .5, [("p1", False)], [("p1", ALEX)])
    faces.delete(faces.profiles()[0]["profile_id"])
    for i in range(3, 14):
        step(jetson, service, clock, i * .5, [("p1", True)])
    assert all("resident_label" not in a.metadata for a in service.alerts.values())


def test_without_named_alert_consent_checks_in_but_never_names(jetson, faces, service, clock):
    faces.delete(faces.profiles()[0]["profile_id"])
    item = jetson.appointments.create(AppointmentInput(resident_label="Sam", room_id="room-01",
                                                       scheduled_at=BASE, demo=True))
    faces.enroll("Sam", ["a", "b", "c"], True, item.appointment_id, item.check_in_token)
    results = run_fall(jetson, service, clock, lambda t: [("p1", ALEX)] if t < 1.5 else [])
    assert results[2][0] == {"status": "recognized", "matches": 3, "checked_in": item.appointment_id}
    assert jetson.appointments.items[item.appointment_id].status == "arrived"
    assert all("resident_label" not in a.metadata for a in service.alerts.values())


def test_template_needs_a_live_track(jetson, service, clock):
    clock.seconds = 0
    response = jetson.post("/face-demo/edge-embeddings", headers=AUTH, json=template(0, "p9"))
    assert response.json() == {"status": "track_unknown"}


@pytest.mark.parametrize("change", [
    dict(embedding=[.5] * 128),                  # not normalized
    dict(embedding=[1.] + [0.] * 126),           # wrong length
    dict(detector="yolo"),
    dict(frame="base64..."),                     # never pixels
])
def test_invalid_templates_are_rejected(jetson, service, clock, change):
    step(jetson, service, clock, 0, [("p1", False)])
    assert jetson.post("/face-demo/edge-embeddings", headers=AUTH, json=template(0, "p1", **change)).status_code == 422


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"},
                                     {**AUTH, "Origin": "http://evil.example"}])
def test_edge_route_needs_the_token_and_no_browser(jetson, service, clock, headers):
    step(jetson, service, clock, 0, [("p1", False)])
    assert jetson.post("/face-demo/edge-embeddings", headers=headers, json=template(0, "p1")).status_code == 403


def test_edge_route_is_off_without_a_configured_token(faces, service, clock):
    app = FastAPI()
    appointments = Appointments(service, ":memory:")
    app.include_router(create_face_router(appointments, service, faces, edge_token=""))
    with TestClient(app, client=("192.168.55.1", 40000)) as client:
        response = client.post("/face-demo/edge-embeddings", headers={"Authorization": "Bearer "},
                               json=template(0, "p1"))
        assert response.status_code == 403
        # The token does not open the loopback-only routes.
        assert client.get("/face-demo/profiles", headers=AUTH).status_code == 403
    appointments.db.close()


def test_ordinary_ingest_cannot_assert_a_name(service, clock):
    for i in range(14):
        clock.seconds = i * .5
        service.ingest([person(i * .5, "p1", i >= 3)])
    assert all("resident_label" not in a.metadata for a in service.alerts.values())
