"""Current display names are independent of the simulation and scoped to camera tracks."""
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from conftest import observation
from hearth.api.identities import create_identity_router
from hearth.monitor.detector import make_event


@pytest.fixture
def display(service):
    app = FastAPI()
    app.include_router(create_identity_router(service))
    service.identity_of = lambda obs: {"recognized": True, "name": "Alex" if obs.person_id == "p1" else None}
    with TestClient(app) as client:
        yield client


def person(t, pid="p1", **changes):
    return observation(t, source="jetson-01", source_type="edge", scenario_id=None,
                       person_id=pid, **changes)


def test_names_are_available_without_a_simulation_or_admission(display, service):
    service.ingest([person(0), person(0, "p2")])
    result = display.get("/rooms/room-01/identities").json()
    assert [(p["person_id"], p["recognized"], p["name"]) for p in result] == [
        ("p1", True, "Alex"), ("p2", True, None)]
    assert result[0]["source"] == "jetson-01" and result[0]["run_id"] == "run-1"
    assert display.get("/rooms/unknown/identities").status_code == 404


def test_absent_and_stale_tracks_lose_current_names(display, service, clock):
    service.ingest([person(0)])
    clock.seconds = 0.1
    service.ingest([person(.1, person_detected=False, features=None)])
    assert display.get("/rooms/room-01/identities").json() == []
    service.ingest([person(.2)])
    clock.seconds = 4
    assert display.get("/rooms/room-01/identities").json() == []


def test_missing_person_does_not_keep_name_while_camera_tracks_others(display, service, clock):
    service.ingest([person(0)])
    clock.seconds = 2
    service.ingest([person(2, "p2")])
    assert [p["person_id"] for p in display.get("/rooms/room-01/identities").json()] == ["p2"]


def test_unrecognized_or_disabled_recognition_returns_no_names(display, service):
    service.ingest([person(0)])
    service.identity_of = lambda obs: {"recognized": False, "name": "old name"}
    assert display.get("/rooms/room-01/identities").json() == []
    service.identity_of = None
    assert display.get("/rooms/room-01/identities").json() == []


def test_event_names_are_captured_at_the_observation_not_reassigned_later(service, clock):
    obs = person(0)
    service.ingest([obs])
    service.incident_labeler = lambda _: {"resident_label": "Alex", "identity_demo": True}
    service.record(make_event(obs, "recovered"))
    assert service.events[-1].metadata["resident_label"] == "Alex"
    clock.seconds = 0.2
    service.ingest([person(.2)])
    service.incident_labeler = lambda _: {"resident_label": "Bo", "identity_demo": True}
    service.record(make_event(obs, "face_present"))
    assert "resident_label" not in service.events[-1].metadata
    assert service.events[-2].metadata["resident_label"] == "Alex"
