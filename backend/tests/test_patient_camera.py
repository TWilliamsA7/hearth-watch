from datetime import timedelta
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("cryptography")
from cryptography.fernet import Fernet
from hearth.monitor.face_demo import FaceDemo, FaceNotVisible
from hearth.monitor.patient_camera import PatientCamera
from hearth.monitor.appointments import Appointments, AppointmentInput
from hearth.api.face_demo import create_face_router
from conftest import BASE, observation
from hearth_schema import NursePreferences


class Extractor:
    def extract(self, frame, **kwargs):
        if frame.startswith("hidden"):
            raise FaceNotVisible("No face visible")
        if frame.startswith("ambiguous"):
            raise ValueError("Multiple faces")
        return [0., 1.] + [0.] * 126 if frame.startswith("other") else [1.] + [0.] * 127


@pytest.fixture
def faces(tmp_path, clock):
    face = FaceDemo(tmp_path / "patients.sqlite", Fernet.generate_key(), Extractor(), clock=clock)
    face.enroll("Alex", ["a", "b", "c"], True, identify_alerts=True)
    yield face
    face.db.close()


def obs(t, label="standing", **changes):
    prediction = dict(model_id="melihuzunoglu/human-fall-detection", model_revision="demo",
                      label=label, confidence=.95, detection_count=1,
                      bbox=dict(x1=.1, y1=.1, x2=.9, y2=.9))
    prediction.update(changes)
    return observation(t, source_type="edge", scenario_id=None, posture_prediction=prediction)


def test_named_fall_after_face_turns_away_and_nurse_receives_it(faces, service, clock):
    nurse = service.save_nurse(NursePreferences(display_name="Nurse", room_ids=["room-01"], alert_kinds=["possible_fall"]))
    queue = service.subscribe(nurse_id=nurse.nurse_id)
    appointments = Appointments(service, ":memory:")
    item = appointments.create(AppointmentInput(resident_label="Alex", room_id="room-01", scheduled_at=BASE, demo=True))
    profile = faces.profiles()[0]
    faces.link_appointment(profile["profile_id"], item.appointment_id, item.check_in_token)
    later = appointments.create(AppointmentInput(resident_label="Alex", room_id="room-01", scheduled_at=BASE+timedelta(days=1), demo=True))
    faces.link_appointment(profile["profile_id"], later.appointment_id, later.check_in_token)
    app = FastAPI()
    app.include_router(create_face_router(appointments, service, faces))
    headers = {"X-Hearth-Face-Demo":"1"}
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1",50000)) as client:
        session = client.post("/face-demo/sessions", headers=headers, json={"room_id":"room-01"}).json()["session_id"]
        for index in range(10):
            clock.seconds = 100 + index * .5
            frame = str(index) if index < 3 else "hidden" + str(index)
            record = obs(clock.seconds, "standing" if index < 3 else "fallen")
            response = client.post("/face-demo/observe", headers=headers, json={"session_id":session,
                "frame":frame, "observation":record.model_dump(mode="json")})
            assert response.status_code == 200, response.text
            assert response.json()["accepted"] == 1
        assert appointments.items[item.appointment_id].status == "arrived"
        assert appointments.items[later.appointment_id].status == "scheduled"
        alert = next(a for a in service.alerts.values() if a.kind == "possible_fall")
        assert alert.metadata["resident_label"] == "Alex"
        assert alert.metadata["identity_method"] == "recent_face_body_track"
        assert alert.metadata["identity_demo"] is True
        messages = [queue.get_nowait() for _ in range(queue.qsize())]
        assert any(m["type"] == "alert" and m["data"]["metadata"].get("resident_label") == "Alex" for m in messages)
        faces.delete(profile["profile_id"])
        clock.seconds += .5
        assert client.post("/face-demo/observe", headers=headers, json={"session_id":session,
                "frame":"one", "observation":obs(clock.seconds).model_dump(mode="json")}).status_code == 422
    appointments.db.close()


@pytest.mark.parametrize("loss", ["gap", "jump", "ambiguous", "unknown", "multiple", "expired"])
def test_identity_cannot_survive_uncertain_tracking(faces, service, clock, loss):
    camera = PatientCamera(faces, service.config)
    session = faces.start("room-01")
    for index in range(3):
        clock.seconds = 100 + index * .5
        result = camera.process(session, obs(clock.seconds), str(index))
    assert result["profile"]["alias"] == "Alex"
    clock.seconds += 2 if loss == "gap" else .5
    changes = {}
    frame = "hidden-next"
    if loss == "jump": changes["bbox"] = dict(x1=0, y1=0, x2=.08, y2=.08)
    if loss == "ambiguous": frame = "ambiguous-next"
    if loss == "unknown": frame = "other-next"
    if loss == "multiple": changes.update(detection_count=2, label="unknown", confidence=None, bbox=None, uncertainty_reason="multiple_people")
    if loss == "expired": faces.sessions[session]["patient"]["until"] = clock.seconds
    result = camera.process(session, obs(clock.seconds, **changes), frame)
    assert "profile" not in result


def test_session_cannot_change_camera_or_use_replay(faces, service, clock):
    camera = PatientCamera(faces, service.config)
    session = faces.start("room-01")
    clock.seconds = 100
    camera.process(session, obs(100), "one")
    with pytest.raises(ValueError):
        camera.process(session, obs(100.5).model_copy(update={"source":"other"}), "two")
    with pytest.raises(ValueError):
        camera.process(session, obs(100.5).model_copy(update={"source_type":"replay"}), "two")


def test_fall_without_enrollment_still_alerts(service, clock):
    for index in range(7):
        clock.seconds = index * .5
        service.ingest([obs(clock.seconds, "fallen")])
    alert = next(a for a in service.alerts.values() if a.kind == "possible_fall")
    assert "resident_label" not in alert.metadata


def test_check_in_only_consent_does_not_label_incidents(faces, service, clock):
    record = faces.profiles()[0]
    faces.delete(record["profile_id"])
    faces.enroll("Check-in only", ["a","b","c"], True)
    session = faces.start("room-01")
    camera = PatientCamera(faces, service.config)
    for index in range(4):
        clock.seconds = 100 + index * .5
        assert "profile" not in camera.process(session, obs(clock.seconds), str(index))


def test_client_sends_same_observation_and_local_frame_only(monkeypatch):
    import base64
    cv = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from hearth.perception.patient_client import PatientClient
    with pytest.raises(ValueError):
        PatientClient("http://192.168.1.2:8002", "room-01")
    client = PatientClient("http://127.0.0.1:8002", "room-01")
    requests = []
    def request(path, body=None, method="POST"):
        requests.append((path, body, method))
        return {"session_id":"session"} if path == "/sessions" else {"accepted":1}
    monkeypatch.setattr(client, "request", request)
    record = obs(100)
    assert client.observe(np.zeros((480,640,3), dtype=np.uint8), record)["accepted"] == 1
    payload = requests[1][1]
    assert payload["observation"] == record.model_dump(mode="json")
    image = cv.imdecode(np.frombuffer(base64.b64decode(payload["frame"]),dtype=np.uint8),cv.IMREAD_COLOR)
    assert image.shape == (480,640,3)
    client.close()
    assert requests[-1] == ("/sessions/session", None, "DELETE")
