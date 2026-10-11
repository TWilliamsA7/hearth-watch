import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("cryptography")
from cryptography.fernet import Fernet, InvalidToken
from hearth.monitor.face_demo import FaceDemo
from hearth.api.face_demo import create_face_router
from hearth.monitor.appointments import Appointments, AppointmentInput
from conftest import BASE


class Extractor:
    def extract(self, frame, **kwargs):
        if frame == "missing":
            raise ValueError("Show exactly one face")
        index = 1 if frame.startswith("other") else 0
        return [float(i == index) for i in range(128)]


@pytest.fixture
def demo(tmp_path):
    now = [100.]
    store = FaceDemo(tmp_path / "faces.sqlite", Fernet.generate_key(), Extractor(), clock=lambda: now[0])
    store.test_time = now
    yield store
    store.db.close()


def enroll(demo, **kwargs):
    return demo.enroll("Volunteer Alpha", ["a", "b", "c"], True, **kwargs)


def test_consent_encryption_restart_and_withdrawal(demo):
    with pytest.raises(ValueError, match="consent"):
        demo.enroll("No", ["a", "b", "c"], False)
    profile = enroll(demo)
    assert "template" not in profile and "appointment_token" not in profile
    payload = demo.db.execute("SELECT payload FROM face_enrollments").fetchone()[0]
    assert b"Volunteer Alpha" not in payload
    assert b"template" not in payload
    assert b"Volunteer Alpha" in demo.cipher.decrypt(payload)
    with pytest.raises(InvalidToken):
        Fernet(Fernet.generate_key()).decrypt(payload)
    session = demo.start("room-01")
    demo.delete(profile["profile_id"])
    assert demo.profiles() == [] and session not in demo.sessions


def test_three_distinct_matches_and_unknown_reset(demo):
    enroll(demo)
    session = demo.start("room-01")
    def observe(frame):
        demo.test_time[0] += 1
        return demo.recognize(session, frame)
    assert observe("one")["matches"] == 1
    assert observe("one")["status"] == "new_frame_required"
    assert observe("two")["matches"] == 1
    assert observe("other-person")["status"] == "unknown"
    assert observe("three")["matches"] == 1
    assert observe("four")["matches"] == 2
    assert observe("five")["status"] == "recognized"
    demo.test_time[0] += demo.RETENTION_SECONDS
    assert demo.profiles() == []
    with pytest.raises(ValueError):
        demo.recognize(session, "six")


@pytest.mark.parametrize("reschedule", [False, True])
def test_only_current_linked_demo_appointment_checks_in(demo, service, reschedule):
    appointments = Appointments(service, ":memory:")
    body = AppointmentInput(resident_label="Fictional", room_id="room-01", scheduled_at=BASE, demo=True)
    item = appointments.create(body)
    app = FastAPI()
    app.include_router(create_face_router(appointments, service, demo))
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000)) as client:
        headers = {"X-Hearth-Face-Demo": "1"}
        assert client.post("/face-demo/sessions", json={"room_id":"room-01"}).status_code == 403
        assert client.get("/face-demo/profiles", headers={"Origin":"https://example.com"}).status_code == 403
        response = client.post("/face-demo/enroll", headers=headers, json={"alias":"Alpha", "consent":True,
            "frames":["a","b","c"], "appointment_id":item.appointment_id})
        assert response.status_code == 200, response.text
        if reschedule:
            appointments.update(item.appointment_id, body)
        session = client.post("/face-demo/sessions", headers=headers, json={"room_id":"room-01"}).json()["session_id"]
        for frame in ["one", "two", "three"]:
            demo.test_time[0] += 1
            response = client.post("/face-demo/recognize", headers=headers, json={"session_id":session,"frame":frame})
            assert response.status_code == 200, response.text
        assert response.json()["status"] == "recognized"
        current = appointments.items[item.appointment_id]
        assert current.status == ("scheduled" if reschedule else "arrived")
        if not reschedule:
            assert current.check_in_method == "face_recognition"
    appointments.db.close()


def test_face_loss_and_long_gap_reset_evidence(demo):
    enroll(demo)
    session = demo.start("room-01")
    assert demo.recognize(session, "one")["matches"] == 1
    demo.test_time[0] += 1
    with pytest.raises(ValueError):
        demo.recognize(session, "missing")
    demo.test_time[0] += 1
    assert demo.recognize(session, "two")["matches"] == 1
    demo.test_time[0] += 4
    assert demo.recognize(session, "three")["matches"] == 1


def test_regular_appointment_enrollment_and_remote_client_rejected(demo, service):
    appointments = Appointments(service, ":memory:")
    item = appointments.create(AppointmentInput(resident_label="Fictional", room_id="room-01", scheduled_at=BASE))
    app = FastAPI()
    app.include_router(create_face_router(appointments, service, demo))
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000)) as client:
        response = client.post("/face-demo/enroll", headers={"X-Hearth-Face-Demo":"1"}, json={
            "alias":"Alpha", "consent":True, "frames":["a","b","c"], "appointment_id":item.appointment_id})
        assert response.status_code == 200
        assert len(demo.profiles()) == 1
        session = client.post('/face-demo/sessions', headers={'X-Hearth-Face-Demo':'1'}, json={'room_id':'room-01'}).json()['session_id']
        for frame in ('one', 'two', 'three'):
            demo.test_time[0] += 1
            result = client.post('/face-demo/recognize', headers={'X-Hearth-Face-Demo':'1'}, json={'session_id':session,'frame':frame})
            assert result.status_code == 200
        assert appointments.items[item.appointment_id].status == 'arrived'
        assert appointments.items[item.appointment_id].check_in_method == 'face_recognition'
    with TestClient(app, base_url="http://localhost", client=("192.168.1.2", 50000)) as client:
        assert client.get("/face-demo/profiles").status_code == 403
    appointments.db.close()


def test_ambiguous_match_does_not_identify(demo):
    enroll(demo)
    demo.enroll("Volunteer Beta", ["other-a", "other-b", "other-c"], True)
    demo.extractor.extract = lambda frame: [2 ** -.5, 2 ** -.5] + [0.] * 126
    session = demo.start("room-01")
    assert demo.recognize(session, "ambiguous")["status"] == "unknown"
