"""Recognition integration boundary: poseNet geometry, consent, and scheduling isolation."""
import asyncio
import threading

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hearth.api.face_demo import create_face_router
from hearth.monitor.appointments import Appointments, AppointmentInput
from hearth.monitor.patient_camera import PatientCamera, TrackedBody
from conftest import observation, UPRIGHT, FLOOR, BASE
from test_patient_camera import faces, obs


HEADERS = {"X-Hearth-Face-Demo":"1"}


def pose_observation(t, floor=False):
    return observation(t, source_type="edge", scenario_id=None, features=FLOOR if floor else UPRIGHT,
                       keypoints=[{"name":"nose", "x":.5,"y":.2},
                                  {"name":"left_hip", "x":.4,"y":.6}])


def body_for(record, track="pose-1", **changes):
    values = dict(detector="posenet", obs_id=record.obs_id, track_id=track, detection_count=1,
                  bbox=dict(x1=.1,y1=.1,x2=.9,y2=.9))
    values.update(changes)
    return TrackedBody(**values)


def test_pose_net_features_can_produce_named_alert_without_yolo(faces, service, clock):
    app = FastAPI()
    appointments = Appointments(service, ":memory:")
    app.include_router(create_face_router(appointments, service, faces))
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1",50000)) as client:
        session = client.post("/face-demo/sessions", headers=HEADERS, json={"room_id":"room-01"}).json()["session_id"]
        for i in range(10):
            clock.seconds = 100 + i * .5
            record = pose_observation(clock.seconds, floor=i >= 3)
            response = client.post("/face-demo/observe", headers=HEADERS, json={
                "session_id":session, "frame":str(i) if i < 3 else "hidden-"+str(i),
                "observation":record.model_dump(mode="json"), "body":body_for(record).model_dump(mode="json")})
            assert response.status_code == 200, response.text
            assert response.json()["accepted"] == 1
        alert = next(a for a in service.alerts.values() if a.kind == "possible_fall")
        assert alert.metadata["resident_label"] == "Alex"
        assert alert.metadata["reason"] == "sustained_floor_level"
        assert all(o.posture_prediction is None for o in service.observations.values())
    appointments.db.close()


def test_pose_body_must_be_same_frame_and_new_tracks_clear_identity(faces, service, clock):
    camera = PatientCamera(faces, service.config)
    session = faces.start("room-01")
    for i in range(3):
        clock.seconds = 100 + i * .5
        record = pose_observation(clock.seconds)
        result = camera.process(session, record, str(i), body=body_for(record))
    assert result["profile"]["alias"] == "Alex"
    clock.seconds += .5
    record = pose_observation(clock.seconds)
    with pytest.raises(ValueError, match="exact observation"):
        camera.process(session, record, "wrong", body=body_for(record, obs_id="old-frame"))
    result = camera.process(session, record, "hidden-new", body=body_for(record, track="pose-2"))
    assert "profile" not in result
    # Delayed frames from a different track cannot move the evidence clock backwards.
    with pytest.raises(ValueError, match="timestamp order"):
        camera.process(session, record, "delayed", body=body_for(record, track="pose-3"))


@pytest.mark.parametrize("count", [0, 2])
def test_pose_ambiguity_clears_identity(faces, service, clock, count):
    camera = PatientCamera(faces, service.config)
    session = faces.start("room-01")
    for i in range(3):
        clock.seconds = 100 + i * .5
        record = pose_observation(clock.seconds)
        camera.process(session, record, str(i), body=body_for(record))
    clock.seconds += .5
    record = pose_observation(clock.seconds)
    assert "profile" not in camera.process(session, record, "hidden", body=body_for(record, detection_count=count, bbox=None))


def test_check_in_only_profile_does_not_crash_observe(faces, service, clock):
    faces.delete(faces.profiles()[0]["profile_id"])
    profile = faces.enroll("Check-in only", ["a","b","c"], True)
    app = FastAPI()
    appointments = Appointments(service, ":memory:")
    item = appointments.create(AppointmentInput(resident_label="Check-in only", room_id="room-01", scheduled_at=BASE, demo=True))
    faces.link_appointment(profile["profile_id"], item.appointment_id, item.check_in_token)
    app.include_router(create_face_router(appointments, service, faces))
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1",50000)) as client:
        session = client.post("/face-demo/sessions", headers=HEADERS, json={"room_id":"room-01"}).json()["session_id"]
        for i in range(3):
            clock.seconds = 100+i*.5
            response = client.post("/face-demo/observe", headers=HEADERS, json={"session_id":session,
                                   "frame":str(i), "observation":obs(clock.seconds).model_dump(mode="json")})
            assert response.status_code == 200, response.text
            assert response.json()["accepted"] == 1
        assert response.json()["identity"]["status"] == "identity_unconfirmed"
        assert "_check_in_profile" not in response.json()["identity"]
        assert "profile" not in response.json()["identity"]
        assert appointments.items[item.appointment_id].status == "arrived"
    appointments.db.close()


def test_slow_face_inference_does_not_block_health_and_busy_frames_are_rejected(faces, service, clock):
    started, release = threading.Event(), threading.Event()
    original = faces.extractor.extract
    def slow(frame, **kwargs):
        started.set()
        if not release.wait(5):
            raise ValueError("Test timeout")
        return original(frame, **kwargs)
    faces.extractor.extract = slow
    app = FastAPI()
    appointments = Appointments(service, ":memory:")
    app.include_router(create_face_router(appointments, service, faces))
    @app.get("/health")
    async def health():
        service.check_health()
        return {"ok":True}
    async def run():
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1",50000))
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
            session = (await client.post("/face-demo/sessions", headers=HEADERS, json={"room_id":"room-01"})).json()["session_id"]
            clock.seconds = 100
            payload = dict(session_id=session, frame="first", observation=obs(100).model_dump(mode="json"))
            task = asyncio.create_task(client.post("/face-demo/observe", headers=HEADERS, json=payload))
            try:
                assert await asyncio.to_thread(started.wait, 2)
                response = await asyncio.wait_for(client.get("/health"), timeout=.5)
                assert response.json()["ok"]
                response = await asyncio.wait_for(client.post("/face-demo/observe", headers=HEADERS, json=payload), timeout=.5)
                assert response.status_code == 503
            finally:
                release.set()
                result = await asyncio.wait_for(task, timeout=2)
            assert result.status_code == 200, result.text
    try:
        asyncio.run(run())
    finally:
        release.set()
        appointments.db.close()
