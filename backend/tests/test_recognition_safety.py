"""Names are incident context, with independent consent and body/hand association."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hearth_schema import HandGesturePrediction
from hearth.api.face_demo import create_face_router
from hearth.monitor.appointments import Appointments
from test_patient_camera import faces, obs


@pytest.mark.parametrize("name_opt_in,hand_inside,expected_name", [(True,True,"Alex"),(True,False,None),(False,True,None)])
def test_fist_alert_name_requires_consent_and_hand_on_matched_body(faces, service, clock, name_opt_in, hand_inside, expected_name):
    if not name_opt_in:
        faces.delete(faces.profiles()[0]["profile_id"])
        faces.enroll("Alex", ["a","b","c"], True, identify_alerts=False)
    appointments = Appointments(service, ":memory:")
    app = FastAPI()
    app.include_router(create_face_router(appointments, service, faces))
    headers={"X-Hearth-Face-Demo":"1"}
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1",50000)) as client:
        session=client.post("/face-demo/sessions",headers=headers,json={"room_id":"room-01"}).json()["session_id"]
        for i in range(8):
            clock.seconds=100+i*.5
            record=obs(clock.seconds)
            if i >= 3:
                record.hand_gesture_prediction=HandGesturePrediction(model_id=service.config.hand_gesture_model_id,
                    model_revision="demo", label="fist", confidence=.99, detection_count=1,
                    bbox=dict(x1=.3,y1=.3,x2=.4,y2=.4) if hand_inside else dict(x1=.92,y1=.3,x2=.99,y2=.4))
            response=client.post("/face-demo/observe", headers=headers, json={"session_id":session,
                "frame":str(i),"observation":record.model_dump(mode="json")})
            assert response.status_code==200,response.text
        alerts=[a for a in service.alerts.values() if a.kind=="help_requested"]
        assert len(alerts)==1
        assert alerts[0].metadata.get("resident_label")==expected_name
        assert alerts[0].metadata["method"]=="hand_gesture"
        assert "_hand_matches_body" not in alerts[0].metadata
        assert all("resident_label" not in event.metadata for event in service.events if event.event_type!="help_requested")
        assert all("resident_label" not in observation.model_dump() for observation in service.observations.values())
    appointments.db.close()


def test_appointment_links_visible_without_tokens_and_can_be_withdrawn(faces, service):
    from hearth.monitor.appointments import AppointmentInput
    from conftest import BASE
    store=Appointments(service, ":memory:")
    item=store.create(AppointmentInput(resident_label="Alex",room_id="room-01",scheduled_at=BASE,demo=True))
    profile=faces.profiles()[0]
    app=FastAPI()
    app.include_router(create_face_router(store,service,faces))
    with TestClient(app,base_url="http://localhost",client=("127.0.0.1",50000)) as client:
        url=f"/face-demo/profiles/{profile['profile_id']}/appointments/{item.appointment_id}"
        headers={"X-Hearth-Face-Demo":"1"}
        assert client.post(url,headers=headers).status_code==200
        result=client.get("/face-demo/profiles").json()[0]
        assert result["appointment_ids"]==[item.appointment_id]
        assert item.check_in_token not in str(result)
        assert client.delete(url,headers=headers).status_code==200
        assert client.get("/face-demo/profiles").json()[0]["appointment_ids"]==[]
        assert store.items[item.appointment_id].status=="scheduled"
        assert faces.profiles()[0]["identify_alerts"] is True
    store.db.close()
