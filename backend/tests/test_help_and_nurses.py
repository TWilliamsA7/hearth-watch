import pytest
from fastapi.testclient import TestClient

from hearth_schema import HelpRequest, NursePreferences
from hearth.api.app import create_app
from hearth.monitor.config import MonitorConfig
from hearth.monitor.service import MonitorService
from conftest import UPRIGHT, observation


def points(raised=True, confidence=.9, side="left"):
    return [dict(name="nose", x=.5, y=.3, confidence=confidence),
            dict(name=side+"_shoulder", x=.4, y=.45, confidence=confidence),
            dict(name=side+"_wrist", x=.4, y=.1 if raised else .65, confidence=confidence)]


def frame(t, raised=True, **changes):
    return observation(t, UPRIGHT, keypoints=points(raised), **changes)


def nurse(service, name="Sam", rooms=None, kinds=None, duty=True):
    return service.save_nurse(NursePreferences(display_name=name, room_ids=rooms or ["room-01"],
                                              alert_kinds=kinds or ["help_requested"], on_duty=duty))


def messages(queue):
    result=[]
    while not queue.empty():
        result.append(queue.get_nowait())
    return result


def help_alerts(service):
    return [a for a in service.alerts.values() if a.kind == "help_requested"]


def test_hold_triggers_once_and_requires_confirmed_release(service):
    service.ingest([frame(t/2) for t in range(4)])
    assert not help_alerts(service)
    service.ingest([frame(t/2) for t in range(4, 15)])
    assert len(help_alerts(service)) == 1
    alert=help_alerts(service)[0]
    assert alert.metadata["method"] == "raised_arm" and alert.metadata["duration_seconds"] == 2
    service.ingest([observation(8, None, person_detected=False)])
    service.ingest([frame(t/2) for t in range(20, 31)])
    assert len(help_alerts(service)) == 1  # Occlusion did not re-arm the signal.
    service.ingest([frame(t/2, False) for t in range(31, 35)])
    service.ingest([frame(t/2) for t in range(35, 40)])
    assert len(help_alerts(service)) == 2


def test_brief_wave_and_missing_keypoints_do_not_request_help(service):
    service.ingest([frame(0), frame(.5), frame(1, False)])
    service.ingest([observation(t/2, UPRIGHT) for t in range(3, 12)])
    assert not help_alerts(service)


def test_gesture_hold_resets_across_gap_or_occlusion(service):
    service.ingest([frame(0), frame(.5), frame(1)])
    service.ingest([observation(1.5, None, person_detected=False), frame(2), frame(2.5), frame(3)])
    service.ingest([frame(10), frame(10.5), frame(11)])
    assert not help_alerts(service)


def test_low_confidence_and_switching_arms_cannot_complete_hold(service):
    service.ingest([observation(t/2, UPRIGHT, keypoints=points(confidence=.1)) for t in range(8)])
    service.ingest([observation(4+t/2, UPRIGHT, keypoints=points(side="left" if t%2 else "right")) for t in range(8)])
    assert not help_alerts(service)


def test_cooldown_prevents_repeated_requests_after_short_release(service):
    service.ingest([frame(t/2) for t in range(5)])
    service.ingest([frame(t/2, False) for t in range(5, 8)])
    service.ingest([frame(t/2) for t in range(8, 23)])
    assert len(help_alerts(service)) == 1
    service.ingest([frame(11.5), frame(12)])
    assert len(help_alerts(service)) == 2


def test_manual_requests_are_idempotent_and_do_not_mark_camera_online(service):
    one=service.request_help(HelpRequest(request_id="one", room_id="room-01"))
    retry=service.request_help(HelpRequest(request_id="one", room_id="room-01"))
    double_click=service.request_help(HelpRequest(request_id="two", room_id="room-01"))
    assert one.alert_id == retry.alert_id == double_click.alert_id
    assert one.source_type == "manual" and not service.health
    assert len(service.events) == 1


def test_subscriptions_route_by_room_kind_and_duty(clock):
    service=MonitorService(MonitorConfig(posture_mode="pose_fallback", additional_room_ids=["room-02"]),clock=clock,utcnow=clock.utcnow)
    matched=nurse(service)
    other_room=nurse(service, "Other room", rooms=["room-02"])
    other_kind=nurse(service, "Equipment", kinds=["source_offline"])
    off=nurse(service, "Off duty", duty=False)
    queues=[service.subscribe(nurse_id=n.nurse_id) for n in (matched,other_room,other_kind,off)]
    room_queue=service.subscribe(room_id="room-02")
    service.request_help(HelpRequest(request_id="one",room_id="room-01"))
    assert [m["type"] for m in messages(queues[0])] == ["event","alert"]
    assert all(not messages(q) for q in queues[1:])
    assert not messages(room_queue)
    service.ingest([observation(0)])
    for q in queues: messages(q)
    clock.seconds=4
    service.check_health()
    assert any(m["type"]=="alert" and m["data"]["kind"]=="source_offline" for m in messages(queues[2]))
    assert not any(m["type"]=="alert" for m in messages(queues[0]))


def test_two_nurses_cannot_claim_same_alert_and_both_see_acceptance(service):
    sam,alex=nurse(service),nurse(service,"Alex")
    queues=[service.subscribe(nurse_id=n.nurse_id) for n in (sam,alex)]
    alert=service.request_help(HelpRequest(request_id="one",room_id="room-01"))
    for q in queues: messages(q)
    service.accept_alert(alert.alert_id,sam.nurse_id)
    for q in queues:
        assert any(m["type"]=="alert" and m["data"]["accepted_by_name"]=="Sam" for m in messages(q))
    with pytest.raises(ValueError,match="Already accepted"):
        service.accept_alert(alert.alert_id,alex.nurse_id)
    with pytest.raises(ValueError,match="Only the nurse"):
        service.transition_alert(alert.alert_id,"resolve",alex.nurse_id)
    with pytest.raises(ValueError):
        service.transition_alert(alert.alert_id,"resolve")
    assert service.transition_alert(alert.alert_id,"resolve",sam.nurse_id).status=="resolved"


def test_off_duty_cannot_accept_but_owner_can_finish_previous_work(service):
    sam=nurse(service,duty=False)
    alert=service.request_help(HelpRequest(request_id="one",room_id="room-01"))
    with pytest.raises(ValueError):service.accept_alert(alert.alert_id,sam.nurse_id)
    prefs=NursePreferences(display_name="Sam",room_ids=["room-01"],alert_kinds=["help_requested"],on_duty=True)
    service.save_nurse(prefs,sam.nurse_id)
    service.accept_alert(alert.alert_id,sam.nurse_id)
    prefs.on_duty=False
    service.save_nurse(prefs,sam.nurse_id)
    assert service.nurse_snapshot(sam.nurse_id).alerts[0].alert_id==alert.alert_id
    assert service.transition_alert(alert.alert_id,"resolve",sam.nurse_id).status=="resolved"


def test_room_snapshot_isolated(clock):
    service=MonitorService(MonitorConfig(posture_mode="pose_fallback", additional_room_ids=["room-02"]),clock=clock,utcnow=clock.utcnow)
    service.request_help(HelpRequest(request_id="one",room_id="room-02"))
    assert not service.snapshot("room-01").alerts
    assert len(service.snapshot("room-02").alerts)==1
    with pytest.raises(ValueError):service.request_help(HelpRequest(request_id="one",room_id="room-01"))


def test_nurse_api_websocket(service):
    with TestClient(create_app(service=service)) as client:
        profile=client.post('/nurse-sessions',json=dict(display_name="Sam",room_ids=["room-01"],alert_kinds=["help_requested"],on_duty=True)).json()
        nurse_id=profile['nurse_id']
        with client.websocket_connect('/ws/nurse/'+nurse_id) as ws:
            assert ws.receive_json()['type']=='nurse_snapshot'
            alert=client.post('/help-requests',json=dict(request_id='one',room_id='room-01')).json()
            assert ws.receive_json()['data']['event_type']=='help_requested'
            assert ws.receive_json()['data']['alert_id']==alert['alert_id']
            accepted=client.post(f"/alerts/{alert['alert_id']}/accept",json=dict(nurse_id=nurse_id))
            assert accepted.json()['accepted_by_name']=='Sam'
            assert ws.receive_json()['data']['event_type']=='alert_acknowledged'
            assert ws.receive_json()['data']['status']=='acknowledged'
        assert client.post(f"/alerts/{alert['alert_id']}/resolve",json=dict(nurse_id=nurse_id)).json()['status']=='resolved'
        assert client.get('/nurse-sessions/'+nurse_id+'/snapshot').json()['alerts'][0]['status']=='resolved'
        assert client.get('/nurse-sessions/missing/snapshot').status_code==404


def test_subscription_edit_sends_replacement_snapshot(service):
    profile=nurse(service)
    queue=service.subscribe(nurse_id=profile.nurse_id)
    service.request_help(HelpRequest(request_id="one",room_id="room-01"))
    messages(queue)
    service.save_nurse(NursePreferences(display_name="Sam",room_ids=["room-01"],alert_kinds=["source_offline"]),profile.nurse_id)
    update=messages(queue)
    assert len(update)==1 and update[0]['type']=='nurse_snapshot'
    assert update[0]['data']['alerts']==[]
