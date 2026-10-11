from datetime import timedelta
from threading import Event as Signal
from fastapi.testclient import TestClient
import pytest

from hearth.api.app import create_app
from hearth.monitor.appointments import AppointmentInput, Appointments, CheckIn
from hearth.monitor.config import MonitorConfig
from hearth.monitor.service import MonitorService
from hearth_schema import NursePreferences
from conftest import BASE, observation


def body(**changes):
    return AppointmentInput(resident_label="Demo guest", room_id="room-01", scheduled_at=BASE, **changes)


def test_fifteen_minute_boundary_and_nurse_websocket_delivery(service, clock):
    store = Appointments(service, ":memory:")
    nurse = service.save_nurse(NursePreferences(display_name="Demo nurse", room_ids=["room-01"],
                                              alert_kinds=["appointment_overdue"]))
    queue = service.subscribe(nurse_id=nurse.nurse_id)
    item = store.create(body())
    clock.seconds = 899
    store.check_due()
    assert not service.alerts
    clock.seconds = 900
    store.check_due()
    store.check_due()
    assert len(service.alerts) == 1
    assert store.items[item.appointment_id].status == "overdue"
    messages = [queue.get_nowait() for _ in range(queue.qsize())]
    assert len([m for m in messages if m["type"] == "alert"]) == 1


def test_reminder_and_resolution_survive_restart(service, clock, tmp_path):
    path = tmp_path / "appointments.sqlite"
    store = Appointments(service, path)
    item = store.create(body(grace_seconds=10, demo=True))
    clock.seconds = 11
    store.check_due()
    alert = next(iter(service.alerts.values()))
    service.transition_alert(alert.alert_id, "ack")
    service.transition_alert(alert.alert_id, "resolve")
    new_monitor = MonitorService(MonitorConfig(), clock=clock, utcnow=clock.utcnow)
    restarted = Appointments(new_monitor, path)
    restarted.check_due()
    assert len(new_monitor.alerts) == 1
    assert new_monitor.alerts[alert.alert_id].status == "resolved"
    assert restarted.items[item.appointment_id].status == "overdue"


def test_arrival_cancel_and_reschedule_prevent_obsolete_reminders(service, clock):
    store = Appointments(service, ":memory:")
    arrived = store.create(body())
    store.check_in(arrived.appointment_id, CheckIn(method="token", token=arrived.check_in_token))
    cancelled = store.create(body())
    store.update(cancelled.appointment_id)
    moved = store.create(body())
    store.update(moved.appointment_id, body().model_copy(update={"scheduled_at": BASE + timedelta(hours=1)}))
    clock.seconds = 901
    store.check_due()
    assert not service.alerts


def test_late_check_in_records_recovery_without_silently_resolving(service, clock):
    store = Appointments(service, ":memory:")
    item = store.create(body(grace_seconds=1))
    clock.seconds = 2
    store.check_due()
    store.check_in(item.appointment_id, CheckIn())
    alert = next(iter(service.alerts.values()))
    assert alert.recovered_at == clock.utcnow()
    assert alert.status == "open"


def test_rescheduling_overdue_retires_old_alert_and_rotates_token(service, clock):
    store = Appointments(service, ":memory:")
    item = store.create(body(grace_seconds=1, demo=True))
    clock.seconds = 2
    store.check_due()
    old = next(iter(service.alerts.values()))
    moved = store.update(item.appointment_id, body().model_copy(update={"scheduled_at": BASE + timedelta(hours=1)}))
    assert old.status == "resolved"
    assert moved.status == "scheduled" and moved.alert_id is None
    assert moved.check_in_token != item.check_in_token
    with pytest.raises(ValueError, match="token"):
        store.check_in(item.appointment_id, CheckIn(method="token", token=item.check_in_token))
    store.check_due()
    assert len(service.alerts) == 1


def test_face_presence_does_not_automatically_identify_or_check_in(service, clock):
    store = Appointments(service, ":memory:")
    item = store.create(body(demo=True))
    service.ingest([observation(0, source_type="edge", face_present=True)])
    assert store.items[item.appointment_id].status == "scheduled"
    with pytest.raises(ValueError, match="token"):
        store.check_in(item.appointment_id, CheckIn(method="face_assisted_demo", source="stub"))
    clock.seconds = 4
    with pytest.raises(ValueError, match="fresh"):
        store.check_in(item.appointment_id, CheckIn(method="face_assisted_demo", source="stub", token=item.check_in_token))
    service.ingest([observation(4, source_type="edge", face_present=True)])
    assert store.check_in(item.appointment_id, CheckIn(method="face_assisted_demo", source="stub",
                         token=item.check_in_token)).status == "arrived"


def test_http_demo_flow_timezone_validation_and_routing(service, clock):
    with TestClient(create_app(service=service)) as client:
        invalid = body().model_dump(mode="json")
        invalid["scheduled_at"] = "2026-10-10T12:00:00"
        assert client.post("/appointments", json=invalid).status_code == 422
        created = client.post("/appointments", json=body(grace_seconds=15, demo=True).model_dump(mode="json"))
        assert created.status_code == 201
        appointment_id = created.json()["appointment_id"]
        clock.seconds = 15
        assert client.get("/appointments").json()[0]["status"] == "overdue"
        assert client.get("/alerts").json()[0]["source_type"] == "simulation"
        assert client.post(f"/appointments/{appointment_id}/check-in", json={"method":"manual"}).status_code == 200
        assert client.post("/appointments/missing/check-in", json={}).status_code == 404


def test_background_timer_emits_without_polling_appointment_endpoint(service, clock):
    raised = Signal()
    service.config.health_poll_seconds = .01
    service.event_listeners.append(lambda event: raised.set() if event.event_type == "appointment_overdue" else None)
    with TestClient(create_app(service=service)) as client:
        response = client.post("/appointments", json=body(grace_seconds=15, demo=True).model_dump(mode="json"))
        assert response.status_code == 201
        clock.seconds = 15
        assert raised.wait(2), "Background scheduler did not raise the reminder"
