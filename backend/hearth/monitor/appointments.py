"""Persistent appointment reminders; face presence never identifies a resident."""
from datetime import timedelta
from pathlib import Path
import secrets
import sqlite3
from typing import Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from hearth_schema import Alert, AlertMessage, Event


class AppointmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    resident_label: str = Field(min_length=1, max_length=80)
    room_id: str
    scheduled_at: AwareDatetime
    grace_seconds: int = Field(default=900, ge=1, le=86400)
    demo: bool = False


class Appointment(AppointmentInput):
    appointment_id: str
    check_in_token: str
    status: Literal["scheduled", "overdue", "arrived", "cancelled"] = "scheduled"
    checked_in_at: AwareDatetime | None = None
    check_in_method: str | None = None
    alert_id: str | None = None


class CheckIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["manual", "token", "face_assisted_demo"] = "manual"
    token: str | None = None
    source: str | None = None


class Appointments:
    def __init__(self, monitor, path):
        self.monitor = monitor
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS appointments (id TEXT PRIMARY KEY, data TEXT NOT NULL, alert TEXT)")
        self.items = {}
        for appointment_json, alert_json in self.db.execute("SELECT data, alert FROM appointments"):
            item = Appointment.model_validate_json(appointment_json)
            self.items[item.appointment_id] = item
            if alert_json:
                alert = Alert.model_validate_json(alert_json)
                # Nurse sessions are not durable; release any previous assignment.
                alert.accepted_by = alert.accepted_by_name = None
                monitor.alerts[alert.alert_id] = alert
                monitor.event_alerts[alert.event_id] = alert.alert_id
        monitor.event_listeners.append(self._save_alert_transition)

    def _save(self, item, alert=None):
        alert = alert or self.monitor.alerts.get(item.alert_id)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO appointments VALUES (?, ?, ?)",
                            (item.appointment_id, item.model_dump_json(), alert.model_dump_json() if alert else None))
        self.items[item.appointment_id] = item

    def _save_alert_transition(self, event):
        alert = self.monitor.alerts.get(event.metadata.get("alert_id"))
        if alert and alert.kind == "appointment_overdue":
            self._save(self.items[alert.metadata["appointment_id"]], alert)

    def create(self, body):
        self.monitor.validate_room(body.room_id)
        item = Appointment(**body.model_dump(), appointment_id=str(uuid4()), check_in_token=secrets.token_urlsafe(24))
        self._save(item)
        return item

    def check_due(self):
        now = self.monitor.utcnow()
        for item in tuple(self.items.values()):
            if item.status != "scheduled" or now < item.scheduled_at + timedelta(seconds=item.grace_seconds):
                continue
            metadata = dict(appointment_id=item.appointment_id, resident_label=item.resident_label,
                            scheduled_at=item.scheduled_at.isoformat(), grace_seconds=item.grace_seconds,
                            reason="arrival_not_confirmed", clock="backend_utc", demo=item.demo)
            event = Event(event_id=str(uuid4()), timestamp=now, source="appointments",
                          source_type="simulation" if item.demo else "manual", room_id=item.room_id,
                          event_type="appointment_overdue", metadata=metadata)
            alert = Alert(alert_id=str(uuid4()), event_id=event.event_id, kind=event.event_type,
                          source=event.source, source_type=event.source_type, room_id=item.room_id,
                          opened_at=now, metadata=metadata)
            updated = item.model_copy(update={"status": "overdue", "alert_id": alert.alert_id})
            # Commit reminder and appointment together before publishing.
            self._save(updated, alert)
            self.monitor.alerts[alert.alert_id] = alert
            self.monitor.event_alerts[event.event_id] = alert.alert_id
            self.monitor.record(event)
            self.monitor.publish(AlertMessage(data=alert))

    def check_in(self, appointment_id, body, *, verified_demo_face=False):
        item = self.items[appointment_id]
        if verified_demo_face and body.method != "token":
            raise ValueError("Verified recognition requires a token-bound appointment")
        if item.status == "cancelled":
            raise ValueError("Appointment is cancelled")
        if body.method != "manual":
            if not body.token or not secrets.compare_digest(body.token, item.check_in_token):
                raise ValueError("Valid appointment check-in token required")
        if body.method == "face_assisted_demo":
            if not item.demo:
                raise ValueError("Face-assisted check-in is only available for demo appointments")
            now = self.monitor.clock()
            observations = [o for o in self.monitor.observations.values()
                          if o.room_id == item.room_id and o.source == body.source
                          and o.source_type == "edge"]
            latest = max((o.timestamp for o in observations), default=None)
            candidates = [o for o in observations if o.timestamp == latest
                          and o.person_detected and o.face_present is True
                          and -self.monitor.config.edge_future_tolerance_seconds <=
                          (self.monitor.utcnow() - o.timestamp).total_seconds() <
                          self.monitor.config.source_timeout_seconds]
            from .detector import source_key
            if not any((entry := self.monitor.health.get(source_key(o))) and entry.status.online
                       and not entry.status.observation_stale and entry.last_observation_receipt is not None
                       and now - entry.last_observation_receipt < self.monitor.config.source_timeout_seconds
                       for o in candidates):
                raise ValueError("A fresh live face-presence observation is required; this does not verify identity")
        if item.status == "arrived":
            return item
        method = "face_recognition" if verified_demo_face else body.method
        updated = item.model_copy(update={"status": "arrived", "checked_in_at": self.monitor.utcnow(),
                                          "check_in_method": method})
        self._save(updated)
        if item.alert_id:
            alert = self.monitor.alerts[item.alert_id]
            alert.recovered_at = updated.checked_in_at
            alert.metadata["check_in_method"] = method
            self._save(updated, alert)
            self.monitor.publish(AlertMessage(data=alert))
        return updated

    def update(self, appointment_id, body=None):
        item = self.items[appointment_id]
        if item.status in {"arrived", "cancelled"}:
            raise ValueError("Appointment is already closed")
        if body:
            self.monitor.validate_room(body.room_id)
            updated = Appointment(**body.model_dump(), appointment_id=item.appointment_id,
                                  check_in_token=secrets.token_urlsafe(24))
        else:
            updated = item.model_copy(update={"status": "cancelled"})
        if item.alert_id:
            alert = self.monitor.alerts[item.alert_id]
            alert.status, alert.resolved_at = "resolved", self.monitor.utcnow()
            alert.metadata["resolution"] = "rescheduled" if body else "cancelled"
            # Persist the old reminder even when the schedule is replaced.
            self._save(updated, alert)
            self.monitor.publish(AlertMessage(data=alert))
        else:
            self._save(updated)
        return updated
