"""Shared monitor entry point for edge, simulator and replay observations."""
import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic
from typing import Callable
from uuid import uuid4

from hearth_schema import (Alert, AlertMessage, Event, EventMessage, Heartbeat, HelpRequest,
                           NursePreferences, NurseProfile, NurseSnapshot, NurseSnapshotMessage,
                           Observation, ObservationMessage, RoomSnapshot,
                           SourceStatus, SourceStatusMessage)

from .config import MonitorConfig
from .detector import FallDetector, SourceKey, TrackKey, make_event, source_key, track_key
from .gestures import HelpGestureDetector


# A source reporting several people shows the most urgent posture among them.
POSTURE_URGENCY = {"fallen": 3, "sitting": 2, "standing": 1}


@dataclass
class HealthEntry:
    status: SourceStatus
    last_receipt: float
    last_observation_receipt: float | None = None
    # person_id -> (posture, confidence, receipt time): each person's latest, for status.posture
    people: dict[str | None, tuple[str, float | None, float]] = field(default_factory=dict)


class MonitorService:
    """In-memory state. Use one server worker. All methods run on its event loop."""

    def __init__(self, config: MonitorConfig, *, clock: Callable[[], float] = monotonic,
                 utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.config, self.clock, self.utcnow = config, clock, utcnow
        self.detector = FallDetector(config)
        self.gestures = HelpGestureDetector(config)
        self.events: deque[Event] = deque(maxlen=config.event_capacity)
        self.event_listeners: list[Callable[[Event], None]] = []
        # API control may explicitly pause a simulated observation clock.
        # Live producers always retain the usual stale-evidence behavior.
        self.observation_clock_paused: Callable[[SourceKey], bool] = lambda key: False
        self.alerts: dict[str, Alert] = {}
        self.event_alerts: dict[str, str] = {}
        self.health: dict[SourceKey, HealthEntry] = {}
        self.observations: dict[TrackKey, Observation] = {}  # latest per person per source
        self.seen: OrderedDict[tuple, None] = OrderedDict()
        self.subscribers: set[asyncio.Queue] = set()
        self.subscription_rooms: dict[asyncio.Queue, str] = {}
        self.subscription_nurses: dict[asyncio.Queue, str] = {}
        self.nurses: dict[str, NurseProfile] = {}
        self.manual_requests: OrderedDict[str, str] = OrderedDict()
        self.offline_alerts: dict[SourceKey, str] = {}
        self.ingest_totals = dict(accepted=0, duplicates=0, out_of_order=0, stale=0, future=0)

    def matches_nurse(self, nurse: NurseProfile, alert: Alert) -> bool:
        return (alert.accepted_by == nurse.nurse_id or
                (nurse.on_duty and alert.room_id in nurse.room_ids and alert.kind in nurse.alert_kinds))

    def should_deliver(self, queue: asyncio.Queue, payload: dict) -> bool:
        data = payload["data"]
        if nurse_id := self.subscription_nurses.get(queue):
            nurse = self.nurses[nurse_id]
            if payload["type"] == "nurse_snapshot":
                return data["nurse"]["nurse_id"] == nurse_id
            if payload["type"] == "alert":
                return self.matches_nurse(nurse, self.alerts[data["alert_id"]])
            if payload["type"] == "event":
                alert_id = data["metadata"].get("alert_id") or self.event_alerts.get(data["event_id"])
                return bool(alert_id in self.alerts and self.matches_nurse(nurse, self.alerts[alert_id]))
            if payload["type"] == "source_status":
                return nurse.on_duty and data["room_id"] in nurse.room_ids
            return False  # No raw observation stream in a nurse subscription.
        if payload["type"] == "nurse_snapshot":
            return False
        room = self.subscription_rooms.get(queue)
        return room is None or data.get("room_id") == room

    def unsubscribe(self, queue: asyncio.Queue):
        self.subscribers.discard(queue)
        self.subscription_rooms.pop(queue, None)
        self.subscription_nurses.pop(queue, None)

    def publish(self, message) -> None:
        payload = message.model_dump(mode="json")
        for queue in tuple(self.subscribers):
            if not self.should_deliver(queue, payload):
                continue
            if queue.full():
                # A slow dashboard must reconnect and obtain a fresh snapshot.
                self.unsubscribe(queue)
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(None)
            else:
                queue.put_nowait(payload)

    def subscribe(self, *, room_id: str | None = None, nurse_id: str | None = None) -> asyncio.Queue:
        if room_id is not None:
            self.validate_room(room_id)
        if nurse_id is not None and nurse_id not in self.nurses:
            raise KeyError(nurse_id)
        queue = asyncio.Queue(maxsize=self.config.websocket_queue_capacity)
        self.subscribers.add(queue)
        if room_id is not None:
            self.subscription_rooms[queue] = room_id
        if nurse_id is not None:
            self.subscription_nurses[queue] = nurse_id
        return queue

    def record(self, event: Event) -> None:
        self.events.append(event)
        for listener in tuple(self.event_listeners):
            listener(event)
        self.publish(EventMessage(data=event))

    def validate_room(self, room_id: str) -> None:
        if room_id not in self.config.room_ids:
            raise ValueError(f"Room {room_id!r} is not configured")

    def open_alert(self, event: Event) -> Alert:
        alert = Alert(alert_id=str(uuid4()), event_id=event.event_id, kind=event.event_type,
                      source=event.source, source_type=event.source_type, room_id=event.room_id,
                      scenario_id=event.scenario_id, run_id=event.run_id,
                      opened_at=event.timestamp, severity=event.metadata.get("severity", "warning"),
                      metadata=event.metadata.copy())
        if event.person_id is not None:  # a source reporting several people: whose alert it is
            alert.metadata["person_id"] = event.person_id
        self.alerts[alert.alert_id] = alert
        self.event_alerts[event.event_id] = alert.alert_id
        self.record(event)
        self.publish(AlertMessage(data=alert))
        return alert

    def touch(self, record: Observation | Heartbeat, *, observation: bool) -> None:
        key, now, received_at = source_key(record), self.clock(), self.utcnow()
        entry = self.health.get(key)
        changed = entry is None or not entry.status.online
        if entry is None:
            entry = HealthEntry(SourceStatus(source=record.source, source_type=record.source_type,
                                room_id=record.room_id, scenario_id=record.scenario_id,
                                run_id=record.run_id, online=True), now)
            self.health[key] = entry
        was_stale = entry.status.observation_stale
        previous_posture = (entry.status.posture, entry.status.posture_model)
        entry.last_receipt, entry.status.last_seen = now, received_at
        entry.status.online = True
        if observation:
            entry.last_observation_receipt = now
            entry.status.last_observation_at = received_at
            entry.status.observation_stale = False
            prediction = record.posture_prediction
            reliable = (record.person_detected and prediction is not None
                        and prediction.model_id == self.config.fall_model_id
                        and prediction.detection_count == 1 and prediction.confidence is not None
                        and prediction.confidence >= self.config.model_confidence_min)
            posture = prediction.label if reliable else "unknown"
            confidence = prediction.confidence if reliable else None
            entry.status.posture_model = prediction.model_id if prediction else None
            if prediction is None and self.config.posture_mode == "pose_fallback":
                state = self.detector.states.get(track_key(record))
                posture = {"upright": "standing", "floor_level": "fallen",
                           "possible_fall": "fallen"}.get(state.posture if state else "unknown", "unknown")
            entry.people[record.person_id] = (posture, confidence, now)
            recent = [p for p in entry.people.values() if now - p[2] < self.config.source_timeout_seconds]
            top = max(recent, key=lambda p: (POSTURE_URGENCY.get(p[0], 0), p[2]))
            entry.status.posture, entry.status.posture_confidence = top[0], top[1]
        if isinstance(record, Heartbeat):
            for name in ("fps", "inference_ms", "mem_used_mb"):
                setattr(entry.status, name, getattr(record, name))
        if changed:
            metadata = {"clock": "backend_receipt"}
            if alert_id := self.offline_alerts.pop(key, None):
                alert = self.alerts[alert_id]
                alert.recovered_at = received_at
                metadata["alert_id"] = alert_id
                self.publish(AlertMessage(data=alert))
            self.record(make_event(record, "source_online", timestamp=received_at, **metadata))
        if (changed or was_stale != entry.status.observation_stale or not observation
                or previous_posture != (entry.status.posture, entry.status.posture_model)):
            self.publish(SourceStatusMessage(data=entry.status))

    def check_health(self) -> None:
        now = self.clock()
        for key, entry in self.health.items():
            changed = False
            if (not entry.status.observation_stale and entry.last_observation_receipt is not None
                    and now - entry.last_observation_receipt >= self.config.source_timeout_seconds):
                entry.status.observation_stale = True
                entry.status.posture, entry.status.posture_confidence = "unknown", None
                if not self.observation_clock_paused(key):
                    self.invalidate_source(key)
                changed = True
            if entry.status.online and now - entry.last_receipt >= self.config.source_timeout_seconds:
                entry.status.online = False
                entry.status.observation_stale = True
                entry.status.posture, entry.status.posture_confidence = "unknown", None
                self.invalidate_source(key)
                event = make_event(entry.status, "source_offline", timestamp=self.utcnow(),
                                   clock="backend_receipt", reason="source_silent")
                self.offline_alerts[key] = self.open_alert(event).alert_id
                changed = True
            if changed:
                self.publish(SourceStatusMessage(data=entry.status))

    def invalidate_source(self, key, *, tracked_only=False):
        for detector in (self.detector, self.gestures):
            for tracked in tuple(detector.states):
                # Track keys end in person_id; None is the untracked (single-person) track.
                if tracked[:5] == key and (not tracked_only or tracked[5] is not None):
                    detector.invalidate(tracked)

    def ingest(self, batch: list[Observation], *, patient_labels=None) -> dict[str, int]:
        # Validate the complete batch before applying anything.
        for obs in batch:
            self.validate_room(obs.room_id)
        self.check_health()
        counts = dict.fromkeys(self.ingest_totals, 0)
        received_at = self.utcnow()
        for obs in batch:
            # Ordering and dedupe are per person: people in one frame share a timestamp.
            key = track_key(obs)
            identity = (*key, obs.obs_id)
            if identity in self.seen:
                counts["duplicates"] += 1
                continue
            # Buffered live frames must not refresh camera health, recover an incident,
            # or advance a temporal hold. Replay/simulation use their own clocks.
            if obs.source_type == "edge":
                age = (received_at - obs.timestamp).total_seconds()
                if age > self.config.edge_observation_max_age_seconds:
                    counts["stale"] += 1
                    continue
                if age < -self.config.edge_future_tolerance_seconds:
                    counts["future"] += 1
                    continue
            previous = self.observations.get(key)
            if previous is not None and obs.timestamp <= previous.timestamp:
                counts["out_of_order"] += 1
                continue
            self.seen[identity] = None
            if len(self.seen) > self.config.dedupe_capacity:
                self.seen.popitem(last=False)
            events = self.detector.process(obs) + self.gestures.process(obs)
            if not obs.person_detected and obs.person_id is None:
                # Clear disappeared resident tracks. The untracked gesture
                # detector may still have a valid hand even without a body.
                self.invalidate_source(source_key(obs), tracked_only=True)
            self.touch(obs, observation=True)
            self.observations[key] = obs
            self.publish(ObservationMessage(data=obs))
            for event in events:
                if event.event_type in {"possible_fall", "help_requested"}:
                    # Only the local recognition route supplies these labels, bound
                    # to this exact observation. Ordinary ingest cannot assert a name.
                    if patient_labels and obs.obs_id in patient_labels:
                        label = patient_labels[obs.obs_id].copy()
                        hand_matches_body = label.pop("_hand_matches_body", False)
                        if (event.event_type != "help_requested" or event.metadata.get("method") != "hand_gesture"
                                or hand_matches_body):
                            event.metadata.update(label)
                    self.open_alert(event)
                    continue
                if event.event_type == "recovered":
                    alert_id = self.event_alerts.get(event.metadata["incident_event_id"])
                    if alert_id:
                        event.metadata["alert_id"] = alert_id
                        alert = self.alerts[alert_id]
                        alert.recovered_at = event.timestamp
                        self.publish(AlertMessage(data=alert))
                self.record(event)
            counts["accepted"] += 1
        for name, count in counts.items():
            self.ingest_totals[name] += count
        return counts

    def heartbeat(self, heartbeat: Heartbeat) -> None:
        self.validate_room(heartbeat.room_id)
        self.check_health()
        self.touch(heartbeat, observation=False)

    def transition_alert(self, alert_id: str, action: str, nurse_id: str | None = None) -> Alert:
        alert = self.alerts[alert_id]
        if action == "ack":
            if alert.status in {"acknowledged", "resolved"}:
                return alert
            alert.status, alert.acknowledged_at = "acknowledged", self.utcnow()
            kind = "alert_acknowledged"
        elif action == "resolve":
            if alert.accepted_by and alert.accepted_by != nurse_id:
                raise ValueError("Only the nurse who accepted this alert can resolve it")
            if alert.status == "resolved":
                return alert
            if alert.status != "acknowledged":
                raise ValueError("Acknowledge the alert before resolving it")
            alert.status, alert.resolved_at = "resolved", self.utcnow()
            kind = "alert_resolved"
        else:
            raise ValueError("Unknown alert action")
        self.record(make_event(alert, kind, timestamp=self.utcnow(), alert_id=alert_id,
                               incident_event_id=alert.event_id, nurse_id=nurse_id, clock="backend_receipt"))
        self.publish(AlertMessage(data=alert))
        return alert

    def save_nurse(self, preferences: NursePreferences, nurse_id: str | None = None) -> NurseProfile:
        for room_id in preferences.room_ids:
            self.validate_room(room_id)
        name = preferences.display_name.strip()
        if not name:
            raise ValueError("Enter a nurse display name")
        if nurse_id is not None and nurse_id not in self.nurses:
            raise KeyError(nurse_id)
        nurse_id = nurse_id or str(uuid4())
        nurse = NurseProfile(**preferences.model_dump(), nurse_id=nurse_id)
        nurse.display_name = name
        nurse.room_ids = list(dict.fromkeys(nurse.room_ids))
        nurse.alert_kinds = list(dict.fromkeys(nurse.alert_kinds))
        self.nurses[nurse_id] = nurse
        self.publish(NurseSnapshotMessage(data=self.nurse_snapshot(nurse_id)))
        return nurse

    def accept_alert(self, alert_id: str, nurse_id: str) -> Alert:
        alert, nurse = self.alerts[alert_id], self.nurses[nurse_id]
        if alert.accepted_by == nurse_id:
            return alert
        if alert.accepted_by:
            raise ValueError(f"Already accepted by {alert.accepted_by_name}")
        if alert.status == "resolved":
            raise ValueError("This alert is already resolved")
        if not self.matches_nurse(nurse, alert):
            raise ValueError("Go on duty and subscribe to this room and situation before accepting")
        alert.accepted_by, alert.accepted_by_name = nurse_id, nurse.display_name
        alert.status, alert.acknowledged_at = "acknowledged", self.utcnow()
        self.record(make_event(alert, "alert_acknowledged", timestamp=alert.acknowledged_at,
                               alert_id=alert_id, nurse_id=nurse_id, nurse_name=nurse.display_name,
                               clock="backend_receipt"))
        self.publish(AlertMessage(data=alert))
        return alert

    def request_help(self, request: HelpRequest) -> Alert:
        self.validate_room(request.room_id)
        if alert_id := self.manual_requests.get(request.request_id):
            alert = self.alerts[alert_id]
            if alert.room_id != request.room_id:
                raise ValueError("Request ID was already used for another room")
            return alert
        # Double-clicks/retries must not flood nurses while a manual request is pending.
        alert = next((a for a in reversed(list(self.alerts.values()))
                      if a.room_id == request.room_id and a.kind == "help_requested"
                      and a.source_type == "manual" and a.status != "resolved"), None)
        if alert is None:
            event = Event(event_id=str(uuid4()), timestamp=self.utcnow(), source="room-panel",
                          source_type="manual", room_id=request.room_id, event_type="help_requested",
                          metadata={"method": "button", "reason": "manual_help_request", "clock": "backend_receipt"})
            alert = self.open_alert(event)
        self.manual_requests[request.request_id] = alert.alert_id
        if len(self.manual_requests) > self.config.dedupe_capacity:
            self.manual_requests.popitem(last=False)
        return alert

    def nurse_snapshot(self, nurse_id: str) -> NurseSnapshot:
        self.check_health()
        nurse = self.nurses[nurse_id]
        alerts = [a for a in self.alerts.values() if self.matches_nurse(nurse, a)]
        alert_ids = {a.alert_id for a in alerts}
        events = [e for e in self.events
                  if (e.metadata.get("alert_id") or self.event_alerts.get(e.event_id)) in alert_ids]
        sources = [entry.status for entry in self.health.values()
                   if nurse.on_duty and entry.status.room_id in nurse.room_ids]
        return NurseSnapshot(nurse=nurse, alerts=alerts, events=events, sources=sources)

    def snapshot(self, room_id: str | None = None) -> RoomSnapshot:
        room_id = room_id or self.config.room_id
        self.validate_room(room_id)
        self.check_health()
        return RoomSnapshot(room_id=room_id,
                            observations=[o for o in self.observations.values() if o.room_id == room_id],
                            events=[e for e in self.events if e.room_id == room_id],
                            alerts=[a for a in self.alerts.values() if a.room_id == room_id],
                            sources=[entry.status for entry in self.health.values() if entry.status.room_id == room_id])
