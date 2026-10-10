"""Shared monitor entry point for edge, simulator and replay observations."""
import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import Callable
from uuid import uuid4

from hearth_schema import (Alert, AlertMessage, Event, EventMessage, Heartbeat,
                           Observation, ObservationMessage, RoomSnapshot,
                           SourceStatus, SourceStatusMessage)

from .config import MonitorConfig
from .detector import FallDetector, SourceKey, make_event, source_key


@dataclass
class HealthEntry:
    status: SourceStatus
    last_receipt: float
    last_observation_receipt: float | None = None


class MonitorService:
    """In-memory state. Use one server worker. All methods run on its event loop."""

    def __init__(self, config: MonitorConfig, *, clock: Callable[[], float] = monotonic,
                 utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.config, self.clock, self.utcnow = config, clock, utcnow
        self.detector = FallDetector(config)
        self.events: deque[Event] = deque(maxlen=config.event_capacity)
        self.alerts: dict[str, Alert] = {}
        self.event_alerts: dict[str, str] = {}
        self.health: dict[SourceKey, HealthEntry] = {}
        self.observations: dict[SourceKey, Observation] = {}
        self.seen: OrderedDict[tuple, None] = OrderedDict()
        self.subscribers: set[asyncio.Queue] = set()

    def publish(self, message) -> None:
        payload = message.model_dump(mode="json")
        for queue in tuple(self.subscribers):
            if queue.full():
                # A slow dashboard must reconnect and obtain a fresh snapshot.
                self.subscribers.discard(queue)
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(None)
            else:
                queue.put_nowait(payload)

    def subscribe(self) -> asyncio.Queue:
        queue = asyncio.Queue(maxsize=self.config.websocket_queue_capacity)
        self.subscribers.add(queue)
        return queue

    def record(self, event: Event) -> None:
        self.events.append(event)
        self.publish(EventMessage(data=event))

    def validate_room(self, room_id: str) -> None:
        if room_id != self.config.room_id:
            raise ValueError(f"Only room {self.config.room_id!r} is configured")

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
        entry.last_receipt, entry.status.last_seen = now, received_at
        entry.status.online = True
        if observation:
            entry.last_observation_receipt = now
            entry.status.last_observation_at = received_at
            entry.status.observation_stale = False
        if isinstance(record, Heartbeat):
            for name in ("fps", "inference_ms", "mem_used_mb"):
                setattr(entry.status, name, getattr(record, name))
        if changed:
            self.record(make_event(record, "source_online", timestamp=received_at,
                                   clock="backend_receipt"))
        if changed or was_stale != entry.status.observation_stale or not observation:
            self.publish(SourceStatusMessage(data=entry.status))

    def check_health(self) -> None:
        now = self.clock()
        for key, entry in self.health.items():
            changed = False
            if (not entry.status.observation_stale and entry.last_observation_receipt is not None
                    and now - entry.last_observation_receipt >= self.config.source_timeout_seconds):
                entry.status.observation_stale = True
                self.detector.invalidate(key)
                changed = True
            if entry.status.online and now - entry.last_receipt >= self.config.source_timeout_seconds:
                entry.status.online = False
                entry.status.observation_stale = True
                self.detector.invalidate(key)
                self.record(make_event(entry.status, "source_offline", timestamp=self.utcnow(),
                                       clock="backend_receipt"))
                changed = True
            if changed:
                self.publish(SourceStatusMessage(data=entry.status))

    def ingest(self, batch: list[Observation]) -> dict[str, int]:
        # Validate the complete batch before applying anything.
        for obs in batch:
            self.validate_room(obs.room_id)
        self.check_health()
        counts = {"accepted": 0, "duplicates": 0, "out_of_order": 0}
        for obs in batch:
            key = source_key(obs)
            identity = (*key, obs.obs_id)
            if identity in self.seen:
                counts["duplicates"] += 1
                continue
            previous = self.observations.get(key)
            if previous is not None and obs.timestamp <= previous.timestamp:
                counts["out_of_order"] += 1
                continue
            self.seen[identity] = None
            if len(self.seen) > self.config.dedupe_capacity:
                self.seen.popitem(last=False)
            self.touch(obs, observation=True)
            self.observations[key] = obs
            self.publish(ObservationMessage(data=obs))
            for event in self.detector.process(obs):
                self.record(event)
                if event.event_type == "possible_fall":
                    alert = Alert(alert_id=str(uuid4()), event_id=event.event_id,
                                  source=obs.source, source_type=obs.source_type, room_id=obs.room_id,
                                  scenario_id=obs.scenario_id, run_id=obs.run_id,
                                  opened_at=event.timestamp, severity=event.metadata["severity"],
                                  metadata=event.metadata.copy())
                    self.alerts[alert.alert_id] = alert
                    self.event_alerts[event.event_id] = alert.alert_id
                    self.publish(AlertMessage(data=alert))
                elif event.event_type == "recovered":
                    alert_id = self.event_alerts.get(event.metadata["incident_event_id"])
                    if alert_id:
                        alert = self.alerts[alert_id]
                        alert.recovered_at = event.timestamp
                        self.publish(AlertMessage(data=alert))
            counts["accepted"] += 1
        return counts

    def heartbeat(self, heartbeat: Heartbeat) -> None:
        self.validate_room(heartbeat.room_id)
        self.check_health()
        self.touch(heartbeat, observation=False)

    def transition_alert(self, alert_id: str, action: str) -> Alert:
        alert = self.alerts[alert_id]
        if action == "ack":
            if alert.status in {"acknowledged", "resolved"}:
                return alert
            alert.status, alert.acknowledged_at = "acknowledged", self.utcnow()
            kind = "alert_acknowledged"
        elif action == "resolve":
            if alert.status == "resolved":
                return alert
            if alert.status != "acknowledged":
                raise ValueError("Acknowledge the alert before resolving it")
            alert.status, alert.resolved_at = "resolved", self.utcnow()
            kind = "alert_resolved"
        else:
            raise ValueError("Unknown alert action")
        self.record(make_event(alert, kind, timestamp=self.utcnow(), alert_id=alert_id,
                               incident_event_id=alert.event_id, clock="backend_receipt"))
        self.publish(AlertMessage(data=alert))
        return alert

    def snapshot(self) -> RoomSnapshot:
        self.check_health()
        return RoomSnapshot(room_id=self.config.room_id, observations=list(self.observations.values()),
                            events=list(self.events), alerts=list(self.alerts.values()),
                            sources=[entry.status for entry in self.health.values()])
