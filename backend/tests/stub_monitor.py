"""Test double for hearth.monitor. NOT the real monitor; its rules are deliberately crude.

It exists to exercise the MonitorLike seam end to end (hearth/eval/batch.py): ordered
ingest, virtual-time silence detection via advance_to, and event shapes the evaluator
scores. The real possible-fall logic belongs to the monitoring core in hearth/monitor.
"""

from datetime import datetime, timedelta
from itertools import count

from hearth_schema.models import Event, Heartbeat, Observation

OFFLINE_AFTER = timedelta(seconds=3)


class StubMonitor:
    def __init__(self) -> None:
        self._ids = count(1)
        self._last_seen: dict[str, datetime] = {}
        self._offline: set[str] = set()
        self._visible: bool | None = None
        self._fast_drop_at: datetime | None = None
        self._floor_since: datetime | None = None
        self._alerting = False

    def _event(self, item, event_type, ts=None, **metadata) -> Event:
        return Event(event_id=f"stub-{next(self._ids)}", timestamp=ts or item.timestamp, source=item.source,
                     source_type=item.source_type, room_id=item.room_id,
                     scenario_id=getattr(item, "scenario_id", None), event_type=event_type, metadata=metadata)  # fmt: skip

    def ingest(self, item: Observation | Heartbeat) -> list[Event]:
        out = []
        self._last_seen[item.source] = item.timestamp
        self._last_item = item
        if item.source in self._offline:
            self._offline.discard(item.source)
            out.append(self._event(item, "source_online"))
        if isinstance(item, Observation):
            out += self._observe(item)
        return out

    def _observe(self, obs: Observation) -> list[Event]:
        out = []
        if obs.person_detected != self._visible:
            self._visible = obs.person_detected
            out.append(self._event(obs, "person_visible" if obs.person_detected else "person_not_visible"))
        f = obs.features
        if f is None or f.torso_angle_deg is None or f.hip_y is None:
            return out
        if (f.vertical_velocity or 0.0) > 0.35:
            self._fast_drop_at = obs.timestamp
        on_floor = f.torso_angle_deg > 60 and f.hip_y > 0.75
        if not on_floor:
            self._floor_since = None
            if self._alerting and f.torso_angle_deg < 30:
                self._alerting = False
                out.append(self._event(obs, "recovered"))
            return out
        self._floor_since = self._floor_since or obs.timestamp
        recent_drop = self._fast_drop_at and obs.timestamp - self._fast_drop_at < timedelta(seconds=4)
        if not self._alerting and recent_drop and obs.timestamp - self._floor_since >= timedelta(seconds=1.5):
            self._alerting = True
            out.append(self._event(obs, "possible_fall"))
        return out

    def advance_to(self, now: datetime) -> list[Event]:
        out = []
        for source, seen in self._last_seen.items():
            if source not in self._offline and now - seen > OFFLINE_AFTER:
                self._offline.add(source)
                item = self._last_item
                out.append(Event(event_id=f"stub-{next(self._ids)}", timestamp=now, source=source,
                                 source_type=item.source_type, room_id=item.room_id, event_type="source_offline"))  # fmt: skip
        return out


def create_monitor() -> StubMonitor:
    return StubMonitor()
