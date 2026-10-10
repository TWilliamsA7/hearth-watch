"""The real monitoring core behind the evaluator's MonitorLike seam.

ServiceMonitor wraps hearth.monitor.service.MonitorService unchanged: the same
dedupe, freshness, posture rules, gesture rules, alert lifecycle and source-health
checks that run behind the live API. Only the clock is swapped: the service reads
time from the records (and from advance_to), so a 30 s run scores in milliseconds
and gives the same answer every time.

Thresholds come from hearth/monitor/config.json unless overridden:

  HEARTH_MONITOR_CONFIG=path/to/config.json   a different config file
  HEARTH_MONITOR_OVERRIDES='{"floor_hold_seconds": 2}'   individual fields on top

Environment variables (rather than factory arguments) so that study worker
processes, which only receive the "module:factory" spec, see the same settings.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from hearth_schema.models import Event, Heartbeat, Observation

from hearth.monitor.config import MonitorConfig
from hearth.monitor.service import MonitorService

CONFIG_ENV = "HEARTH_MONITOR_CONFIG"
OVERRIDES_ENV = "HEARTH_MONITOR_OVERRIDES"


def load_config(path: str | None = None, overrides: dict | None = None) -> MonitorConfig:
    config = MonitorConfig.load(path or os.environ.get(CONFIG_ENV) or None)
    if overrides is None:
        overrides = json.loads(os.environ.get(OVERRIDES_ENV) or "{}")
    if overrides:
        config = MonitorConfig.model_validate({**config.model_dump(), **overrides})
    return config


class ServiceMonitor:
    """MonitorLike over a MonitorService whose clock is the record time."""

    def __init__(self, config: MonitorConfig) -> None:
        self._now: datetime | None = None
        self.service = MonitorService(config, clock=self._seconds, utcnow=self._utcnow)
        self._pending: list[Event] = []
        self.service.event_listeners.append(self._pending.append)

    def _utcnow(self) -> datetime:
        if self._now is None:
            raise RuntimeError("ServiceMonitor used before any record set its clock")
        return self._now

    def _seconds(self) -> float:
        return self._utcnow().timestamp()

    def _set_clock(self, now: datetime) -> None:
        if self._now is None or now > self._now:
            self._now = now

    def _new_events(self) -> list[Event]:
        events = list(self._pending)
        self._pending.clear()
        return events

    def ingest(self, item: Observation | Heartbeat) -> list[Event]:
        self._set_clock(item.timestamp)
        if isinstance(item, Heartbeat):
            self.service.heartbeat(item)
        else:
            self.service.ingest([item])
        return self._new_events()

    def advance_to(self, now: datetime) -> list[Event]:
        self._set_clock(now)
        self.service.check_health()
        return self._new_events()


def create_monitor() -> ServiceMonitor:
    return ServiceMonitor(load_config())
