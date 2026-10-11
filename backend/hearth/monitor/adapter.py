"""Single-record adapter with an explicit clock for headless sensor evaluation."""
from datetime import datetime, timezone

from hearth_schema import Heartbeat, Observation

from .config import MonitorConfig
from .service import MonitorService


class MonitorAdapter:
    def __init__(self, config: MonitorConfig | None = None):
        self.now = datetime(1970, 1, 1, tzinfo=timezone.utc)
        self.service = MonitorService(config or MonitorConfig.load(),
                                      clock=lambda: self.now.timestamp(), utcnow=lambda: self.now)
        self._events = []
        self.service.event_listeners.append(self._events.append)

    def _capture(self, action):
        # Capture publications, rather than slicing the bounded event log.
        self._events.clear()
        try:
            action()
            return self._events.copy()
        finally:
            self._events.clear()

    def ingest(self, item: Observation | Heartbeat):
        self.now = max(self.now, item.timestamp)
        return self._capture(lambda: self.service.heartbeat(item) if isinstance(item, Heartbeat)
                             else self.service.ingest([item]))

    def advance_to(self, now: datetime):
        if now < self.now:
            raise ValueError("Monitor clock cannot move backwards")
        self.now = now
        return self._capture(self.service.check_health)
