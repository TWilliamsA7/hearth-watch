from datetime import datetime, timedelta, timezone

import pytest

from hearth_schema import Observation
from hearth.monitor.config import MonitorConfig
from hearth.monitor.service import MonitorService

BASE = datetime(2026, 10, 10, tzinfo=timezone.utc)
UPRIGHT = dict(torso_angle_deg=5, head_y=.2, hip_y=.5, bbox_aspect=.4)
FLOOR = dict(torso_angle_deg=90, head_y=.85, hip_y=.85, bbox_aspect=1.8)


def observation(t, features=FLOOR, **changes):
    data = dict(obs_id=f"obs-{t:g}", timestamp=BASE + timedelta(seconds=t),
                source="stub", source_type="simulation", room_id="room-01",
                scenario_id="scenario", run_id="run-1", person_detected=True, features=features)
    data.update(changes)
    return Observation(**data)


class Clock:
    seconds = 0.0

    def __call__(self):
        return self.seconds

    def utcnow(self):
        return BASE + timedelta(seconds=self.seconds)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def service(clock):
    return MonitorService(MonitorConfig(), clock=clock, utcnow=clock.utcnow)
