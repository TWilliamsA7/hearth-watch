"""Temporal posture rules. This module sees sensor observations only."""
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from hearth_schema import Event, Observation

from .config import MonitorConfig

SourceKey = tuple[str, str, str, str | None, str | None]


def source_key(record) -> SourceKey:
    return (record.room_id, record.source_type, record.source, record.scenario_id, record.run_id)


def make_event(record, kind: str, timestamp: datetime | None = None, **metadata) -> Event:
    return Event(event_id=str(uuid4()), timestamp=timestamp or record.timestamp,
                 source=record.source, source_type=record.source_type, room_id=record.room_id,
                 scenario_id=record.scenario_id, run_id=record.run_id,
                 event_type=kind, metadata=metadata)


def inside_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        # Include the boundary.
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) < 1e-9 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
        previous = current
    return inside


@dataclass
class PostureState:
    posture: str = "unknown"
    last_timestamp: datetime | None = None
    floor_since: datetime | None = None
    incident_event_id: str | None = None
    person_visible: bool | None = None
    face_visible: bool | None = None


class FallDetector:
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.states: dict[SourceKey, PostureState] = {}

    def invalidate(self, key: SourceKey) -> None:
        """Drop pending evidence on gaps; never recover an existing incident from silence."""
        if state := self.states.get(key):
            state.posture = "unknown"
            state.floor_since = None
            state.person_visible = None
            state.face_visible = None

    def process(self, obs: Observation) -> list[Event]:
        key = source_key(obs)
        state = self.states.setdefault(key, PostureState())
        if state.last_timestamp is not None:
            gap = (obs.timestamp - state.last_timestamp).total_seconds()
            if gap <= 0:
                return []
            if gap > self.config.max_observation_gap_seconds:
                self.invalidate(key)
        state.last_timestamp = obs.timestamp
        events = []
        if obs.person_detected != state.person_visible:
            state.person_visible = obs.person_detected
            events.append(make_event(obs, "person_visible" if obs.person_detected else "person_not_visible"))
        # Face absence does not override a positive person/pose detection.
        if obs.face_present is not None and obs.face_present != state.face_visible:
            state.face_visible = obs.face_present
            events.append(make_event(obs, "face_present" if obs.face_present else "face_absent"))

        f, c = obs.features, self.config
        if not obs.person_detected or f is None:
            state.posture, state.floor_since = "unknown", None
            return events
        upright = (f.torso_angle_deg is not None and f.torso_angle_deg <= c.upright_angle_max_deg
                   and f.head_y is not None and f.hip_y is not None
                   and f.head_y < f.hip_y < c.floor_hip_y_min)
        floor = (f.torso_angle_deg is not None
                 and c.floor_angle_min_deg <= f.torso_angle_deg <= c.floor_angle_max_deg
                 and f.head_y is not None and f.head_y >= c.floor_head_y_min
                 and f.hip_y is not None and f.hip_y >= c.floor_hip_y_min
                 and (f.bbox_aspect is None or f.bbox_aspect >= c.floor_bbox_aspect_min))
        if upright:
            if state.incident_event_id:
                events.append(make_event(obs, "recovered", incident_event_id=state.incident_event_id))
                state.incident_event_id = None
            state.posture, state.floor_since = "upright", None
        elif floor:
            if state.incident_event_id:
                state.posture = "possible_fall"
                return events
            # A missing descent does not make sustained floor posture safe.
            if state.floor_since is None:
                state.floor_since = obs.timestamp
            state.posture = "floor_level"
            duration = (obs.timestamp - state.floor_since).total_seconds()
            if duration >= c.floor_hold_seconds:
                metadata = {"posture": "floor_level", "duration_seconds": duration,
                            "reason": "sustained_floor_level", "severity": "warning"}
                hips = [p for p in obs.keypoints or []
                        if p.name in {"left_hip", "right_hip"}
                        and p.confidence >= c.zone_keypoint_confidence_min]
                if hips:
                    x, y = sum(p.x for p in hips) / len(hips), sum(p.y for p in hips) / len(hips)
                    for zone in c.zones:
                        if inside_polygon(x, y, zone.polygon):
                            metadata.update(severity="info", zone=zone.name,
                                            downgrade_reason="floor_posture_in_configured_" + zone.kind)
                            break
                event = make_event(obs, "possible_fall", **metadata)
                state.incident_event_id, state.posture = event.event_id, "possible_fall"
                events.append(event)
        else:
            state.floor_since = None
            state.posture = ("descending" if f.vertical_velocity is not None
                             and f.vertical_velocity >= c.descending_velocity_min else "unknown")
        return events
