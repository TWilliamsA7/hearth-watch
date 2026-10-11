import pytest

from hearth_schema import Heartbeat
from hearth.monitor.config import MonitorConfig, Zone
from hearth.monitor.detector import FallDetector, track_key
from hearth.monitor.service import MonitorService
from conftest import BASE, FLOOR, UPRIGHT, observation


def falls(service):
    return [event for event in service.events if event.event_type == "possible_fall"]


def test_sustained_floor_alerts_once_then_recovers(service):
    service.ingest([observation(0, UPRIGHT), observation(.5, dict(UPRIGHT, torso_angle_deg=45,
                                                               vertical_velocity=.3))])
    state = service.detector.states[track_key(observation(0))]
    assert state.posture == "descending"
    service.ingest([observation(t) for t in (1, 1.5, 2, 2.5, 3, 3.5)])
    assert not falls(service)
    service.ingest([observation(t) for t in (4, 4.5, 5)])
    assert len(falls(service)) == len(service.alerts) == 1
    assert falls(service)[0].metadata["duration_seconds"] == 3
    service.ingest([observation(5.5, UPRIGHT), observation(6, UPRIGHT)])
    assert [e.event_type for e in service.events].count("recovered") == 1
    alert = next(iter(service.alerts.values()))
    assert alert.recovered_at is not None
    assert alert.status == "open"  # Human workflow remains independent.
    service.ingest([observation(t) for t in (6.5, 7, 7.5, 8, 8.5, 9, 9.5)])
    assert len(falls(service)) == 2


def test_people_sharing_a_frame_are_tracked_separately(service):
    """A multi-person source sends one observation per person with the same timestamp."""
    for t in (0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4):
        counts = service.ingest([observation(t, FLOOR, obs_id=f"p1-{t:g}", person_id="p1"),
                                 observation(t, UPRIGHT, obs_id=f"p2-{t:g}", person_id="p2")])
        assert counts["accepted"] == 2 and counts["out_of_order"] == 0
    assert [e.person_id for e in falls(service)] == ["p1"]
    assert service.detector.states[track_key(observation(4, person_id="p2"))].posture == "upright"


@pytest.mark.parametrize("features", [UPRIGHT,
    dict(torso_angle_deg=15, head_y=.35, hip_y=.6, bbox_aspect=.65),
    pytest.param(dict(torso_angle_deg=75, head_y=.55, hip_y=.5, bbox_aspect=1.1), marks=pytest.mark.xfail(
        strict=True, reason="Known: with the shipped floor_head_y_min=.40 / floor_hip_y_min=.45 (calibrated "
                            "for the Jetson camera), a person reclined mid-frame reads as floor-level and alerts. "
                            "It passed only while tests used the old .65 defaults.")),
    dict(torso_angle_deg=90), None])
def test_normal_or_incomplete_features_do_not_alert(service, features):
    service.ingest([observation(t / 2, features) for t in range(20)])
    assert not falls(service)


def test_brief_floor_does_not_alert(service):
    service.ingest([observation(t / 2) for t in range(6)] + [observation(3, UPRIGHT)])
    assert not falls(service)


def test_occlusion_resets_hold_and_never_recovers(service):
    service.ingest([observation(t / 2) for t in range(6)])
    service.ingest([observation(3, person_detected=False), observation(3.5), observation(4)])
    assert not falls(service)
    service.ingest([observation(t / 2) for t in range(9, 14)])
    assert len(falls(service)) == 1
    service.ingest([observation(7, None, person_detected=False)])
    assert not any(e.event_type == "recovered" for e in service.events)
    assert next(iter(service.alerts.values())).status == "open"


def test_timestamp_gap_cannot_count_as_floor_evidence(service):
    service.ingest([observation(0), observation(.5), observation(10)])
    assert not falls(service)
    service.ingest([observation(t / 2) for t in range(21, 27)])
    assert len(falls(service)) == 1


def test_face_absence_does_not_make_person_absent(service):
    service.ingest([observation(0, UPRIGHT, face_present=True),
                    observation(.5, UPRIGHT, face_present=None),
                    observation(1, UPRIGHT, face_present=False)])
    kinds = [e.event_type for e in service.events]
    assert kinds.count("face_present") == kinds.count("face_absent") == 1
    assert "person_not_visible" not in kinds


def test_duplicate_and_out_of_order_inputs_cannot_recover(service):
    service.ingest([observation(t / 2) for t in range(7)])
    result = service.ingest([observation(3), observation(2.9, UPRIGHT, obs_id="late")])
    assert result == dict(accepted=0, duplicates=1, out_of_order=1, stale=0, future=0)
    assert len(service.alerts) == 1
    assert next(iter(service.alerts.values())).recovered_at is None


def test_sources_and_runs_are_isolated(service):
    service.ingest([observation(t / 2, source_type="edge") for t in range(7)])
    service.ingest([observation(t / 2, run_id="run-2") for t in range(7)])
    assert len(falls(service)) == 2
    assert {e.source_type for e in falls(service)} == {"edge", "simulation"}
    assert len(service.detector.states) == 2


@pytest.mark.parametrize("source_type", ["edge", "simulation", "replay"])
def test_every_source_uses_same_pipeline(service, source_type):
    service.ingest([observation(t / 2, source_type=source_type) for t in range(7)])
    assert len(falls(service)) == 1
    assert falls(service)[0].metadata["duration_seconds"] == 3


def test_offline_online_emit_once_and_receipt_clock_controls_health(service, clock):
    service.ingest([observation(0)])
    clock.seconds = 3
    service.check_health()
    service.check_health()
    status = service.snapshot().sources[0]
    assert not status.online and status.observation_stale
    service.ingest([observation(.5)])  # Old scenario timestamp cannot keep source offline.
    assert service.snapshot().sources[0].online
    assert not falls(service)
    kinds = [e.event_type for e in service.events]
    assert kinds.count("source_offline") == 1 and kinds.count("source_online") == 2


def test_heartbeat_keeps_connection_online_but_observations_become_stale(service, clock):
    service.ingest([observation(0), observation(.5), observation(1)])
    clock.seconds = 2
    service.heartbeat(Heartbeat(timestamp=BASE, source="stub", source_type="simulation",
                                room_id="room-01", scenario_id="scenario", run_id="run-1"))
    clock.seconds = 3
    service.check_health()
    status = service.snapshot().sources[0]
    assert status.online and status.observation_stale
    assert service.detector.states[track_key(observation(0))].floor_since is None


def test_alert_transitions_are_idempotent_and_resolution_needs_ack(service):
    service.ingest([observation(t / 2) for t in range(7)])
    alert_id = next(iter(service.alerts))
    with pytest.raises(ValueError):
        service.transition_alert(alert_id, "resolve")
    for action in ("ack", "ack", "resolve", "resolve", "ack"):
        service.transition_alert(alert_id, action)
    assert service.alerts[alert_id].status == "resolved"
    kinds = [e.event_type for e in service.events]
    assert kinds.count("alert_acknowledged") == kinds.count("alert_resolved") == 1
    service.ingest([observation(3.5), observation(4)])
    assert len(service.alerts) == 1  # Resolution does not re-arm ongoing posture.


def test_zone_downgrades_without_hiding_event(clock):
    config = MonitorConfig(posture_mode="pose_fallback", zones=[Zone(name="bed", polygon=[(.2, .7), (.8, .7), (.8, 1), (.2, 1)])])
    service = MonitorService(config, clock=clock, utcnow=clock.utcnow)
    keypoints = [dict(name="left_hip", x=.5, y=.85, confidence=.9)]
    service.ingest([observation(t / 2, keypoints=keypoints) for t in range(7)])
    assert len(falls(service)) == len(service.alerts) == 1
    alert = next(iter(service.alerts.values()))
    assert alert.severity == "info" and alert.metadata["zone"] == "bed"


def test_zone_without_hip_location_cannot_suppress(clock):
    service = MonitorService(MonitorConfig(posture_mode="pose_fallback", zones=[Zone(name="bed", polygon=[(0, 0), (1, 0), (1, 1)])]),
                             clock=clock, utcnow=clock.utcnow)
    service.ingest([observation(t / 2) for t in range(7)])
    assert next(iter(service.alerts.values())).severity == "warning"


def test_low_confidence_hip_cannot_downgrade(clock):
    service = MonitorService(MonitorConfig(posture_mode="pose_fallback", zones=[Zone(name="bed", polygon=[(0, 0), (1, 0), (1, 1), (0, 1)])]),
                             clock=clock, utcnow=clock.utcnow)
    keypoints = [dict(name="left_hip", x=.5, y=.85, confidence=.1)]
    service.ingest([observation(t / 2, keypoints=keypoints) for t in range(7)])
    assert next(iter(service.alerts.values())).severity == "warning"


def test_offline_never_resolves_existing_alert(service, clock):
    service.ingest([observation(t / 2) for t in range(7)])
    clock.seconds = 4
    service.check_health()
    alert = next(iter(service.alerts.values()))
    assert alert.status == "open" and alert.recovered_at is None
    assert not any(e.event_type == "recovered" for e in service.events)


def test_detector_itself_ignores_old_samples():
    detector = FallDetector(MonitorConfig(posture_mode="pose_fallback"))
    detector.process(observation(1))
    assert detector.process(observation(.5, UPRIGHT)) == []
    assert detector.states[track_key(observation(0))].posture == "floor_level"


def test_bounded_event_log_and_slow_subscriber(clock):
    service = MonitorService(MonitorConfig(posture_mode="pose_fallback", event_capacity=2, websocket_queue_capacity=1),
                             clock=clock, utcnow=clock.utcnow)
    queue = service.subscribe()
    service.ingest([observation(t / 2) for t in range(7)])
    assert len(service.events) == 2
    assert queue.get_nowait() is None
    assert queue not in service.subscribers
